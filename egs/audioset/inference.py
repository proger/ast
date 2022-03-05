# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: a inference script for single audio, heavily base on demo.py and traintest.py
import os
import sys
import csv
import argparse
import subprocess
import wave
from contextlib import contextmanager
import json

import numpy as np
import torch
import torchaudio
from loguru import logger

#torchaudio.set_audio_backend("soundfile")       # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


@contextmanager
def mkwave(filename, sr=16000):
    with subprocess.Popen(["ffmpeg", '-loglevel', 'quiet', "-i", filename,
                          "-f", "wav", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-"],
                          stdout=subprocess.PIPE) as p:
        with wave.open(p.stdout) as wf:
            yield wf
    if p.returncode != 0:
        raise Exception('data y u no convert')

class MediaError(Exception):
    pass

def ffprobe_duration(filename: str) -> float:
    c = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split() + [str(filename)]
    try:
        return float(subprocess.check_output(c))
    except:
        raise MediaError('not media')


def make_features(wav_name, mel_bins, target_length=1024):
    data = b''
    with mkwave(wav_name) as wf:
        while (chunk := wf.readframes(4000)):
            data += chunk

    waveform = torch.frombuffer(data, dtype=torch.int16)
    waveform = waveform / torch.iinfo(torch.int16).max
    waveform = waveform.unsqueeze(0)

    #waveform, sr = torchaudio.load(wav_name)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    parser.add_argument("--model_path", type=str,
                        default='/home/proger/ast/pretrained_models/audioset_10_10_0.4593.pth',
                        help="the trained model you want to test")
    parser.add_argument('--output', type=str, required=True, help='where to write json output')
    parser.add_argument('audio_paths', nargs='+',
                        help='the audio you want to predict',
                        type=str)

    args = parser.parse_args()

    label_csv = './data/class_labels_indices.csv'       # label and indices for audioset data

    torch.inference_mode(True)

    # 2. load the best model and the weights
    checkpoint_path = args.model_path
    ast_mdl = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)

    logger.info(f'checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)

    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()


    def infer(audio_path):
        try:
            total_duration = ffprobe_duration(audio_path)
        except MediaError:
            return

        feats = make_features(audio_path, mel_bins=128)           # shape(1024, 128)

        # assume each input spectrogram has 100 time frames
        input_tdim = feats.shape[0]
        assert input_tdim == 1024, input_tdim

        feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature

        with torch.no_grad():
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()[0]

        # 4. map the post-prob to label
        labels = load_label(label_csv)

        sorted_indexes = np.argsort(result_output)[::-1]

        return json.dumps({
            'filename': audio_path,
            'total_duration': total_duration,
            'audioset_top10': {
                np.array(labels)[sorted_indexes[k]].lower(): float(result_output[sorted_indexes[k]]) for k in range(10)
            }
        })


    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(32)
    from tqdm import tqdm
    import random

    with open(args.output, 'a') as output:
        tasks = [executor.submit(infer, audio_path) for audio_path in args.audio_paths]
        for fu in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            try:
                if result := fu.result():
                    print(result, file=output, flush=random.random() < 0.1)
            except:
                logger.exception('some task crashed')
