# -*- coding: utf-8 -*-
"""
Preprocesses .wav to Mel-spectrograms using Mel-GAN vocoder and saves them to pickle files.
MelGAN vocoder: https://github.com/descriptinc/melgan-neurips
"""

import os
import argparse
import pickle
import glob
import random
import numpy as np
from tqdm import tqdm

import librosa
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

SAMPLING_RATE = 22050  # Fixed sampling rate


def normalize_mel(wavspath):
    wav_files = glob.glob(os.path.join(
        wavspath, '**', '*.wav'), recursive=True)  # source_path
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    mel_list = list()
    for wavpath in tqdm(wav_files, desc='Preprocess wav to mel'):
        wav_orig, _ = librosa.load(wavpath, sr=SAMPLING_RATE, mono=True)
        spec = vocoder(torch.tensor([wav_orig]))

        if spec.shape[-1] >= 64:    # training sample consists of 64 randomly cropped frames
            mel_list.append(spec.cpu().detach().numpy()[0])

    mel_concatenated = np.concatenate(mel_list, axis=1)
    mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
    mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

    mel_normalized = list()
    for mel in mel_list:
        assert mel.shape[-1] >= 64, f"Mel spectogram length must be greater than 64 frames, but was {mel.shape[-1]}"
        app = (mel - mel_mean) / mel_std
        mel_normalized.append(app)

    return mel_normalized, mel_mean, mel_std


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_dataset(data_path, speaker_id, cache_folder='./cache/'):
    """Preprocesses dataset of .wav files by converting to Mel-spectrograms.

    Args:
        data_path (str): Directory containing .wav files of the speaker.
        speaker_id (str): ID of the speaker.
        cache_folder (str, optional): Directory to hold preprocessed data. Defaults to './cache/'.
    """

    print(f"Preprocessing data for speaker: {speaker_id}.")

    mel_normalized, mel_mean, mel_std = normalize_mel(data_path)

    if not os.path.exists(os.path.join(cache_folder, speaker_id)):
        os.makedirs(os.path.join(cache_folder, speaker_id))

    np.savez(os.path.join(cache_folder, speaker_id, f"{speaker_id}_norm_stat.npz"),
             mean=mel_mean,
             std=mel_std)

    save_pickle(variable=mel_normalized,
                fileName=os.path.join(cache_folder, speaker_id, f"{speaker_id}_normalized.pickle"))

    print(f"Preprocessed and saved data for speaker: {speaker_id}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_directory', type=str, default='vcc2018/vcc2018_training',
                        help='Directory holding VCC2018 dataset.')
    parser.add_argument('--preprocessed_data_directory', type=str, default='vcc2018_preprocessed/vcc2018_training',
                        help='Directory holding preprocessed VCC2018 dataset.')
    parser.add_argument('--speaker_ids', nargs='+', type=str, default=['VCC2SM3', 'VCC2TF1'],
                        help='Source speaker id from VCC2018.')

    args = parser.parse_args()

    for speaker_id in args.speaker_ids:
        data_path = os.path.join(args.data_directory, speaker_id)
        preprocess_dataset(data_path=data_path, speaker_id=speaker_id,
                           cache_folder=args.preprocessed_data_directory)
