"""
Defines the util functions associated with the cycleGAN VC pipeline.
"""
import io
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import ToTensor

import librosa
import librosa.display

from tqdm import tqdm


import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def decode_melspectrogram(vocoder, melspectrogram, mel_mean, mel_std):
    denorm_converted = melspectrogram * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted.unsqueeze(0))
    return rev


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def get_audio_transforms(phase, sample_rate=16000, n_mels=36):
    if phase == 'train':
        transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=n_mels, hop_length=2048//4, n_fft=2048),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
        )
    elif phase == 'valid':
        transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=2048//4, n_fft=2048
        )

    return transforms


def data_processing(data, phase, n_mels=36):
    spectrograms_A = []
    spectrograms_B = []

    for (input_A, input_B) in data:
        (waveform_A, sample_rate_A, _, _, _) = input_A
        audio_transforms_A = get_audio_transforms(phase, sample_rate_A, n_mels)
        spec_A = audio_transforms_A(waveform_A).squeeze(0)[
            :, :128].transpose(0, 1)
        print(f'spec_A shape is {spec_A.shape}')
        spectrograms_A.append(spec_A)

        (waveform_B, sample_rate_B, _, _, _) = input_B
        audio_transforms_B = get_audio_transforms(phase, sample_rate_B, n_mels)
        spec_B = audio_transforms_B(waveform_B).squeeze(0)[
            :, :128].transpose(0, 1)
        print(f'spec_B shape is {spec_B.shape}')
        spectrograms_B.append(spec_B)

    spectrograms_A = nn.utils.rnn.pad_sequence(
        spectrograms_A, batch_first=True).transpose(1, 2)
    spectrograms_B = nn.utils.rnn.pad_sequence(
        spectrograms_B, batch_first=True).transpose(1, 2)

    return (spectrograms_A, spectrograms_B)


def get_img_from_fig(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = canvas.buffer_rgba()
    return np.asarray(buf)


def get_waveform_fig(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
        
    image = Image.open(buf)
    image = ToTensor()(image)
    
    plt.close(figure)
    
    return image

def get_mel_spectrogram_fig(spec, title="Mel-Spectrogram"):
    figure, ax = plt.subplots()
    canvas = FigureCanvas(figure)
    S_db = librosa.power_to_db(10**spec.numpy().squeeze(), ref=np.max)
    img = librosa.display.specshow(S_db, ax=ax, y_axis='log', x_axis='time')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
        
    image = Image.open(buf)
    image = ToTensor()(image)
    
    plt.close(figure)
    return image

    
            
            
        
    
