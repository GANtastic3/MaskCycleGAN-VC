import numpy as np
from tqdm import tqdm
import os
import pandas as pd

import librosa
import torch
import pickle

from cycleGAN_VC3.model import Generator
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from saver.model_saver import ModelSaver


class CycleGANGenerate(object):
    def __init__(self, args):
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.mini_batch_size = args.batch_size
        self.device = args.device

        self.vocoder = torch.hub.load(
            'descriptinc/melgan-neurips', 'load_melgan')
        self.sample_rate = args.sample_rate

        self.data_dir = args.data_dir
        self.source_id = args.source_id
        self.save_dir = args.save_dir
        self.saver = ModelSaver(args)

        # Generator
        self.generator_A2B = Generator().to(self.device)

        # Load from previous ckpt
        self.saver.load_model(self.generator_A2B, "generator_A2B",
                              args.ckpt_path, None, None)

        voc_wav_files = self.read_manifest(dataset="voc", speaker_id=self.source_id)
        print(f'Found {len(voc_wav_files)} wav files')
        self.dataset_A, self.dataset_A_mean, self.dataset_A_std = self.normalize_mel(voc_wav_files, self.data_dir, sr=self.sample_rate)
        self.n_samples = len(self.dataset_A)
        print(f'n_samples = {self.n_samples}')

    def read_manifest(self, split=None, dataset=None, speaker_id=None):
        # Load manifest file which defines dataset
        manifest_path = os.path.join('./manifests', f'{dataset}_manifest.csv')
        df = pd.read_csv(manifest_path, sep=',')

        # Filter by speaker_id
        df['speaker_id'] = df['speaker_id'].astype(str)
        df = df[df['speaker_id'] == speaker_id]
        wav_files = df['wav_file'].tolist()

        return wav_files

    def normalize_mel(self, wav_files, data_dir, sr=22050):
        vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        mel_list = dict()
        for wavpath in tqdm(wav_files, desc='Preprocess wav to mel'):
            wav_orig, _ = librosa.load(os.path.join(data_dir, wavpath), sr=sr, mono=True)
            spec = vocoder(torch.tensor([wav_orig]))
            assert wavpath not in mel_list
            mel_list[wavpath] = spec.cpu().detach().numpy()[0]

        mel_concatenated = np.concatenate(list(mel_list.values()), axis=1)
        mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
        mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

        mel_normalized = dict()
        for wavpath, mel in mel_list.items():
            app = (mel - mel_mean) / mel_std
            assert wavpath not in mel_normalized
            mel_normalized[wavpath] = app

        return mel_normalized, mel_mean, mel_std
    
    def save_pickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def run(self):
        
        converted_specs = dict()
        for i, (wavpath, melspec) in enumerate(tqdm(self.dataset_A.items())):
            real_A = torch.tensor(melspec).unsqueeze(0).to(self.device, dtype=torch.float)
            fake_B_normalized = self.generator_A2B(real_A, torch.ones_like(real_A)).squeeze(0).detach().cpu().numpy()
            fake_B = fake_B_normalized * self.dataset_A_std + self.dataset_A_mean
            converted_specs[wavpath] = fake_B
        
        print(f"Saving to ~/data/converted/voc_converted_{self.source_id}.pickle")
        self.save_pickle(variable=converted_specs,
                fileName=os.path.join('/home/ubuntu/data', "converted", f"voc_converted_{self.source_id}.pickle"))


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()
    cycleGAN = CycleGANGenerate(args)
    cycleGAN.run()
