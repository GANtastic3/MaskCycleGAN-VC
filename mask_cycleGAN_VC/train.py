import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import pickle

from cycleGAN_VC3.model import Generator, Discriminator
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from dataset.dataset import Dataset
from dataset.vc_dataset import trainingDataset
from cycleGAN_VC3.utils import get_audio_transforms, data_processing, decode_melspectrogram, get_img_from_fig, get_waveform_fig, get_mel_spectrogram_fig
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class CycleGANTraining(object):
    def __init__(self, args):
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.decay_after = args.decay_after
        self.mini_batch_size = args.batch_size
        self.cycle_loss_lambda = args.cycle_loss_lambda
        self.identity_loss_lambda = args.identity_loss_lambda
        self.device = args.device
        self.epochs_per_save = args.epochs_per_save
        self.epochs_per_plot = args.epochs_per_plot

        self.vocoder = torch.hub.load(
            'descriptinc/melgan-neurips', 'load_melgan')
        self.sample_rate = args.sample_rate

        self.dataset_A = self.loadPickleFile(args.normalized_dataset_A_path)
        dataset_A_norm_stats = np.load(args.norm_stats_A_path)
        # TODO: fix to mean and std after running data preprocessing script again
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']
        self.dataset_B = self.loadPickleFile(args.normalized_dataset_B_path)
        dataset_B_norm_stats = np.load(args.norm_stats_B_path)
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        self.n_samples = len(self.dataset_A)
        print(f'n_samples = {self.n_samples}')
        self.generator_lr_decay = self.generator_lr / float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        self.discriminator_lr_decay = self.discriminator_lr / float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        print(f'generator_lr_decay = {self.generator_lr_decay}')
        print(f'discriminator_lr_decay = {self.discriminator_lr_decay}')
        self.num_frames = args.num_frames
        self.dataset = trainingDataset(datasetA=self.dataset_A,
                                       datasetB=self.dataset_B,
                                       n_frames=args.num_frames,
                                       max_mask_len=args.max_mask_len)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                            batch_size=self.mini_batch_size,
                                                            shuffle=True,
                                                            drop_last=False)

        self.validation_dataset = trainingDataset(datasetA=self.dataset_A,
                                                  datasetB=self.dataset_B,
                                                  n_frames=args.num_frames_validation,
                                                  max_mask_len=args.max_mask_len,
                                                  valid=True)
        self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 drop_last=False)

        self.logger = TrainLogger(args, len(self.train_dataloader.dataset))
        self.saver = ModelSaver(args)

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)
        self.discriminator_A2 = Discriminator().to(self.device)
        self.discriminator_B2 = Discriminator().to(self.device)

        # Optimizer
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters()) + \
            list(self.discriminator_A2.parameters()) + \
            list(self.discriminator_B2.parameters())

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # Load from previous ckpt
        if args.continue_train:
            self.saver.load_model(
                self.generator_A2B, "generator_A2B", None, self.generator_optimizer)
            self.saver.load_model(self.generator_B2A,
                                  "generator_B2A", None, None)
            self.saver.load_model(self.discriminator_A, 
                                  "discriminator_A", None, self.discriminator_optimizer)
            self.saver.load_model(self.discriminator_B,
                                  "discriminator_B", None, None)
            self.saver.load_model(self.discriminator_A2, 
                                  "discriminator_A2", None, None)
            self.saver.load_model(self.discriminator_B2,
                                  "discriminator_B2", None, None)

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.start_epoch()

            for i, (real_A, mask_A, real_B, mask_B) in enumerate(tqdm(self.train_dataloader)):
                self.logger.start_iter()
                num_iterations = (
                    self.n_samples // self.mini_batch_size) * epoch + i

                real_A = real_A.to(self.device, dtype=torch.float)
                mask_A = mask_A.to(self.device, dtype=torch.float)
                real_B = real_B.to(self.device, dtype=torch.float)
                mask_B = mask_B.to(self.device, dtype=torch.float)

                # Train Generator
                fake_B = self.generator_A2B(real_A, mask_A)
                cycle_A = self.generator_B2A(fake_B, torch.ones_like(fake_B))
                fake_A = self.generator_B2A(real_B, mask_B)
                cycle_B = self.generator_A2B(fake_A, torch.ones_like(fake_A))
                identity_A = self.generator_B2A(real_A, torch.ones_like(real_A))
                identity_B = self.generator_A2B(real_B, torch.ones_like(real_B))
                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                # for the second step adverserial loss
                d_fake_cycle_A = self.discriminator_A2(cycle_A)
                d_fake_cycle_B = self.discriminator_B2(cycle_B)

                # Generator Cycle Loss
                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                # Generator Identity Loss
                identityLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                g_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                g_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # Generator second step adverserial loss
                generator_loss_A2B_2nd = torch.mean((1 - d_fake_cycle_B) ** 2)
                generator_loss_B2A_2nd = torch.mean((1 - d_fake_cycle_A) ** 2)

                # Total Generator Loss
                g_loss = g_loss_A2B + g_loss_B2A + \
                    generator_loss_A2B_2nd + generator_loss_B2A_2nd + \
                    self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss
                # self.generator_loss_store.append(generator_loss.item())

                # Backprop for Generator
                self.reset_grad()
                g_loss.backward()
                self.generator_optimizer.step()

                # Train Discriminator

                # Discriminator Feed Forward
                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)

                d_real_A2 = self.discriminator_A2(real_A)
                d_real_B2 = self.discriminator_B2(real_B)

                generated_A = self.generator_B2A(real_B, mask_B)
                d_fake_A = self.discriminator_A(generated_A)

                # For Second Step Adverserial Loss A->B
                cycled_B = self.generator_A2B(generated_A, torch.ones_like(generated_A))
                d_cycled_B = self.discriminator_B2(cycled_B)

                generated_B = self.generator_A2B(real_A, mask_A)
                d_fake_B = self.discriminator_B(generated_B)

                # For Second Step Adverserial Loss B->A
                cycled_A = self.generator_B2A(generated_B, torch.ones_like(generated_B))
                d_cycled_A = self.discriminator_A2(cycled_A)

                # Loss Functions
                d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # Second Step Adverserial Loss
                d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
                d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                d_loss_A2_real = torch.mean((1 - d_real_A2) ** 2)
                d_loss_B2_real = torch.mean((1 - d_real_B2) ** 2)
                d_loss_A_2nd = (d_loss_A2_real + d_loss_A_cycled) / 2.0
                d_loss_B_2nd = (d_loss_B2_real + d_loss_B_cycled) / 2.0

                # Final Loss for discriminator with the second step adverserial loss
                d_loss = (d_loss_A + d_loss_B) / 2.0 + \
                    (d_loss_A_2nd + d_loss_B_2nd) / 2.0
                # self.discriminator_loss_store.append(d_loss.item())

                # Backprop for Discriminator
                self.reset_grad()
                d_loss.backward()
                self.discriminator_optimizer.step()

                # if num_iterations % args.steps_per_print == 0:
                #     print(f"Epoch: {epoch} Step: {num_iterations} Generator Loss: {generator_loss.item()} Discriminator Loss: {d_loss.item()}")

                self.logger.log_iter(
                    loss_dict={'g_loss': g_loss.item(), 'd_loss': d_loss.item()})

                self.logger.end_iter()
                # adjust learning rates
                if self.logger.global_step > self.decay_after:
                    self.identity_loss_lambda = 0
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')

            if self.logger.epoch % self.epochs_per_plot == 0:
                # Log spectrograms
                real_mel_A_fig = get_mel_spectrogram_fig(
                    real_A[0].detach().cpu())
                fake_mel_A_fig = get_mel_spectrogram_fig(
                    generated_A[0].detach().cpu())
                real_mel_B_fig = get_mel_spectrogram_fig(
                    real_B[0].detach().cpu())
                fake_mel_B_fig = get_mel_spectrogram_fig(
                    generated_B[0].detach().cpu())
                self.logger.visualize_outputs({"real_voc_spec": real_mel_A_fig, "fake_coraal_spec": fake_mel_B_fig,
                                               "real_coraal_spec": real_mel_B_fig, "fake_voc_spec": fake_mel_A_fig})

                # Decode spec->wav
                real_wav_A = decode_melspectrogram(self.vocoder, real_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                fake_wav_A = decode_melspectrogram(self.vocoder, generated_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                real_wav_B = decode_melspectrogram(self.vocoder, real_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                fake_wav_B = decode_melspectrogram(self.vocoder, generated_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()

                # # Log wav
                # real_wav_A_fig = get_waveform_fig(real_wav_A, self.sample_rate)
                # fake_wav_A_fig = get_waveform_fig(fake_wav_A, self.sample_rate)
                # real_wav_B_fig = get_waveform_fig(real_wav_B, self.sample_rate)
                # fake_wav_B_fig = get_waveform_fig(fake_wav_B, self.sample_rate)
                # self.logger.visualize_outputs({"real_voc_wav": real_wav_A_fig, "fake_coraal_wav": fake_wav_B_fig,
                #                                "real_coraal_wav": real_wav_B_fig, "fake_voc_wav": fake_wav_A_fig})

                # Convert spectrograms from validation set to wav and log to tensorboard
                real_mel_full_A, real_mel_full_B = next(
                    iter(self.validation_dataloader))
                real_mel_full_A = real_mel_full_A.to(
                    self.device, dtype=torch.float)
                real_mel_full_B = real_mel_full_B.to(
                    self.device, dtype=torch.float)
                fake_mel_full_B = self.generator_A2B(real_mel_full_A, torch.ones_like(real_mel_full_A))
                fake_mel_full_A = self.generator_B2A(real_mel_full_B, torch.ones_like(real_mel_full_B))
                real_wav_full_A = decode_melspectrogram(self.vocoder, real_mel_full_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                fake_wav_full_A = decode_melspectrogram(self.vocoder, fake_mel_full_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                real_wav_full_B = decode_melspectrogram(self.vocoder, real_mel_full_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                fake_wav_full_B = decode_melspectrogram(self.vocoder, fake_mel_full_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                self.logger.log_audio(
                    real_wav_full_A.T, "real_voc_audio", self.sample_rate)
                self.logger.log_audio(fake_wav_full_A.T, "fake_voc_audio", self.sample_rate)
                self.logger.log_audio(
                    real_wav_full_B.T, "real_coraal_audio", self.sample_rate)
                self.logger.log_audio(fake_wav_full_B.T, "fake_coraal_audio", self.sample_rate)

            if self.logger.epoch % self.epochs_per_save == 0:
                self.saver.save(self.logger.epoch, self.generator_A2B,
                                self.generator_optimizer, None, args.device, "generator_A2B")
                self.saver.save(self.logger.epoch, self.generator_B2A,
                                self.generator_optimizer, None, args.device, "generator_B2A")
                self.saver.save(self.logger.epoch, self.discriminator_A,
                                self.discriminator_optimizer, None, args.device, "discriminator_A")
                self.saver.save(self.logger.epoch, self.discriminator_B,
                                self.discriminator_optimizer, None, args.device, "discriminator_B")
                self.saver.save(self.logger.epoch, self.discriminator_A2,
                                self.discriminator_optimizer, None, args.device, "discriminator_A2")
                self.saver.save(self.logger.epoch, self.discriminator_B2,
                                self.discriminator_optimizer, None, args.device, "discriminator_B2")

            self.logger.end_epoch()


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()
    cycleGAN = CycleGANTraining(args)
    cycleGAN.train()
