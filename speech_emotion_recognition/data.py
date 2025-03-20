import os

import lightning as pl
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram

import speech_emotion_recognition.audio_preprocessing as ap


class CREMADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.classes = sorted(os.listdir(data_dir))
        self.cls_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = [
            (os.path.join(data_dir, cls, file), self.cls_to_idx[cls])
            for cls in self.classes
            for file in os.listdir(os.path.join(data_dir, cls))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        waveform, sample_rate = torchaudio.load(file_path)
        sample_rate = torch.tensor(sample_rate)
        label = torch.tensor(label)

        if self.transform:
            audio_item = self.transform((waveform, sample_rate))
        else:
            audio_item = waveform

        return audio_item, label


class CREMADataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.transform = nn.Sequential(
            ap.WaveformPreprocessingPipeline(
                sample_rate=config.data.preprocessing.sample_rate,
                max_sample_time=config.data.preprocessing.max_sample_time,
            ),
            MelSpectrogram(
                sample_rate=config.data.preprocessing.sample_rate,
                n_fft=config.data.preprocessing.n_fft,
                hop_length=config.data.preprocessing.hop_length,
                f_min=config.data.preprocessing.f_min,
                f_max=config.data.preprocessing.f_max,
                n_mels=config.data.preprocessing.n_mels,
            ),
            ap.MelSpecPreprocessingPipeline(
                top_db=config.data.preprocessing.top_db,
                noise_offset_db=config.data.preprocessing.noise_offset_db,
            ),
        )

    def setup(self, stage=None):
        self.train_dataset = CREMADataset(
            self.config.data.data_loading.train_data_path,
            self.transform,
        )
        self.test_dataset = CREMADataset(
            self.config.data.data_loading.test_data_path,
            self.transform,
        )
        self.val_dataset = CREMADataset(
            self.config.data.data_loading.val_data_path,
            self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
        )
