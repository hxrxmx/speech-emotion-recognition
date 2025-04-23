import os

import lightning as pl
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram

from speech_emotion_recognition.preprocessing import (
    AudioAugmentationsPipeline,
    MelSpecPreprocessingPipeline,
    WaveformPreprocessingPipeline,
)


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
        waveform, sr = torchaudio.load(file_path)

        if self.transform:
            tensor = self.transform((waveform, sr))
        else:
            tensor = waveform

        return tensor, label


class CREMADataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self._transforms_setup()

    def _transforms_setup(self):
        aug_transform = AudioAugmentationsPipeline(
            sample_rate=self.config.data.preprocessing.sample_rate,
            min_shift=self.config.data.augmentations.min_shift,
            max_shift=self.config.data.augmentations.max_shift,
            min_pitch_shift_semitones=self.config.data.augmentations.min_pitch_shift_st,
            max_pitch_shift_semitones=self.config.data.augmentations.max_pitch_shift_st,
            min_gain_db=self.config.data.augmentations.min_gain_db,
            max_gain_db=self.config.data.augmentations.max_gain_db,
            lp_min_cutoff_freq=self.config.data.augmentations.lp_min_cutoff_freq,
            lp_max_cutoff_freq=self.config.data.preprocessing.f_max,
            hp_min_cutoff_freq=self.config.data.preprocessing.f_min,
            hp_max_cutoff_freq=self.config.data.augmentations.hp_max_cutoff_freq,
        )

        train_waveform_transform = WaveformPreprocessingPipeline(
            sample_rate=self.config.data.preprocessing.sample_rate,
            max_sample_time=self.config.data.preprocessing.max_sample_time,
            aug_transform=aug_transform,
        )

        val_waveform_transform = WaveformPreprocessingPipeline(
            sample_rate=self.config.data.preprocessing.sample_rate,
            max_sample_time=self.config.data.preprocessing.max_sample_time,
            aug_transform=None,
        )

        mel_spec = MelSpectrogram(
            sample_rate=self.config.data.preprocessing.sample_rate,
            n_fft=self.config.data.preprocessing.n_fft,
            hop_length=self.config.data.preprocessing.hop_length,
            win_length=self.config.data.preprocessing.win_length,
            f_min=self.config.data.preprocessing.f_min,
            f_max=self.config.data.preprocessing.f_max,
            n_mels=self.config.data.preprocessing.n_mels,
        )

        mel_spec_transform = MelSpecPreprocessingPipeline(
            top_db=self.config.data.preprocessing.top_db,
            noise_offset_db=self.config.data.preprocessing.noise_offset_db,
            n_target_time_frames=self.config.data.preprocessing.n_target_time_frames,
        )

        self.train_transform = nn.Sequential(
            train_waveform_transform,
            mel_spec,
            mel_spec_transform,
        )

        self.val_transform = nn.Sequential(
            val_waveform_transform,
            mel_spec,
            mel_spec_transform,
        )

        self.test_transform = nn.Sequential(
            val_waveform_transform,
            mel_spec,
            mel_spec_transform,
        )

    def setup(self, stage=None):
        self.train_dataset = CREMADataset(
            data_dir=self.config.data.data_loading.train_data_path,
            transform=self.train_transform,
        )
        self.val_dataset = CREMADataset(
            data_dir=self.config.data.data_loading.val_data_path,
            transform=self.val_transform,
        )
        self.test_dataset = CREMADataset(
            data_dir=self.config.data.data_loading.test_data_path,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            persistent_workers=True,
        )
