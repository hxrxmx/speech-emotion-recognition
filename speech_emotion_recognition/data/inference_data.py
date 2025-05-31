from typing import Iterable

import lightning as L
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram

from speech_emotion_recognition.data.preprocessing import (
    MelSpecPreprocessingPipeline,
    WaveformPreprocessingPipeline,
)


class AudioPredictDataset(Dataset):
    def __init__(self, paths, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path = self.samples[index]
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if self.transform:
            tensor = self.transform((waveform, sr))
        else:
            tensor = waveform

        return tensor


class AudioPredictDataModule(L.LightningDataModule):
    def __init__(self, config, predict_paths: Iterable[str]):
        super().__init__()
        self.paths = predict_paths

        self.config = config

        self._transform_setup()

    def _transform_setup(self):
        waveform_transform = WaveformPreprocessingPipeline(
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

        self.transform = nn.Sequential(
            waveform_transform,
            mel_spec,
            mel_spec_transform,
        )

    def setup(self, stage=None):
        self.dataset = AudioPredictDataset(
            self.paths,
            self.transform,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
