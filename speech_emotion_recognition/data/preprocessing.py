import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import AmplitudeToDB, Resample


class WaveformPreprocessingPipeline(nn.Module):
    def __init__(self, sample_rate, max_sample_time, aug_transform=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.length = int(sample_rate * max_sample_time)
        self.aug_transform = aug_transform

    def resample(self, waveform, orig_rate):
        if orig_rate != self.sample_rate:
            resampler = Resample(orig_freq=orig_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        return waveform

    def pad_or_trim(self, waveform):
        current_length = waveform.size(-1)
        if current_length < self.length:
            waveform = F.pad(waveform, (0, self.length - current_length))
        else:
            waveform = waveform[..., : self.length]
        return waveform

    def forward(self, waveform_sr):
        waveform = self.resample(*waveform_sr)
        waveform = self.pad_or_trim(waveform)
        if self.aug_transform:
            waveform = self.aug_transform(waveform)
        return waveform


class MelSpecPreprocessingPipeline(nn.Module):
    def __init__(self, top_db, noise_offset_db, n_target_time_frames):
        super().__init__()
        self.top_db = top_db
        self.noise_offset_db = noise_offset_db
        self.n_target_time_frames = n_target_time_frames
        self.ampl_to_db = AmplitudeToDB(top_db=self.top_db)

    def trim_noise(self, spec):
        spec = spec - spec.min()
        spec = torch.where(
            spec < self.noise_offset_db, 0.0, spec - self.noise_offset_db
        )
        return spec

    def normalize(self, spec):
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec

    def resize_length(self, spec):
        spec = F.interpolate(
            spec,
            size=(self.n_target_time_frames,),
            mode="linear",
            align_corners=True,
        )
        return spec

    def forward(self, spec):
        spec = self.ampl_to_db(spec)
        spec = self.trim_noise(spec)
        spec = self.resize_length(spec)
        spec = self.normalize(spec)
        return spec
