import torch
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, Resample


class WaveformPreprocessingPipeline(torch.nn.Module):
    def __init__(self, sample_rate=16000, max_sample_time=5.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.length = int(sample_rate * max_sample_time)

    def _resample(self, waveform, orig_rate):
        resampler = Resample(orig_freq=orig_rate, new_freq=self.sample_rate)
        return resampler(waveform)

    def _normalizing_waveform(self, waveform):
        return (waveform - torch.mean(waveform)) / torch.std(waveform)

    def _pad_or_trim(self, waveform):
        current_length = waveform.size(-1)
        if current_length < self.length:
            waveform = F.pad(waveform, (0, self.length - current_length))
        elif current_length > self.length:
            waveform = waveform[..., : self.length]
        return waveform

    def forward(self, waveform_sr):
        waveform = self._resample(*waveform_sr)
        waveform = self._normalizing_waveform(waveform)
        waveform = self._pad_or_trim(waveform)
        return waveform


class MelSpecPreprocessingPipeline(torch.nn.Module):
    def __init__(self, top_db=60.0, noise_offset_db=-20.0, n_target_time_frames=None):
        super().__init__()
        self.top_db = top_db
        self.noise_offset_db = noise_offset_db
        self.n_target_time_frames = n_target_time_frames
        self.ampl_to_db = AmplitudeToDB(top_db=self.top_db)

    def _trim_noise(self, mel_spec):
        noise_floor = torch.median(mel_spec)
        threshold = noise_floor + self.noise_offset_db
        mel_spec = torch.where(
            mel_spec < threshold, torch.zeros_like(mel_spec), mel_spec
        )
        return mel_spec

    def _spectrogram_normalization(self, mel_spec):
        return mel_spec / (mel_spec.std() + 1e-6)

    def _resize_mel_spec_length(self, mel_spec):
        mel_spec_resized = F.interpolate(
            mel_spec.unsqueeze(0),
            size=(mel_spec.shape[1], self.n_target_time_frames),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return mel_spec_resized

    def forward(self, mel_spec):
        mel_spec = self._spectrogram_normalization(mel_spec)
        mel_spec = self.ampl_to_db(mel_spec)
        mel_spec = self._trim_noise(mel_spec)
        if self.n_target_time_frames:
            mel_spec = self._resize_mel_spec_length(mel_spec)
        return mel_spec
