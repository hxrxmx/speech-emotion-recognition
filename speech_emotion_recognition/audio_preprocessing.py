import torch
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, Resample


class WaveformPreprocessingPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        max_sample_time=5.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.length = int(sample_rate * max_sample_time)

    def _resamaple(self, waveform, orig_rate):
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
        waveform = self._resamaple(*waveform_sr)
        waveform = self._normalizing_waveform(waveform)
        waveform = self._pad_or_trim(waveform)
        return waveform


class MelSpecPreprocessingPipeline(torch.nn.Module):
    def __init__(
        self,
        top_db=60.0,
        noise_offset_db=-20.0,
    ):
        super().__init__()
        self.top_db = top_db
        self.noise_offset_db = noise_offset_db
        self.ampl_to_db = AmplitudeToDB(top_db=self.top_db)

    def _trim_noise(self, spectrogram):
        noise_floor = torch.median(spectrogram)
        threshold = noise_floor + self.noise_offset_db
        spectrogram = torch.where(
            spectrogram < threshold, torch.zeros_like(spectrogram), spectrogram
        )
        return spectrogram

    def _spectrogram_normalization(self, spectrogram):
        return spectrogram / torch.std(spectrogram + 1e-6)

    def forward(self, spectrogram):
        spectrogram = self._spectrogram_normalization(spectrogram)
        spectrogram = self.ampl_to_db(spectrogram)
        spectrogram = self._trim_noise(spectrogram)
        return spectrogram
