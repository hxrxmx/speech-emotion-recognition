import torch
import torch.nn.functional as F
import torch_audiomentations as A
from torch import nn
from torchaudio.transforms import AmplitudeToDB, Resample


class WaveformPreprocessingPipeline(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        max_sample_time=4.0,
        aug_transform=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.length = int(sample_rate * max_sample_time)
        self.aug_transform = aug_transform

    def _resample(self, waveform, orig_rate):
        if orig_rate != self.sample_rate:
            resampler = Resample(orig_freq=orig_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        return waveform

    def _pad_or_trim(self, waveform):
        current_length = waveform.size(-1)
        if current_length < self.length:
            waveform = F.pad(waveform, (0, self.length - current_length))
        elif current_length > self.length:
            waveform = waveform[..., : self.length]
        return waveform

    def _zero_shift(self, wf):
        wf = wf - wf.mean()
        return wf

    def forward(self, waveform_sr):
        waveform = self._resample(*waveform_sr)
        waveform = self._zero_shift(waveform)
        if self.aug_transform:
            waveform = self.aug_transform(waveform)
        waveform = self._pad_or_trim(waveform)
        return waveform


class MelSpecPreprocessingPipeline(nn.Module):
    def __init__(self, top_db, noise_offset_db, n_target_time_frames):
        super().__init__()
        self.top_db = top_db
        self.noise_offset_db = noise_offset_db
        self.n_target_time_frames = n_target_time_frames
        self.ampl_to_db = AmplitudeToDB(top_db=self.top_db)

    def _trim_noise(self, mel_spec):
        mel_spec = torch.where(
            mel_spec < self.noise_offset_db, torch.min(mel_spec), mel_spec
        )
        return mel_spec

    def _spectrogram_normalization(self, mel_spec):
        return (mel_spec - mel_spec.min()) / (mel_spec.std() + 1e-6)

    def _resize_mel_spec_length(self, mel_spec):
        mel_spec_resized = F.interpolate(
            mel_spec,
            size=(self.n_target_time_frames,),
            mode="linear",
            align_corners=True,
        )
        return mel_spec_resized

    def forward(self, mel_spec):
        mel_spec = self.ampl_to_db(mel_spec)
        mel_spec = self._trim_noise(mel_spec)
        mel_spec = self._spectrogram_normalization(mel_spec)
        if self.n_target_time_frames:
            mel_spec = self._resize_mel_spec_length(mel_spec)
        return mel_spec


class AudioAugmentationsPipeline(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        min_shift=-0.5,
        max_shift=0.5,
        min_pitch_shift_semitones=-1,
        max_pitch_shift_semitones=1,
        min_gain_db=-3.0,
        max_gain_db=3.0,
        hp_min_cutoff_freq=100,
        hp_max_cutoff_freq=800,
        lp_min_cutoff_freq=2500,
        lp_max_cutoff_freq=8000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.pipeline = A.Compose(
            [
                A.PitchShift(
                    min_transpose_semitones=min_pitch_shift_semitones,
                    max_transpose_semitones=max_pitch_shift_semitones,
                    sample_rate=sample_rate,
                    p=0.3,
                    output_type="tensor",
                ),
                A.AddColoredNoise(
                    p=0.2,
                    output_type="tensor",
                ),
                A.PolarityInversion(
                    p=0.5,
                    output_type="tensor",
                ),
                A.Gain(
                    min_gain_in_db=min_gain_db,
                    max_gain_in_db=max_gain_db,
                    p=0.5,
                    output_type="tensor",
                ),
                A.Shift(
                    min_shift=min_shift,
                    max_shift=max_shift,
                    p=0.8,
                    output_type="tensor",
                ),
                A.HighPassFilter(
                    min_cutoff_freq=hp_min_cutoff_freq,
                    max_cutoff_freq=hp_max_cutoff_freq,
                    p=0.3,
                    output_type="tensor",
                ),
                A.LowPassFilter(
                    min_cutoff_freq=lp_min_cutoff_freq,
                    max_cutoff_freq=lp_max_cutoff_freq,
                    p=0.3,
                    output_type="tensor",
                ),
                A.AddColoredNoise(p=0.1, output_type="tensor"),
            ],
            p=1.0,
            output_type="tensor",
        )

    def forward(self, waveform):
        augmented = self.pipeline(
            waveform.unsqueeze(1), sample_rate=self.sample_rate
        ).squeeze(1)
        return augmented
