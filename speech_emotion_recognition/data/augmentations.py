import torch_audiomentations as A
from torch import nn


class AudioAugmentationsPipeline(nn.Module):
    def __init__(
        self,
        sample_rate,
        min_shift,
        max_shift,
        min_pitch_shift_semitones,
        max_pitch_shift_semitones,
        min_gain_db,
        max_gain_db,
        hp_min_cutoff_freq,
        hp_max_cutoff_freq,
        lp_min_cutoff_freq,
        lp_max_cutoff_freq,
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
                A.AddColoredNoise(p=0.2, output_type="tensor"),
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
                    rollover=False,
                    p=1.0,
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
        augmented = self.pipeline(waveform.unsqueeze(0)).squeeze(0)
        return augmented
