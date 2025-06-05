from torch import nn
from torchaudio.transforms import MelSpectrogram

from speech_emotion_recognition.data.preprocessing import (
    MelSpecPreprocessingPipeline,
    WaveformPreprocessingPipeline,
)


class InferencePreprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

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

    def forward(self, tensor):
        return self.transform(tensor)
