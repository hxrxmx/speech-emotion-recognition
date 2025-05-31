from torch import nn


class FullPipeline(nn.Module):
    def __init__(self, waveform_pipeline, mel_spec, spec_pipeline, model):
        super().__init__()
        self.waveform_pipeline = waveform_pipeline
        self.mel_spec = mel_spec
        self.spec_pipeline = spec_pipeline
        self.model = model

    def forward(self, waveform, sr):
        tensor = self.waveform_pipeline((waveform, sr))
        tensor = self.mel_spec(tensor)
        tensor = self.spec_pipeline(tensor)
        return self.model(tensor)
