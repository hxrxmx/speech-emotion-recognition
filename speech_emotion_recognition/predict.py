from pathlib import Path

import fire
import torch
import torchaudio
from hydra import compose, initialize

from speech_emotion_recognition.inference.loading import load_model
from speech_emotion_recognition.inference.preprocessing import InferencePreprocessing


def predict(
    paths: str,
    ckpt_path: str = None,
    config_path: str = "../conf",
    config_name: str = "config",
):
    with initialize(config_path=config_path, version_base=None):
        config = compose(config_name=config_name)
    if ckpt_path:
        config.inference.ckpt_path = ckpt_path

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config.inference.ckpt_path, device, config.model.num_classes)

    transform = InferencePreprocessing(config)

    spectrograms = []
    for path in paths.split():
        path = Path(path)
        waveform, sr = torchaudio.load(path)

        spec = transform((waveform, sr)).unsqueeze(0).to(device)
        spectrograms.append(spec)

    batch = batch = torch.cat(spectrograms, dim=0)

    with torch.no_grad():
        logits = model(batch)
        preds = torch.argmax(logits, dim=1).tolist()

    print("Predicted classes:", preds)


if __name__ == "__main__":
    fire.Fire(predict)
