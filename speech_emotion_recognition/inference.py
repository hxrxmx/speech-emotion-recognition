import fire
import lightning as L
import torch
from hydra import compose, initialize

from speech_emotion_recognition.core.classifier import AudioClassifier
from speech_emotion_recognition.core.model import EmotionSpeechClassifier
from speech_emotion_recognition.data.inference_data import AudioPredictDataModule


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

    model = EmotionSpeechClassifier.load_from_checkpoint(
        config.inference.ckpt_path,
        model=AudioClassifier(config.model.num_classes),
        config=config,
    )
    model.eval()

    dm = AudioPredictDataModule(config, paths.split())

    trainer = L.Trainer(accelerator="auto", devices="auto", logger=False)
    pred = trainer.predict(model, datamodule=dm)
    pred = torch.cat(pred)

    print(f"Predicted classes: {pred.tolist()}")


if __name__ == "__main__":
    fire.Fire(predict)
