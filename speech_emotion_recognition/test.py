import fire
import lightning as L
from hydra import compose, initialize

from speech_emotion_recognition.classifier import AudioClassifier
from speech_emotion_recognition.data import CREMADataModule
from speech_emotion_recognition.model import EmotionSpeechClassifier


def test(
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

    dm = CREMADataModule(config)

    trainer = L.Trainer(accelerator="auto", devices="auto", logger=False)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    fire.Fire(test)
