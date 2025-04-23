import hydra
import lightning as L
from classifier import AudioClassifier
from model import EmotionSpeechClassifier
from omegaconf import DictConfig

from data import CREMADataModule


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    dm = CREMADataModule(config)

    model = EmotionSpeechClassifier.load_from_checkpoint(
        config.inference.ckpt_path,
        model=AudioClassifier(config.model.num_classes),
        config=config,
    )
    trainer = L.Trainer(accelerator="auto", devices="auto")

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
