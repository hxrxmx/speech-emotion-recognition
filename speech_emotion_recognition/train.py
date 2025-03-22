import hydra
import lightning as L
from classifier import AudioClassifier
from model import EmotionSpeechClassifier
from omegaconf import DictConfig

from data import CREMADataModule


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    dm = CREMADataModule(config)
    model = EmotionSpeechClassifier(
        AudioClassifier(num_classes=config.model.num_classes),
        lr=config.training.lr,
    )

    loggers = []
    callbacks = []

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        log_every_n_steps=1,
        accelerator="gpu",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
