import hydra
import lightning as L
from classifier import AudioClassifier
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import EmotionSpeechClassifier
from omegaconf import DictConfig

from data import CREMADataModule


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    dm = CREMADataModule(config)

    model = EmotionSpeechClassifier(
        AudioClassifier(num_classes=config.model.num_classes),
        config=config,
    )

    loggers = [
        WandbLogger(
            project=config.logging.project,
            name=config.logging.name,
            save_dir=config.logging.save_dir,
        )
    ]

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/cnn_with_transformer_enc/",
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            filename="model-{epoch:02d}-{val_loss:.4f}-{val_acc:.3f}",
            save_last=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
