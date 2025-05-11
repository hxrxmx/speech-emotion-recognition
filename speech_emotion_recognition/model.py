import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from speech_emotion_recognition.loss import FocalLoss


class EmotionSpeechClassifier(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.config = config

        self.model = model

        weights = torch.tensor(list(config.model.weights.values()), dtype=torch.float)
        self.loss = FocalLoss(weight=weights)

        self.acc = MulticlassAccuracy(config.model.num_classes)
        self.f1 = MulticlassF1Score(config.model.num_classes)
        self.conf_mat = MulticlassConfusionMatrix(config.model.num_classes)

        self.val_preds = []
        self.val_targets = []

    def forward(self, tensor):
        return self.model(tensor)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.training.sheduler.factor,
            patience=self.config.training.sheduler.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions_logits = self(data)

        loss = self.loss(predictions_logits, targets)
        acc = self.acc(predictions_logits, targets)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("lr", lr, prog_bar=True, on_step=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions_logits = self(data)

        loss = self.loss(predictions_logits, targets)
        acc = self.acc(predictions_logits, targets)
        f1 = self.f1(predictions_logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        preds = predictions_logits.argmax(dim=1)
        self.val_preds.append(preds)
        self.val_targets.append(targets)

        return {
            "val_loss": loss,
            "val_acc": acc,
            "val_f1": f1,
        }

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_preds).cpu().numpy()
        all_targets = torch.cat(self.val_targets).cpu().numpy()

        class_labels = [str(i) for i in range(self.config.model.num_classes)]

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    "val_conf_mat": wandb.plot.confusion_matrix(
                        preds=all_preds, y_true=all_targets, class_names=class_labels
                    ),
                }
            )

        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions = self(data)

        loss = self.loss(predictions, targets)
        acc = self.acc(predictions, targets)
        f1 = self.f1(predictions, targets)

        self.log("test_acc", acc)
        self.log("test_f1", f1)
        self.log("test_loss", loss)
        return {
            "test_loss": loss,
            "test_acc": acc,
            "test_f1": f1,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, _ = batch
        predictions = self(data)

        return torch.argmax(predictions, dim=1)
