import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class EmotionSpeechClassifier(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.config = config

        self.model = model
        self.lr = config.training.lr

        self.acc = MulticlassAccuracy(config.model.num_classes)

        weights = torch.tensor(list(config.model.weights.values()), dtype=torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inp):
        return self.model(inp)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            factor=self.config.training.sheduler.factor,
            patience=self.config.training.sheduler.patience,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions = self(data)

        loss = self.loss_fn(predictions, targets)
        acc = self.acc(predictions, targets)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("lr", lr, prog_bar=True, on_step=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions = self(data)

        loss = self.loss_fn(predictions, targets)
        acc = self.acc(predictions, targets)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions = self(data)

        loss = self.loss_fn(predictions, targets)
        acc = self.acc(predictions, targets)

        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        predictions = self(data)

        return torch.argmax(predictions, dim=1)
