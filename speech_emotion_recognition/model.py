import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class EmotionSpeechClassifier(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.lr = lr

        self.acc = MulticlassAccuracy(6)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inp):
        return self.model(inp)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        data, logits = batch
        predictions = self(data)

        loss = self.loss_fn(predictions, logits)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, logits = batch
        predictions = self(data)

        loss = self.loss_fn(predictions, logits)
        acc = self.acc(torch.softmax(predictions, dim=1), logits)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, logits = batch
        predictions = self(data)

        acc = self.acc(torch.softmax(predictions, dim=1), logits)

        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, logits = batch
        predictions = self(data)

        return torch.argmax(predictions)
