from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.loggers import WandbLogger


class LocalPlot(L.Callback):
    def __init__(
        self,
        class_names=None,
    ):
        super().__init__()
        self.class_names = class_names

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
        }

        self.metrics_to_plot = {
            "Loss": {"train_loss": "Train Loss", "val_loss": "Validation Loss"},
            "Accuracy": {
                "train_acc": "Train Accuracy",
                "val_acc": "Validation Accuracy",
            },
            "F1 Score": {"val_f1": "Validation F1 Score"},
        }

        self.default_plot_dir = Path("plots/")

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        wandb_logger = None
        if isinstance(trainer.logger, WandbLogger):
            wandb_logger = trainer.logger
        elif isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break

        if wandb_logger and wandb_logger.experiment:
            self.plot_dir = Path(wandb_logger.experiment.dir) / "plots"
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"Metrics plots into: {self.plot_dir}")
        else:
            print(
                f"Warning: WandbLogger not found. "
                f"Graphs will be saved to default directory '{self.default_plot_dir}'."
            )
            self.plot_dir = self.default_plot_dir
            self.plot_dir.mkdir(self.plot_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.history["train_loss"].append(
                trainer.callback_metrics["train_loss"].item()
            )
        if "train_acc" in trainer.callback_metrics:
            self.history["train_acc"].append(
                trainer.callback_metrics["train_acc"].item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.history["val_loss"].append(trainer.callback_metrics["val_loss"].item())
        if "val_acc" in trainer.callback_metrics:
            self.history["val_acc"].append(trainer.callback_metrics["val_acc"].item())
        if "val_f1" in trainer.callback_metrics:
            self.history["val_f1"].append(trainer.callback_metrics["val_f1"].item())

        if not self.history["val_loss"]:
            return

        epochs_range = np.arange(trainer.current_epoch)

        for metric_group, metrics_map in self.metrics_to_plot.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for history_key, plot_label in metrics_map.items():
                if history_key in self.history and self.history[history_key]:
                    if len(self.history[history_key]) >= trainer.current_epoch:
                        ax.plot(
                            epochs_range,
                            self.history[history_key][: trainer.current_epoch],
                            label=plot_label,
                        )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_group)
            ax.set_title(f"{metric_group} over {trainer.current_epoch} epochs")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            filename = "metrics.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath)

            plt.close(fig)
