import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma

        if weight is not None:
            self.weight = weight / weight.sum() * weight.numel()
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.weight.to(logits.device) if self.weight is not None else None,
        )
        probs = torch.exp(-ce_loss)
        focal_loss = (1 - probs) ** self.gamma * ce_loss

        return focal_loss.mean()
