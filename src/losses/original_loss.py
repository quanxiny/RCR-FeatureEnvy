import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """The focal BCE loss used by the published CG-LSMN implementation."""

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(loss) if self.reduce else loss

