import torch
import torch.nn as nn


class MaskedLoss(nn.Module):
    def __init__(self, loss_fn):
        super(MaskedLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target, mask):
        # mask: Tensor[bool], trg_key_padding_mask
        loss = self.loss_fn(input, target)
        return (loss * torch.logical_not(mask)).mean()