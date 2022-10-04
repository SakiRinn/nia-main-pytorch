import torch
import torch.nn as nn


class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, mask):
        # mask: Tensor[bool], trg_key_padding_mask
        loss = self.cross_entropy(input, target)
        return (loss * torch.logical_not(mask)).mean()