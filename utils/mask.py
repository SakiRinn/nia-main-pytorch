import torch


def key_padding_mask(seq: torch.Tensor, padding_value=0.0):
    # seq: (N, T)
    mask = torch.eq(seq, torch.tensor(padding_value)).to(torch.float32)
    return mask    # (N, T)


def square_subsequent_mask(size: int):
    # size: The length of target.
    mask = torch.triu(torch.ones((size, size)).to(torch.bool), diagonal=1)
    return mask    # (size, size)