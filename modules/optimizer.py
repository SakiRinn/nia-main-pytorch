import torch


class TransformerLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1, verbose=False):
        super(TransformerLR, self).__init__(optimizer, last_epoch, verbose)
        self.optimizer = optimizer
        self.d_model = d_model    # embedding dim
        self.warmup_steps = warmup_steps

    def step(self):
        dynamic_lr = self.d_model**(-0.5) * min(self._step_count**(-0.5),
                                                self._step_count * (self.warmup_steps**-1.5))
        return [dynamic_lr for group in self.optimizer.param_groups]