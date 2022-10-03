from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):
    """To update the learning rate after a step, use the method `step()`. """
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        super(TransformerLR, self).__init__(optimizer, last_epoch)
        self.optimizer = optimizer
        self.d_model = d_model    # embedding dim
        self.warmup_steps = warmup_steps

    def get_lr(self):
        dynamic_lr = self.d_model**(-0.5) * min(self._step_count**(-0.5),
                                                self._step_count * (self.warmup_steps**-1.5))
        return [dynamic_lr for group in self.optimizer.param_groups]