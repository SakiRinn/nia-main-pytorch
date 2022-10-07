from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):
    """To update the learning rate after a step, use the method `step()`. """
    def __init__(self, optimizer, d_model, start_steps=0, *, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model    # embedding dim
        self.start_steps = start_steps
        self.warmup_steps = warmup_steps

        super(TransformerLR, self).__init__(optimizer)

    def get_lr(self):
        value = min((self.start_steps + self._step_count)**(-0.5),
                    (self.start_steps + self._step_count) * (self.warmup_steps**-1.5))
        dynamic_lr = self.d_model**(-0.5) * value
        return [dynamic_lr for group in self.optimizer.param_groups]