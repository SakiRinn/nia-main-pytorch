def get_model(name: str, *args, **kwargs):
    import modules

    if hasattr(modules, name):
        try:
            return getattr(modules, name)(*args, **kwargs)
        except TypeError:
            return getattr(modules, name)()
    raise AttributeError(f'There is no class named {name}.')


def get_loss(name: str, *args, **kwargs):
    import torch.nn
    import modules.loss

    if hasattr(torch.nn, name):
        try:
            return getattr(torch.nn, name)(*args, **kwargs)
        except TypeError:
            return getattr(torch.nn, name)()
    elif hasattr(modules.loss, name):
        try:
            return getattr(modules.loss, name)(*args, **kwargs)
        except TypeError:
            return getattr(modules.loss, name)()
    raise AttributeError(f'There is no class named {name}.')


def get_activation(name: str, *args, **kwargs):
    import torch.nn

    if hasattr(torch.nn, name):
        try:
            return getattr(torch.nn, name)(*args, **kwargs)
        except TypeError:
            return getattr(torch.nn, name)()
    raise AttributeError(f'There is no class named {name}.')


def get_optimizer(name: str, params, lr, *args, **kwargs):
    import torch.optim
    import modules.optimizer

    if hasattr(torch.optim, name):
        try:
            return getattr(torch.optim, name)(params, lr, *args, **kwargs)
        except TypeError:
            return getattr(torch.optim, name)(params, lr)
    elif hasattr(modules.optimizer, name):
        try:
            return getattr(modules.optimizer, name)(params, lr, *args, **kwargs)
        except TypeError:
            return getattr(modules.optimizer, name)(params, lr)
    raise AttributeError(f'There is no class named {name}.')


def get_lr_scheduler(name: str, *args, **kwargs):
    import modules.optimizer

    if hasattr(modules.optimizer, name):
        try:
            return getattr(modules.optimizer, name)(*args, **kwargs)
        except TypeError:
            return getattr(modules.optimizer, name)()
    raise AttributeError(f'There is no class named {name}.')
