def get_model(name: str, *args, **kwargs):
    import torch.nn
    if hasattr(torch.nn, name.capitalize()):
        try:
            return getattr(torch.nn, name.capitalize())(*args, **kwargs)
        except TypeError:
            return getattr(torch.nn, name.capitalize())()
    raise AttributeError(f'There is no class named {name}.')


def get_loss(name: str, *args, **kwargs):
    import torch.nn
    import modules.loss
    if hasattr(torch.nn, name.capitalize()):
        try:
            return getattr(torch.nn, name.capitalize())(*args, **kwargs)
        except TypeError:
            return getattr(torch.nn, name.capitalize())()
    elif hasattr(modules.loss, name.capitalize()):
        try:
            return getattr(modules.loss, name.capitalize())(*args, **kwargs)
        except TypeError:
            return getattr(modules.loss, name.capitalize())()
    raise AttributeError(f'There is no class named {name}.')


def get_activation(name: str, *args, **kwargs):
    import torch.nn
    if hasattr(torch.nn, name.capitalize()):
        try:
            return getattr(torch.nn, name.capitalize())(*args, **kwargs)
        except TypeError:
            return getattr(torch.nn, name.capitalize())()
    raise AttributeError(f'There is no class named {name}.')


def get_optimizer(name: str, *args, **kwargs):
    import torch.optim
    import modules.optimizer
    if hasattr(torch.optim, name.capitalize()):
        try:
            return getattr(torch.optim, name.capitalize())(*args, **kwargs)
        except TypeError:
            return getattr(torch.optim, name.capitalize())()
    elif hasattr(modules.optimizer, name.capitalize()):
        try:
            return getattr(modules.optimizer, name.capitalize())(*args, **kwargs)
        except TypeError:
            return getattr(modules.optimizer, name.capitalize())()
    raise AttributeError(f'There is no class named {name}.')
