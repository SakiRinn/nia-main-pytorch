def get_model(name: str):
    import modules
    return eval('modules.' + name)


def get_loss(name: str):
    import torch.nn as nnLoss
    return eval('nnLoss.' + name)()


def get_activation(name: str):
    import torch.nn as nnAct
    if name == 'Softmax':
        return nnAct.Softmax(dim=-1)
    return eval('nnAct.' + name)()


def get_optimizer(name: str):
    import torch.optim as optim
    return eval('optim.' + name)


def get_lr_scheduler(name: str):
    import modules.optimizer as optim_
    return eval('optim_.' + name)