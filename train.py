import utils
from datasets import ResDataset
import logging
import argparse

from torch.utils.data import DataLoader
import torch


def train(resume=''):
    # Config
    model_cfg = utils.load_config('./configs/model.yaml')
    run_cfg = utils.load_config('./configs/run.yaml')
    device = run_cfg['device']
    end_epoch = run_cfg['epoch']

    # Dataset
    dataset = ResDataset(device)
    data_loader = DataLoader(dataset,
                             batch_size=run_cfg['train']['batch_size'],
                             num_workers=4,
                             shuffle=True,
                             pin_memory=True)

    # Model
    model_name = run_cfg['model']
    model_params = model_cfg[model_name]
    if model_name == 'seq2seq':
        teacher_forcing_ratio = model_params['teacher_forcing_ratio']
        del model_params['teacher_forcing_ratio']
    model = utils.get_model(run_cfg['model'])().to(device)

    # Resume
    start_epoch = 0
    if resume != '':
        checkpoint, start_epoch = utils.find_checkpoint(resume)
        model.load_state_dict(checkpoint)

    # Train
    for epoch in range(start_epoch, end_epoch):
        for step, src, trg in enumerate(data_loader):
            ...


if __name__ == "__main__":
    train()
