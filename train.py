import utils.getter as getter
import utils.fileIO as fileIO
import utils.mask as mask
from modules.optimizer import TransformerLR
from modules.loss import MaskedLoss
from datasets import ResDataset

import random
import numpy as np
import os
from tqdm import trange
import argparse

import torch
from torch.utils.data import DataLoader


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train(resume=''):
    # Config
    dataset_cfg = fileIO.load_config('./configs/dataset.yaml')
    model_cfg = fileIO.load_config('./configs/model.yaml')
    run_cfg = fileIO.load_config('./configs/run.yaml')

    logger, exp_id = fileIO.init_train(run_cfg['checkpoint_dir'])
    exp_dir = os.path.join(run_cfg['checkpoint_dir'], f"experiment_{exp_id}")
    device = run_cfg['device']
    end_epoch = run_cfg['train']['epochs']
    lr = run_cfg['train']['lr']['initial']
    lr_scheduler = run_cfg["train"]['lr']['scheduler']

    setup_seed(run_cfg['random_seed'])

    # Dataset
    dataset = ResDataset(training=True, validate_split=dataset_cfg['validate_split'])
    data_loader = DataLoader(dataset,
                             batch_size=run_cfg['train']['batch_size'],
                             num_workers=run_cfg['train']['num_workers'],
                             shuffle=True,
                             pin_memory=True)

    # Model
    model_name = run_cfg['model']
    model_params = model_cfg[model_name]
    input_vocab_len, output_vocab_len = dataset.vocab_len()
    model = getter.get_model(run_cfg['model'],
                             input_vocab_len,
                             output_vocab_len - 1,
                             **model_params).to(device)
    model.train()

    # Resume
    start_epoch = 0
    if resume != '':
        state, start_epoch = fileIO.find_checkpoint(resume)
        model.load_state_dict(state)
    start_steps = start_epoch * len(data_loader)

    # Module
    loss_fn = getter.get_loss(run_cfg['train']['loss'])
    if model_name == 'Transformer':
        loss_fn = MaskedLoss(loss_fn)
    optimizer = getter.get_optimizer(run_cfg['train']['optimizer'], model.parameters(), lr)
    lr_scheduler = getter.get_lr_scheduler(lr_scheduler, optimizer, model_cfg[model_name]['embedding_dim'], start_steps=start_steps)

    # Main
    for epoch in trange(start_epoch + 1, end_epoch + 1):
        # Step
        for step, data in enumerate(data_loader, start=1):
            src, trg = data[0].to(device), data[1].to(device)
            trg_in, trg_real = trg[:, 0:-1], trg[:, 1:]
            optimizer.zero_grad()

            if model_name == 'Transformer':
                src_key_padding_mask = mask.key_padding_mask(src).to(device)
                trg_key_padding_mask = mask.key_padding_mask(trg_in).to(device)
                attn_mask = mask.square_subsequent_mask(trg_in.size(1)).to(device)
                out = model(src, trg_in, src_key_padding_mask, trg_key_padding_mask, attn_mask)
                loss = loss_fn(out, trg_real, trg_key_padding_mask)
            else:
                out = model(src, trg_in)
                loss = loss_fn(out, trg_real)

            loss.backward()
            if step % run_cfg['train']['log_interval_steps'] == 0 or step == len(data_loader):
                logger.info('[Epoch {}/Step {}] loss: {:.6f}, lr: {:.6f}'.format(
                    epoch, step, loss.item(), lr_scheduler.get_last_lr()[0]
                ))
            optimizer.step()
            lr_scheduler.step()

        # Save
        if epoch % run_cfg['train']['saving_interval_epochs'] == 0:
            fileIO.save_checkpoint(exp_dir, model, epoch, exp_id=exp_id)
            logger.info('Saving model successfully.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Path to experiment_*/')
    opt = parser.parse_args()

    train(opt.resume)
