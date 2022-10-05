import os
import re
import glob
import yaml
import logging

import torch
import torch.nn as nn


def load_config(path: str):
    output = None
    with open(path, 'r') as f:
        output = yaml.load(f, yaml.FullLoader)

    if 'Seq2Seq' in output.keys():
        output['Seq2Seq']['encoder_dropout'] = output['Seq2Seq']['dropout']['encoder']
        output['Seq2Seq']['decoder_dropout'] = output['Seq2Seq']['dropout']['decoder']
        del output['Seq2Seq']['dropout']
    if 'Transformer' in output.keys():
        output['Transformer']['encoder_dropout'] = output['Transformer']['dropout']['encoder']
        output['Transformer']['decoder_dropout'] = output['Transformer']['dropout']['decoder']
        del output['Transformer']['dropout']

    return output


def find_checkpoint(exp_dir: str):
    '''Load the latest checkpoint in "exp_dir/". '''
    checkpoints = glob.glob(os.path.join(exp_dir, 'ckpt_*_*.pth'))
    if not checkpoints:
        return None, 0
    epoch = max([int(re.findall(r'\d+', ckpt)[-1]) for ckpt in checkpoints])
    ckpt_dir = glob.glob(os.path.join(exp_dir, f'ckpt_*_{epoch}.pth'))
    if len(ckpt_dir) != 1:
        raise FileExistsError("There are checkpoints at the same epoch but for different models.")
    return torch.load(ckpt_dir[0]), epoch


def save_checkpoint(dir: str, model: nn.Module, epoch=0, *, exp_id=0):
    '''Save the checkpoint in "dir/experiment_*/". '''
    exp_dir = os.path.join(dir, f"experiment_{exp_id}")
    torch.save(model.state_dict(), os.path.join(exp_dir, f'ckpt_{model._get_name()}_{epoch}.pth'))
    return model.state_dict()


def init_train(dir):
    '''Get the logger and the current exp_id and create a new "dir/experiment_*/". '''
    experiments = glob.glob(os.path.join(dir, 'experiment_*'))
    exp_id = max([int(re.findall(r'\d+', exp)[-1]) for exp in experiments]) + 1 if experiments else 1
    exp_dir = os.path.join(dir, f"experiment_{exp_id}")
    os.mkdir(exp_dir)

    logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'),
                        format='[%(levelname)s] %(asctime)s: %(message)s',
                        datefmt='%y-%b-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, exp_id


def init_eval(ckpt_dir):
    '''Get the logger and create a new "../eval/" if not exists. '''
    ckpt_name = ckpt_dir.split("/")[-1]
    dir = os.path.join(*ckpt_dir.split("/")[:-1])
    eval_dir = os.path.join(dir, 'eval')

    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    logging.basicConfig(filename=os.path.join(eval_dir, f'[Eval] {ckpt_name}.log'),
                        format='[%(levelname)s] %(asctime)s: %(message)s',
                        datefmt='%y-%b-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, eval_dir