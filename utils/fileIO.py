import os
import re
import glob
import yaml
import torch


def load_config(path: str):
    output = None
    with open(path, 'r') as f:
        output = yaml.load(f, yaml.FullLoader)
    if 'seq2seq' in output.keys():
        output['seq2seq']['encoder_dropout'] = output['seq2seq']['dropout']['encoder']
        output['seq2seq']['decoder_dropout'] = output['seq2seq']['dropout']['decoder']
        del output['seq2seq']['dropout']
    if 'transformer' in output.keys():
        output['transformer']['encoder_dropout'] = output['transformer']['dropout']['encoder']
        output['transformer']['decoder_dropout'] = output['transformer']['dropout']['decoder']
        del output['transformer']['dropout']
    return output


def find_checkpoint(exp_dir: str):
    '''Load the latest checkpoint in "exp_dir/". '''
    checkpoints = glob.glob(os.path.join(exp_dir, 'ckpt_*_*.pth'))
    if not checkpoints:
        return None
    exp_id = max([int(re.findall('\d+', ckpt)[-1]) for ckpt in checkpoints])
    ckpt_dir = glob.glob(os.path.join(exp_dir, f'ckpt_*_{exp_id}.pth'))
    if len(ckpt_dir) != 1:
        raise FileExistsError("There are the same checkpoints but for different models in the directory.")
    return torch.load(ckpt_dir[0])


def save_checkpoint(dir: str, model, epoch=0):
    '''Save the checkpoint in new "dir/experiment_*/". '''
    experiments = glob.glob(os.path.join(dir, 'experiment_*'))
    if epoch == 0:
        exp_id = max([int(re.findall('\d+', exp)[-1]) for exp in experiments]) + 1 if experiments else 1
        exp_dir = os.path.join(dir, f"experiment_{exp_id}")
        os.mkdir(exp_dir)
    else:
        exp_id = max([int(re.findall('\d+', exp)[-1]) for exp in experiments])
        exp_dir = os.path.join(dir, f"experiment_{exp_id}")
    torch.save(model, os.path.join(exp_dir, f'ckpt_{model._get_name()}_{epoch}.pth'))