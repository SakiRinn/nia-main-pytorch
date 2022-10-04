import os
import re
import glob
import yaml
import torch


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
    epoch = max([int(re.findall('\d+', ckpt)[-1]) for ckpt in checkpoints])
    ckpt_dir = glob.glob(os.path.join(exp_dir, f'ckpt_*_{epoch}.pth'))
    if len(ckpt_dir) != 1:
        raise FileExistsError("There are checkpoints at the same epoch but for different models.")
    return torch.load(ckpt_dir[0]), epoch


def save_checkpoint(dir: str, model, epoch=0):
    '''Save the checkpoint in new "dir/experiment_*/". '''
    experiments = glob.glob(os.path.join(dir, 'experiment_*'))
    if epoch == 0:
        exp_id = max([int(re.findall(r'\d+', exp)[-1]) for exp in experiments]) + 1 if experiments else 1
        exp_dir = os.path.join(dir, f"experiment_{exp_id}")
        os.mkdir(exp_dir)
    else:
        exp_id = max([int(re.findall(r'\d+', exp)[-1]) for exp in experiments])
        exp_dir = os.path.join(dir, f"experiment_{exp_id}")
    torch.save(model, os.path.join(exp_dir, f'ckpt_{model._get_name()}_{epoch}.pth'))