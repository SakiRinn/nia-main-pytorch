import utils.fileIO as fileIO
import yaml

import torch


def train(resume=''):
    # Load configs
    model_cfg = fileIO.load_config('./configs/model.yaml')
    dataset_cfg = fileIO.load_config('./configs/dataset.yaml')
    run_cfg = fileIO.load_config('./configs/run.yaml')



def test():
    ...


if __name__ == "__main__":
    train()
