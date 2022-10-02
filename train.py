import utils.fileIO as fileIO
import yaml

import torch


def train(resume=''):
    # Load configs
    model_cfg = fileIO.load_config('./config/model.yaml')
    dataset_cfg = fileIO.load_config('./config/dataset.yaml')
    run_cfg = fileIO.load_config('./config/run.yaml')



def test():
    ...


if __name__ == "__main__":
    train()
