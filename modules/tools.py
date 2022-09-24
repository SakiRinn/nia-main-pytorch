import os
import numpy as np


def find_checkpoint_file(path):
    checkpoint_file = [path + f for f in os.listdir(path) if 'weights' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]