# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-18 11:09
@Author  : zzy
@File    : utils.py

"""
import os
import random
import torch
import numpy as np


def set_seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_split_data(model, data, split):
    split_x = [model.wv[idx] for idx, e in enumerate(data[split]) if e and idx in model.wv]
    split_y = [data.y[idx] for idx, e in enumerate(data[split]) if e and idx in model.wv]

    return split_x, split_y
