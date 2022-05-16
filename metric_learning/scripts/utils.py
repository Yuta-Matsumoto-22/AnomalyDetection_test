import os
import random

import numpy as np

import torch

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def device_setting(gpu):
    if gpu == -1:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda:' + str(gpu)
    else:
        device = 'cpu'

    return device

def calc_cosine_similarity(np_array_a, np_array_b):
    eps = 1e-6
    dis = np_array_a @ np_array_b.T
    norm_a = (np_array_a * np_array_a).sum(1, keepdims=True) ** (0.5)
    norm_b = (np_array_b * np_array_b).sum(1, keepdims=True) ** (0.5)
    similarity_matrix = dis / (norm_a+eps) / (norm_b.T+eps)

    return similarity_matrix
