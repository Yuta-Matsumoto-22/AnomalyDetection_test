import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


def data_split(x, y, test_size=0.2, val_size=0.2, random_state=0):
    # attr:
    #   split x, y to train_x, val_x, test_x, train_y, val_y, test_y
    # input: 
    #   x, y, test_size=0.2, val_size=0.2, random_state=0
    # return: 
    #   train_x, val_x, test_x, train_y, val_y, test_y
    v_size = val_size / (1.0 - test_size)
    tr_val_x, test_x, tr_val_y, test_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
    train_x, val_x, train_y, val_y = train_test_split(tr_val_x, tr_val_y, test_size=v_size, random_state=random_state)

def image_check(img, nums):
    for i in range(nums):
        plt.imshow(img[i], cmap='gray')
        plt.show()
