"""
===========================================
Metric
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu; Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause
import os
import random
import numpy as np

import torch

realmin = 2.2e-10

# randomness
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# math
def log_max(x):
    return np.log(np.maximum(x, realmin))

def cosine_simlarity(A, B):
    # A: N*D, B: N*D
    [N, D] = A.shape
    inter_product = np.matmul(A, np.transpose(B))  # N*N
    len_A = np.sqrt(np.sum(A * A, axis=1, keepdims=True))
    len_B = np.sqrt(np.sum(B * B, axis=1, keepdims=True))
    len_AB = np.matmul(len_A, np.transpose(len_B))
    cos_AB = inter_product / (len_AB + realmin)
    cos_AB[(np.arange(N), np.arange(N))] = 1
    return cos_AB

def standardization(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / (sigma + 2.2e-8)
