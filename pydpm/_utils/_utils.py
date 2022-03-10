"""
===========================================
Metric
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu
# License: BSD-3-Clause

import numpy as np

realmin = 2.2e-10

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

