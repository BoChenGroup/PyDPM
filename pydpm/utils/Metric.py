"""
===========================================
Metric
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from scipy.special import gamma

realmin = 2.2e-10

def log_max(x):
    return np.log(np.maximum(x, realmin))

def Poisson_Likelihood(X, X_re):

    # X[np.where(X>100)] = 100
    # X_re[np.where(X>100)] = 100

    Likelihood = np.sum(X*log_max(X_re) - X_re - log_max(gamma(X_re + 1)))
    return Likelihood

def Reconstruct_Error(X, X_re):
    return np.power(X - X_re, 2).sum()