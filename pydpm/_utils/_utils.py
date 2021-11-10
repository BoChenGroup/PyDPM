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
