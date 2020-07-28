"""
===========================================
Poisson Gamma Dynamical Systems Demo
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
import scipy.io as sio
data = sio.loadmat('./mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
from pydpm.model import PGDS
model=PGDS(100,'cpu')
#model=PGDS(100,'gpu')
model.initial(train_data)
model.train(200)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(model.Likelihood, label="lh")
plt.legend(loc='lower right')
plt.show()