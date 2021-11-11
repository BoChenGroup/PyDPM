"""
===========================================
Deep Poisson Gamma Dynamical Systems Demo
Dandan Guo, Bo Chen and Hao Zhang
Published in Neural Information Processing Systems 2018

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import DPGDS

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = DPGDS([200, 100, 50], 'gpu')
model.initial(train_data)
train_local_params = model.train(200, train_data)
train_local_params = model.test(200, train_data)
test_local_params = model.test(200, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8519
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/DPGDS.npy')

