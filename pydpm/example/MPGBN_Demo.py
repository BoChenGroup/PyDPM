"""
===========================================
Multimodal Poisson Gamma Belief Network
Chaojie Wang, Bo Chen and Mingyuan Zhou
Published in In AAAI Conference on Artificial Intelligence

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import MPGBN

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
train_data_1 = train_data[:360, :]
train_data_2 = train_data[360:, :]

test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
test_data_1 = test_data[:360, :]
test_data_2 = test_data[360:, :]

train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]


# create the model and deploy it on gpu or cpu
model = MPGBN([128, 64, 32], device='gpu')
model.initial(train_data_1, train_data_2)
train_local_params = model.train(100, train_data_1, train_data_2)
train_local_params = model.test(100, train_data_1, train_data_2)
test_local_params = model.test(100, test_data_1, test_data_2)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8549
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/PGBN.npy')


