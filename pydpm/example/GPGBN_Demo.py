"""
===========================================
Deep Relational Topic Modeling via Graph Poisson Gamma Belief Network
Chaojie Wang, Hao Zhang, Bo Chen, Dongsheng Wang, Zhengjue Wang
Published in Advances in Neural Information Processing System 2020

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Wei Zhao <13279389260@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import GPGBN

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*25), order='C')[:, 0:50]
test_data = np.array(np.ceil(data['test_mnist']*25), order='C')[:, 0:50]  # todo 如何进行测试
train_label = data['train_label'][:50]
test_label = data['test_label'][0:50]



# create the model and deploy it on gpu or cpu
model = GPGBN([128, 64, 32], device='gpu')
model.initial(train_data)
train_local_params = model.train(2000, train_data)
train_local_params = model.test(2000, train_data)
test_local_params = model.test(2000, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8549
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/PGBN.npy')


