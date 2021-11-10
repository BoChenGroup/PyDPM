"""
===========================================
Deep Poisson Factor Analysis Demo
Scalable Deep Poisson Factor Analysis for Topic Modeling
Zhe Gan, Changyou Chen, Ricardo Henao, David Carlson, Lawrence Carin
Publised in International Conference on Machine Learning 2015

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import DPFA

# load data
data = sio.loadmat('./data/mnist_gray.mat')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = DPFA([128, 64, 32], 'gpu')  # topics of 3 layers
model.initial(train_data)  # use the shape of train_data to initialize the params of model
burnin, collection = 100, 80
train_local_params = model.train(burnin, collection, train_data)
train_local_params = model.test(burnin, collection, train_data)
test_local_params = model.test(burnin, collection, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.9099
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/DPFA.npy')



