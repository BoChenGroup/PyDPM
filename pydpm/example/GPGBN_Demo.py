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
from pydpm._utils import cosine_simlarity

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*25), order='C')[:, 0:50]
train_label = data['train_label'][:50]

# construct the adjacency matrix
train_graph = cosine_simlarity(train_data.T, train_data.T)
train_graph[np.where(train_graph < 0.5)] = 0

# create the model and deploy it on gpu or cpu
model = GPGBN([128, 64, 32], device='gpu')
model.initial(train_data)
train_local_params = model.train(500, train_data, train_graph)

# save the model after training
model.save()
# model.load('./save_models/PGBN.npy')


