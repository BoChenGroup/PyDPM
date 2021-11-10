"""
===========================================
WEDTM Demo
Inter and Intra Topic Structure Learning with Word Embeddings
He Zhao, Lan Du, Wray Buntine, Mingyaun Zhou
Published in International Council for Machinery Lubrication 2018

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import WEDTM

# load data
dataset = sio.loadmat('./data/WS.mat')
train_data = np.asarray(dataset['doc'].todense()[:, dataset['train_idx'][0]-1])[:, ::10].astype(int)
test_data = np.asarray(dataset['doc'].todense()[:, dataset['test_idx'][0]-1])[:, ::5].astype(int)
train_label = dataset['labels'][dataset['train_idx'][0]-1][::10, :]
test_label = dataset['labels'][dataset['test_idx'][0]-1][::5, :]

# params of model
T = 3  # vertical layers
S = 3  # sub topics
K = [100] * T  # topics in each layers

# create the model and deploy it on gpu or cpu
model = WEDTM(K, 'gpu')
model.initial(dataset['doc'])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(dataset['embeddings'], S, 300, train_data)
train_local_params = model.test(dataset['embeddings'], S, 300, train_data)
test_local_params = model.test(dataset['embeddings'], S, 300, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/WEDTM.npy')


