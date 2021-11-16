"""
===========================================
Poisson Factor Analysis DirBN(Dirichlet belief networks) Demo
Dirichlet belief networks for topic structure learning
He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou
Publihsed in Conference and Workshop on Neural Information Processing Systems 2018

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import DirBN

# load data
data = sio.loadmat('./data/mnist_gray.mat')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:499]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1499]
train_label = data['train_label'][:499]
test_label = data['train_label'][1000:1499]

# create the model and deploy it on gpu or cpu
model = DirBN([100, 100], 'gpu')  # topics of each layers
model.initial(train_data)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(90, train_data)
train_local_params = model.test(90, train_data)
test_local_params = model.test(90, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.78
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/DirBN.npy')


