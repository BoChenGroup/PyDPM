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

from pydpm.metric import ACC
from pydpm.model import DPGDS

from torchvision import datasets, transforms
from pydpm.dataloader.image_data import tensor_transforms

# load dataset
# define transform for dataset and load orginal dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../../dataset/mnist/', train=True, download=True)
test_dataset = datasets.MNIST(root='../../dataset/mnist/', train=False, download=False)

# transform dataset and reshape the dataset into [batch_size, feature_num]
train_data = tensor_transforms(train_dataset.data, transform)
train_data = train_data.permute(1, 2, 0).reshape(len(train_dataset), -1)# len(train_dataset, 28*28)
test_data = tensor_transforms(test_dataset.data, transform)
test_data = test_data.permute(1, 2, 0).reshape(len(test_dataset), -1)
train_label = train_dataset.train_labels
test_label = test_dataset.test_labels

# transpose the dataset to fit the model and convert a tensor to numpy array
train_data = np.array(np.ceil(train_data[:999, :].T.numpy() * 5), order='C')
test_data = np.array(np.ceil(test_data[:999, :].T.numpy() * 5), order='C')
train_label = train_label.numpy()[:999]
test_label = test_label.numpy()[:999]

# create the model and deploy it on gpu or cpu
model = DPGDS([200, 100, 50], 'gpu')  # topics of each layers
model.initial(train_data)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(train_data, iter_all=200)
train_local_params = model.test(train_data, iter_all=200)
test_local_params = model.test(test_data, iter_all=200)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8519
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/DPGDS.npy')


