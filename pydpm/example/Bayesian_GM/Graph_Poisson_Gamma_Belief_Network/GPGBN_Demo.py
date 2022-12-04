"""
===========================================
Deep Relational Topic Modeling via Graph Poisson Gamma Belief Network
Chaojie Wang, Hao Zhang, Bo Chen, Dongsheng Wang, Zhengjue Wang
Published in Advances in Neural Information Processing System 2020

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Wei Zhao <13279389260@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause

import os
import numpy as np
import scipy.io as sio

from pydpm.metric import ACC
from pydpm.model import GPGBN
from pydpm.dataloader.image_data import tensor_transforms
from pydpm.dataloader.graph_data import graph_from_data, graph_from_edges
from pydpm.utils import cosine_simlarity

from torchvision import datasets, transforms
from torch_geometric.datasets import Planetoid

# # load dataset (Cora) cost too much memory
# path = '../../dataset/Planetoid'
# if not os.path.exists(path):
#     os.mkdir(path)
# dataset = Planetoid(path, 'cora')
# dataset = dataset[0]
#
# graph = graph_from_edges(dataset.edge_index, dataset.num_nodes, to_sparsetesor=False)[1]
# # transpose the dataset to fit the model and convert a tensor to numpy array
# train_data = dataset.x.T.numpy()

# load dataset (MNIST)
# define transform for dataset and load orginal dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../../dataset/mnist/', train=True, download=True)
test_dataset = datasets.MNIST(root='../../dataset/mnist/', train=False, download=False)

# transform dataset and reshape the dataset into [batch_size, feature_num]
train_data = tensor_transforms(train_dataset.data, transform)
train_data = train_data.permute([1, 2, 0]).reshape(len(train_dataset), -1)  # len(train_dataset, 28*28)
test_data = tensor_transforms(test_dataset.data, transform)
test_data = test_data.permute([1, 2, 0]).reshape(len(test_dataset), -1)
train_label = train_dataset.train_labels
test_label = test_dataset.test_labels

# transpose the dataset to fit the model and convert a tensor to numpy array
train_data = np.array(np.ceil(train_data[:999, :].T.numpy() * 5), order='C')
train_label = train_label.numpy()[:999]

# construct the adjacency matrix
graph = graph_from_data(train_data.T, 0.5, binary=False)



# create the model and deploy it on gpu or cpu
model = GPGBN([128, 64, 32], device='gpu')
model.initial(train_data)
train_local_params = model.train(train_data, graph, iter_all=100)

# save the model after training
model.save()
# model.load('./save_models/PGBN.npy')


