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
import argparse
import scipy.io as sio

from torchvision import datasets, transforms
from torch_geometric.datasets import Planetoid

from pydpm.model import GPGBN
from pydpm.metric import ACC
from pydpm.dataloader.image_data import tensor_transforms
from pydpm.dataloader.graph_data import Graph_Processer
from pydpm.utils import cosine_simlarity

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--device", type=str, default='gpu')

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/GPGBN.npy', help="the path of loading model")

parser.add_argument("--z_dims", type=list, default=[128, 64, 32], help="number of topics at diffrent layers in PGBN")

# optim
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")

args = parser.parse_args()

# =========================================== Dataset ===================================================================== #
# define transform for dataset and load orginal dataset

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
train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False)

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
graph_processer = Graph_Processer()
graph = graph_processer.graph_from_node_feature(train_data.T, 0.5, binary=False)

# =========================================== Model ===================================================================== #
# create the model and deploy it on gpu or cpu
model = GPGBN(K=args.z_dims, device=args.device)
model.initial(train_data)

train_local_params = model.train(train_data, graph, num_epochs=args.num_epochs)

# save the model after training
model.save(args.save_path)
# load the model
model.load(args.load_path)


