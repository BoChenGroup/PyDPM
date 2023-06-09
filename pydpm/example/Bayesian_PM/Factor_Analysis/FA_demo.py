"""
===========================================
Factor Analysis
===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>;
# License: BSD-3-Claus

import numpy as np
import scipy.io as sio
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pydpm.model import FA
from pydpm.utils.utils import *
from pydpm.dataloader.image_data import tensor_transforms

# # load data
# data = sio.loadmat('../../../dataset/FA_data.mat')
# train_data = np.array(data['x1'])

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--device", type=str, default='gpu')

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/FA.npy', help="the path of loading model")

parser.add_argument("--z_dim", type=int, default=128, help="number of components according dataset")

# optim
parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs of training")

args = parser.parse_args()
# =========================================== Dataset ===================================================================== #
# define transform for dataset and load orginal dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False)

# transform dataset and reshape the dataset into [batch_size, feature_num]
train_data = tensor_transforms(train_dataset.data, transform)
train_data = train_data.permute([1, 2, 0]).reshape([len(train_dataset), -1])  # len(train_dataset, 28*28)
test_data = tensor_transforms(test_dataset.data, transform)
test_data = test_data.permute([1, 2, 0]).reshape([len(test_dataset), -1])
train_label = train_dataset.train_labels
test_label = test_dataset.test_labels

# transpose the dataset to fit the model and convert a tensor to numpy array
# !!! Transposition, data: [D, N]
train_data = np.array(np.ceil(train_data[:999, :].T.numpy()), order='C')
# test_data = np.array(np.ceil(test_data[:999, :].T.numpy()), order='C')
train_data = standardization(train_data)
# test_data = standardization(test_data)
train_label = train_label.numpy()[:999]
# test_label = test_label.numpy()[:999]

# create the model and deploy it on gpu or cpu
model =FA(args.z_dim, 'gpu')
model.initial(train_data)  # use the shape of train_data to initialize the params of model

# train and evaluation
train_local_params = model.train(train_data, args.num_epochs)

x_hat = np.matmul(train_local_params.w, train_local_params.z)

# visualization for one sample
plt.plot(train_data[:, 880], 'ro', marker='*', label="train data")
plt.plot(x_hat[:, 880], 'bo', marker='v', label="reconstruction")
plt.legend(loc="best")
plt.show()

# save the model after training
model.save()
