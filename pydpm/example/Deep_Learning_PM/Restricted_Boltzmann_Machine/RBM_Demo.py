"""
===========================================
RBM
A Practical Guide to Training
Restricted Boltzmann Machines
Geoffrey Hinton
Publihsed in 2010
===========================================
"""
# Author: Muyao Wang <flare935694542@163.com>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse
import numpy as np

import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision.utils import save_image

from pydpm.model import RBM

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/RBM.pth', help="the path of loading model")

parser.add_argument("--n_vis", type=int, default=784, help="dimensionality of visible units")
parser.add_argument("--n_hin", type=int, default=500, help="dimensionality of latent units")
parser.add_argument("--k", type=int, default=1, help="layers of RBM")

# optim
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================================== Model ===================================================================== #
# model
model = RBM(n_vis=args.n_vis, n_hin=args.n_hin, k=args.k)
model_opt = optim.SGD(model.parameters(), lr=args.lr)

# train
for epoch_idx in range(args.num_epochs):
    v, v1 = model.train_one_epoch(dataloader=train_loader, model_opt=model_opt, epoch_idx=epoch_idx, args=args)

# save
model.save(args.save_path)
# load
model.load(args.load_path)

# =========================================== Visualization ===================================================================== #
# visualize
os.makedirs("../../output/images", exist_ok=True)
print('sample image,please wait!')
with torch.no_grad():
    save_image(v.view(-1, 1, 28, 28), '../../output/images/RBM_l_real_' + '.png')
    save_image(v1.view(-1, 1, 28, 28), '../../output/images/RBM_l_generate_' + '.png')
print('complete!!!')


# device .test_one_epoch
