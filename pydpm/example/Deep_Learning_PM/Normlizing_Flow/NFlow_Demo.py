"""
===========================================
RealNVP
DENSITY ESTIMATION USING REAL NVP
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
Publihsed in  2017

===========================================
"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from pydpm.model import NFlow
from pydpm.model.deep_learning_pm.nflow import RealNVP_2D

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/NFlow.pth', help="the path of loading model")

parser.add_argument("--sample_num", type=int, default=512)
parser.add_argument("--flows_num", type=int, default=2)
parser.add_argument("--flow_name", type=str, default="RealNVP_2D")
parser.add_argument("--hid_dim", type=int, default=128)

# optim
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
data, label = make_moons(n_samples=args.sample_num, noise=0.05)
data = torch.tensor(data, dtype=torch.float32)
# Normalization
for i in range(data.shape[1]):
    data[:, i] = (data[:, i] - torch.mean(data[:, i])) / torch.std(data[:, i])
dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

# =========================================== Model ===================================================================== #
# model
flows = [RealNVP_2D(dim=2, hidden_dim=args.hid_dim, device=args.device) for _ in range(args.flows_num)]

model = NFlow(in_dim=2, flows=flows, device=args.device)
model_opt = optim.Adam(model.parameters(), lr=args.lr)

# train
for epoch_idx in range(args.num_epochs):
    local_z = model.train_one_epoch(model_opt=model_opt, dataloader=dataloader, epoch=epoch_idx, num_epochs=args.num_epochs)
    if epoch_idx == args.num_epochs - 1:
        test_local_z = model.test_one_epoch(dataloader=dataloader)

# save
model.save(args.save_path)
# load
model.load(args.load_path)

# =========================================== Visualization ===================================================================== #
# visualize
os.makedirs("../output/images", exist_ok=True)
print('sample image,please wait!')

plt.figure(figsize=(8, 3))
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], marker=".", color="b", s=10)
plt.title("Training data")
plt.subplot(1, 3, 2)
plt.scatter(test_local_z[:, 0], test_local_z[:, 1], marker=".", color="r", s=10)
plt.title("Latent space")
plt.subplot(1, 3, 3)
samples = model.sample(args.sample_num).cpu().detach().numpy()
plt.scatter(samples[:, 0], samples[:, 1], marker=".", color="b", s=10)
plt.title("Generated samples")
plt.savefig("../output/images/nflow.png")
plt.show()

