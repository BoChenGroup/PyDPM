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

import argparse
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from pydpm.model import NFlow, RealNVP_2D
import torch

os.makedirs("../output/images", exist_ok=True)
parser = argparse.ArgumentParser()

parser.add_argument("--sample_num", type=int, default=512)
parser.add_argument("--flows_num", type=int, default=2)
parser.add_argument("--flow_name", type=str, default="RealNVP_2D")
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--hid_dim", type=int, default=128)
parser.add_argument("--device", type=int, default="cpu")
args = parser.parse_args()

# Dataset
data, label = make_moons(n_samples=args.sample_num, noise=0.05)
data = torch.tensor(data, dtype=torch.float32)
# Normalization
for i in range(data.shape[1]):
    data[:, i] = (data[:, i] - torch.mean(data[:, i])) / torch.std(data[:, i])
dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=128, shuffle=True)


flows = [RealNVP_2D(dim=2, hidden_dim=args.hid_dim, device=args.device) for _ in range(args.flows_num)]

model = NFlow(in_dim=2, flows=flows, device=args.device)
model_opt = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(args.n_epochs):
    local_z = model.train_one_epoch(model_opt=model_opt, dataloader=dataloader, epoch=epoch, n_epochs=args.n_epochs)
    if epoch == args.n_epochs - 1:
        test_local_z = model.test_one_epoch(dataloader=dataloader)


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