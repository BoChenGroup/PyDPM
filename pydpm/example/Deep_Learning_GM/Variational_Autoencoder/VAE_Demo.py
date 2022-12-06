"""
===========================================
VAE
Auto-Encoding Variational Bayes
Diederik P. Kingma， Max Welling
Publihsed in 2014

===========================================
"""

# Author: Muyao Wang <flare935694542@163.com>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse

from pydpm.model import VAE

import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import save_image

os.makedirs("../output/images", exist_ok=True)
parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--h_dim1", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--h_dim2", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--z_dim", type=int, default=2, help="dimensionality of the z latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
args = parser.parse_args()

# MNIST Dataset
train_dataset = datasets.MNIST(root='../../dataset/mnist/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../../dataset/mnist/', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

model = VAE(x_dim=args.img_size**2, h_dim1=args.h_dim1, h_dim2=args.h_dim2, z_dim=args.z_dim, device='cuda:0')
model_opt = optim.Adam(model.parameters(), lr=args.lr)

# Training

for epoch in range(args.n_epochs):
    local_mu, local_log_var = model.train_one_epoch(model_opt=model_opt, dataloader=train_loader, epoch=epoch, n_epochs=args.n_epochs)
    if epoch % 25 == 0:
        test_mu, test_log_var = model.test_one_epoch(dataloader=test_dataset)

# Save model
model.save()
# Load model
model.load('../save_models/VAE.pth')
# sample image
print('sample image,please wait!')
with torch.no_grad():
    sample = model.sample(64)
    save_image(sample.view(64,1,28, 28), '../images/VAE_l_sample_' + '.png')
    show_image = model.show()
    save_image(show_image.view(144, 1, 28, 28), '../images/VAE_l_show_' + '.png', nrow=12)

print('complete!!!')