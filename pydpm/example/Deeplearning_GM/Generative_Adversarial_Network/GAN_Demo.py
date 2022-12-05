"""
===========================================
GAN
Generative Adversarial Networks
IJ Goodfellow，J Pouget-Abadie，M Mirza，B Xu，D Warde-Farley，S Ozair，A Courville，Y Bengio
Publihsed in 2014

===========================================
"""

# Author: Muyao Wang <flare935694542@163.com>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from pydpm.model import GAN
import torch

os.makedirs("../output/images", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--g_latent_dim", type=int, default=100, help="generator dimensionality of the latent space")
parser.add_argument("--z_dim", type=int, default=100, help="generator dimensionality of the z latent space")
parser.add_argument("--d_latent_dim", type=int, default=100, help="discriminator dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=800, help="interval betwen image samples")
args = parser.parse_args()


# MNIST Dataset
transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='../../dataset/mnist/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize generator and discriminator
img_shape = (args.channels, args.img_size, args.img_size)
model = GAN(img_shape, g_latent_dim=args.g_latent_dim, z_dim=args.z_dim, d_latent_dim=args.d_latent_dim, device='cuda:0')

# Optimizers
model_opt_G = torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
model_opt_D = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Training
for epoch in range(args.n_epochs):
    model.train_one_epoch(model_opt_G=model_opt_G, model_opt_D=model_opt_D, dataloader=dataloader, sample_interval=args.sample_interval, epoch=epoch, n_epochs=args.n_epochs)

model.save()
model.load('../save_models/GAN.pth')
print(model)
print('sample image,please wait!')
save_image(model.sample(64), "../images/GAN_images.png", nrow=8, normalize=True)
print('complete!!!')