"""
===========================================
InfoGAN
InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
Publihsed in 2016

===========================================
"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from pydpm.model import InfoGAN
from pydpm.utils.utils import unnormalize_to_zero_to_one
# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/GAN.pth', help="the path of loading model")

parser.add_argument("--z_dim", type=int, default=62, help="generator dimensionality of the noise")
parser.add_argument("--dis_ch", type=int, default=1, help="generator dimensionality of the latent space")
parser.add_argument("--dis_ch_dim", type=int, default=10, help="discriminator dimensionality of the latent space")
parser.add_argument("--con_ch", type=int, default=2, help="discriminator dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 1 for mnist
parser.add_argument("--sample_interval", type=int, default=800, help="interval betwen image samples")

# optim
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor()])
train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================================== Model ===================================================================== #
# model
# Initialize generator and discriminator
img_shape = (args.channels, args.img_size, args.img_size)
model = InfoGAN(args=args, device=args.device)

# Optimizers
model_opt_G = torch.optim.Adam([{'params': model.generator.parameters()}, {'params': model.netQ.parameters()}], lr=args.lr, betas=(args.b1, args.b2))
model_opt_D = torch.optim.Adam([{'params': model.discriminator.parameters()}, {'params': model.netD.parameters()}], lr=args.lr, betas=(args.b1, args.b2))


# train
for epoch_idx in range(args.num_epochs):
    model.train_one_epoch(model_opt_G=model_opt_G, model_opt_D=model_opt_D, dataloader=train_loader, sample_interval=args.sample_interval, epoch=epoch_idx, n_epochs=args.num_epochs)

# save
model.save(args.save_path)
# load
model.load(args.load_path)

# ===================== Visualization ============================== #
os.makedirs("../../output/images", exist_ok=True)
print('sample image, please wait!')
save_image(model.sample(64), "../../output/images/InfoGAN_images.png", nrow=8, normalize=True)
print('complete!!!')