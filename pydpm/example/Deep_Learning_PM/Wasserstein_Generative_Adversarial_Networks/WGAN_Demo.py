"""
===========================================
WGAN
Wasserstein GAN
Martin Arjovsky, Soumith Chintala, and Leon Bottou,
Publihsed in 2017

===========================================
"""

# Author: Bufeng Ge <20009100138@stu.xidian.edu.cn>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from pydpm.model import WGAN
 
# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/WGAN.pth', help="the path of loading model")

parser.add_argument("--g_z_dim", type=int, default=128, help="generator dimensionality of the noise")
parser.add_argument("--g_hid_dims", type=list, default=[100, 200, 400, 800], help="generator dimensionality of the latent space")
parser.add_argument("--d_hid_dims", type=list, default=[256, 128], help="discriminator dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 1 for mnist
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--n_critic", type=int, default=100, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

# optim
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================================== Model ===================================================================== #
# model
# Initialize generator and discriminator
img_shape = (args.channels, args.img_size, args.img_size)
model = WGAN(img_shape, g_z_dim=args.g_z_dim, g_hid_dims=args.g_hid_dims, d_hid_dims=args.d_hid_dims, device=args.device)

# Optimizers
model_opt_G = torch.optim.RMSprop(model.generator.parameters(), lr=args.lr)
model_opt_D = torch.optim.RMSprop(model.discriminator.parameters(), lr=args.lr)

# train
for epoch_idx in range(args.num_epochs):
    model.train_one_epoch(args=args, model_opt_G=model_opt_G, model_opt_D=model_opt_D, dataloader=train_loader, epoch=epoch_idx, n_epochs=args.num_epochs)


# save
model.save(args.save_path)
# load
model.load(args.load_path)

# ===================== Visualization ================= #
os.makedirs("../../output/images", exist_ok=True)
print('sample image, please wait!')
save_image(model.sample(64), "../../output/images/WGAN_images.png", nrow=8, normalize=True)
print('complete!!!')

