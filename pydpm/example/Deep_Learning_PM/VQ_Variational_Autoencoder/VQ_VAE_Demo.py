"""
===========================================
VQ-VAE
Neural Discrete Representation Learning
Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
Publihsed in 2017

===========================================
"""

# Author: Muyao Wang <flare935694542@163.com>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse
import sys
import torch
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image
from pydpm.model import VQVAE

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/VQVAE.pth', help="the path of loading model")
parser.add_argument("--embed_dim", type=int, default=2, help="dimensionality of the codebook the same as z_dim")
parser.add_argument("--num_embed", type=int, default=8000, help="number of codebook")
parser.add_argument("--z_dim", type=int, default=2, help="dimensionality of the z latent space")
parser.add_argument("--encoder_hid_dims", type=int, default=[512, 256], help="dimensionality of the latent space")
parser.add_argument("--decoder_hid_dims", type=int, default=[512, 256], help="dimensionality of the latent space")

# optim
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================================== Model ===================================================================== #
# model
model = VQVAE(embed_dim=args.embed_dim, num_embed=args.num_embed, in_dim=args.img_size**2, z_dim=args.z_dim, encoder_hid_dims=args.encoder_hid_dims, decoder_hid_dims=args.decoder_hid_dims, device=args.device)
model_opt = optim.Adam(model.parameters(), lr=args.lr)

# train
for epoch_idx in range(args.num_epochs):
    model.train_one_epoch(dataloader=train_loader, model_opt=model_opt, epoch_idx=epoch_idx, args=args)
    if epoch_idx % 25 == 0:
        model.test_one_epoch(dataloader=test_loader)

# save
model.save(args.save_path)
# load
model.load(args.load_path)

# =================== Visualization ====================== #
os.makedirs("../../output/images", exist_ok=True)
print('sample image,please wait!')
with torch.no_grad():
    sample = model.sample(64)
    save_image(sample.view(64, 1, 28, 28), '../../output/images/VQVAE_sample.png')
print('complete!!!')

