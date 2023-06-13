"""
===========================================
CVAE
Learning Structured Output Representation using Deep Conditional Generative Models
Kihyuk Sohn, Xinchen Yan and Honglak Lee
Publihsed in 2015

===========================================
"""
# Author: Bufeng Ge <20009100138@stu.xidian.edu.cn>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse
import sys
import torch
import numpy as np
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image
from pydpm.model import CVAE

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/CVAE.pth', help="the path of loading model")

parser.add_argument("--z_dim", type=int, default=64, help="dimensionality of the z latent space")
parser.add_argument("--encoder_hid_dims", type=int, default=[512, 256], help="dimensionality of the latent space")
parser.add_argument("--decoder_hid_dims", type=int, default=[512, 256], help="dimensionality of the latent space")

# optim
parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transforms.ToTensor(), download=False)
args.cond_dim = len(train_dataset.classes)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

model = CVAE(cond_dim=args.cond_dim, in_dim=args.img_size**2, z_dim=args.z_dim, encoder_hid_dims=args.encoder_hid_dims, decoder_hid_dims=args.decoder_hid_dims, device='cuda:0')
model_opt = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.num_epochs):
    local_mu, local_log_var = model.train_one_epoch(model_opt=model_opt, dataloader=train_loader, epoch=epoch, n_epochs=args.num_epochs)
    if epoch % 25 == 0:
        test_mu, test_log_var = model.test_one_epoch(dataloader=test_loader)

# Save model
model.save(args.save_path)
# Load model
model.load(args.load_path)

# =================== Visualization ====================== #
os.makedirs("../../output/images", exist_ok=True)
# sample image
print('sample image, please wait!')
with torch.no_grad():
    sample = model.sample(64, torch.tensor([7]*64))#[random.randint(0,9) for i in range(64)])
    save_image(sample.view(64,1,28, 28), '../../output/images/VAE_sample_7.png')
print('complete!!!')
