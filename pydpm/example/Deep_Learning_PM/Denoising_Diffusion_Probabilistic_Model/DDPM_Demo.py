'''
===========================================
DDPM
Denoising Diffusion Probabilistic Models
Jonathan Ho, Ajay Jain, Pieter Abbeel
Published in NIPS 2020

===========================================
'''

# Author: Xinyang Liu <lxy771258012@163.com>, Muyao Wang <flare935694542@163.com>
# License: BSD-3-Clause

import os
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from pydpm.model import DDPM

parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0, help="the id of gpu to deploy")

# dataset
parser.add_argument("--dataset", type=str, default='CIFAR10', help="the name of dataset")
parser.add_argument("--dataset_path", type=str, default='../../dataset', help="the file path of dataset")

# network settings
parser.add_argument("--T", type=int, default=1000, help="Number of time steps in DDPM")
parser.add_argument("--in_channel", type=int, default=3, help="Number of channels in the input image")
parser.add_argument("--channel", type=int, default=128, help="Number of channels after head layer")
parser.add_argument("--channel_mult", type=list, default=[1, 2, 3, 4], help="Number of mult-channels")
parser.add_argument("--attn", type=list, default=[2], help="Number of attention-blocks")
parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual-blocksn")
parser.add_argument("--dropout", type=list, default=0.15, help="Dropout ratio")

# ddpm settings
parser.add_argument("--beta_1", type=float, default=1e-4, help="the level of noise in first step of forward process")
parser.add_argument("--beta_T", type=float, default=0.02, help="The level of noise in T-th step of forward process")

# optimizer
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--grad_clip", type=float, default=1., help="grad_clip")

# training
parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="batch size of dataloader")

# sampling
parser.add_argument("--model_path", type=str, default="../../save_models/DDPM.pth", help="path to save/load model")
parser.add_argument("--noisy_path", type=str, default="../../output/noisy.png", help="path to save noisy")
parser.add_argument("--image_path", type=str, default="../../output/image.png", help="path to save sampled images")

args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else f'cuda:{args.gpu_id}'

# dataset
dataset = CIFAR10(
    root=args.dataset_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
if dataset == 'CIFAR10':
    args.in_channel = 3
elif dataset == 'mnist':
    args.in_channel = 1
else:
    assert print('args.in_channel must be given')

net_config = {"in_channel": args.in_channel,
              "channel": args.channel,
              "channel_mult": args.channel_mult,
              "attn": args.attn,
              "num_res_blocks": args.num_res_blocks,
              "dropout": args.dropout}
ddpm_config = {"beta_1": 1e-4, "beta_T": 0.02}

model = DDPM(T=args.T, net_cfg=net_config, ddpm_cfg=ddpm_config, device=args.device)

model_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=model_opt, T_max=args.num_epochs, eta_min=0, last_epoch=-1)

for epoch in range(args.num_epochs):
    model.train_one_epoch(dataloader, model_opt, epoch, args)
    cosine_scheduler.step()
    if (epoch + 1) % 10 == 0:
        model.save()
        model_test = DDPM(T=args.T, net_cfg=net_config, ddpm_cfg=ddpm_config, device=args.device)
        model_test.load(args.model_path)
        noisy, images = model_test.test_one_epoch(net_model=model_test.net, args=args)

        saveNoisy = torch.clamp(noisy * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, args.noisy_path, nrow=8)
        sampledImgs = images * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, args.image_path, nrow=8)
