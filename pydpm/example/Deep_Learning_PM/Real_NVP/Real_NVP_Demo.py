"""Training procedure for real NVP.
"""

import os
import argparse
import torch
import torch.distributions as distributions
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import datasets, transforms

import numpy as np
from pydpm.model import RealNVP
from pydpm.model.deep_learning_pm.realnvp import DataInfo

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0)

# dataset
parser.add_argument('--dataset', type=str, default='mnist', help='dataset to be modeled.')
parser.add_argument("--data_path", type=str, default='../../../dataset/mnist/', help="the path of loading data")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/realNVP.pth', help="the path of loading model")

parser.add_argument('--base_dim', type=int, default=64, help='features in residual blocks of first few layers.')
parser.add_argument('--res_blocks', type=int, default=8, help='number of residual blocks per group.')
parser.add_argument('--bottleneck', type=int, default=0, help='whether to use bottleneck in residual blocks.')
parser.add_argument('--skip', type=int, default=1, help='whether to use skip connection in coupling layers.')
parser.add_argument('--weight_norm', type=int, default=1, help='whether to apply weight normalization.')
parser.add_argument('--coupling_bn', type=int, default=1, help='whether to apply batchnorm after coupling layers.')
parser.add_argument('--affine', type=int, default=1, help='whether to use affine coupling.')

# optim
parser.add_argument('--batch_size', type=int, default=64, help='number of images in a mini-batch.')
parser.add_argument('--num_epochs', type=int, default=500, help='maximum number of training epoches.')
parser.add_argument('--sample_size', type=int, default=64, help='number of images to generate.')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='beta1 in Adam optimizer.')
parser.add_argument('--decay', type=float, default=0.999, help='beta2 in Adam optimizer.')
parser.add_argument('--scale_reg', type=float, default=5e-5, help='L2 regularization strength')

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# mnist
data_info = DataInfo(args.dataset, 1, 28)  # if cifar10, channels: 1->3
train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
args.image_size = 28
# =========================================== Model ===================================================================== #
# model

# Set prior
prior = distributions.Normal(torch.tensor(0.).to(args.device), torch.tensor(1.).to(args.device))

# Initial model
model = RealNVP(datainfo=data_info, prior=prior, device=args.device, args=args).to(args.device)
model_opt = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.momentum, args.decay), eps=1e-7)

# train
best_log_ll = float('-inf')
for epoch_idx in range(args.num_epochs):
    log_ll_mean = model.train_one_epoch(dataloader=train_loader, model_opt=model_opt, epoch=epoch_idx, args=args)
    if epoch_idx % 5 == 0:
        log_ll_mean = model.test_one_epoch(dataloader=test_loader, epoch=epoch_idx, args=args)
        if log_ll_mean > best_log_ll:
            # save
            model.save(args.save_path)

# load
model.load(args.load_path)

# =================== Visualization ====================== #
os.makedirs("../../output/images", exist_ok=True)
print('sample image, please wait!')
with torch.no_grad():
    sample = model.sample(64)
    save_image(sample.view(64, 1, 28, 28), '../../output/images/realNVP_sample.png')
print('complete!!!')




















