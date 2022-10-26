"""
===========================================
GAN
Generative Adversarial Networks
IJ Goodfellow，J Pouget-Abadie，M Mirza，B Xu，D Warde-Farley，S Ozair，A Courville，Y Bengio
Publihsed in 2014
===========================================
"""
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from pydpm._model._GAN import GAN
import torch

os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()
# GPU
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--g_latent_dim", type=int, default=100, help="generator dimensionality of the latent space")
parser.add_argument("--z_dim", type=int, default=100, help="generator dimensionality of the z latent space")
parser.add_argument("--d_latent_dim", type=int, default=100, help="discriminator dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=800, help="interval betwen image samples")
parser.add_argument('--load_epoch', type=int, default=200,help = 'load this epoch')
parser.add_argument('--checkpoints_interval', type=int, default=50)
parser.add_argument('--checkpoints_path', type=str, default='save_models/debug')
parser.add_argument('--checkpoints_file', type=str, default='')
parser.add_argument('--load_checkpoints_file', type=str, default='')
opt = parser.parse_args()

# choose gpu
if torch.cuda.is_available() and opt.gpu_id >= 0:
    device = torch.device('cuda:%d' % opt.gpu_id)
else:
    device = torch.device('cpu')
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if device!='cpu' else False


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
GAN = GAN(img_shape, g_latent_dim=opt.g_latent_dim, z_dim=opt.z_dim, d_latent_dim=opt.d_latent_dim, device=device)
generator = GAN.generator
discriminator = GAN.discriminator
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Configure data loader
os.makedirs("../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# models path
if not os.path.exists(opt.checkpoints_path):
    os.makedirs(opt.checkpoints_path)

save_path = os.path.join(opt.checkpoints_path, opt.checkpoints_file)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def train(epoch):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        # -----------------
        #  Train Generator
        # -----------------
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.g_latent_dim))))
        # Generate a batch of images
        gen_imgs = generator(z)
        if (i % 3 == 0):
            optimizer_G.zero_grad()
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        if (opt.checkpoints_interval > 0
                and (epoch + 1) % opt.checkpoints_interval == 0):
            GAN.save(epoch, save_path)
        if (i % 200 == 0):
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    train(epoch)
GAN.load(opt.load_epoch, os.path.join(opt.checkpoints_path, opt.load_checkpoints_file))
print(GAN)
print('sample image,please wait!')
save_image(GAN.sample(64), "./images/images.png", nrow=8, normalize=True)
print('complete!!!')
