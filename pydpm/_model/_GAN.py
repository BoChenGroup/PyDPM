"""
===========================================
GAN
Generative Adversarial Networks
IJ Goodfellow，J Pouget-Abadie，M Mirza，B Xu，D Warde-Farley，S Ozair，A Courville，Y Bengio
Publihsed in 2014
===========================================
"""
import numpy as np
import torch.nn as nn
import torch

class GAN(nn.Module):
    def __init__(self, img_shape, g_latent_dim=100, z_dim=128, d_latent_dim=512, device='cuda:0'):
        """
        The basic model for GAN
        Inputs:
            img_shape : [tuple] the size of input
            g_latent_dim : [int] generator latent dimension;
            z_dim : [int] the noise dimension
            d_latent_dim : [int] discriminator latent dimension
            device : [str] 'cpu' or 'gpu';
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.g_latent_dim = g_latent_dim
        self.generator = Generator(latent_dim = g_latent_dim, z_dim = z_dim, img_shape = self.img_shape).to(device)
        self.discriminator =Discriminator(latent_dim = d_latent_dim, img_shape = self.img_shape).to(device)
        self.device = device

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            gen_imgs : [tensor] a batch of images
        """
        # Sample noise as generator input
        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.g_latent_dim))).to(self.device)
        # Generate a batch of images
        gen_imgs = self.generator(z)
        return gen_imgs

    def save(self, epoch, checkpoints):
        """
        Save model
        Inputs:
            epoch     : [int] train epoch;
            checkpoints : [str] trained model path;
        """
        '======>>saving'
        torch.save({'epoch': epoch + 1,
                    'state_dict': self.state_dict()},
                   checkpoints + '_epochs{}.pth'.format(epoch + 1))

    def load(self, start_ep, checkpoints):
        """
        Load model
        Inputs:
            start_ep     : the epoch of checkpoint;
            checkpoints : trained model path;
        """
        try:
            print("Loading Chechpoint from ' {} '".format(checkpoints + '_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(checkpoints + '_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['state_dict'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
            self.start_epoch = 0
        except:
            print("No Checkpoint Exists At '{}'".format(checkpoints))
            self.start_epoch = 0


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim = 100, z_dim = 128):
        """
        The basic model for generator
        Inputs:
            img_shape : [tuple] the size of input
            latent_dim : [int] generator latent dimension;
            z_dim : [int] the noise dimension
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, self.z_dim, normalize=False),
            *block(self.z_dim, self.z_dim*2),
            *block(self.z_dim*2, self.z_dim*4),
            *block(self.z_dim*4, self.z_dim*8),
            nn.Linear(self.z_dim*8, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward process of generator
        Inputs:
            z : [tuple] the size of input
        Outputs；
            img : [tensor] return the generated image
        """
        z = z.float()
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, latent_dim = 256):
        """
        The basic model for discriminator
        Inputs:
            img_shape : [tuple] the size of input
            latent_dim : [int] discriminator latent dimension;
        """
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), self.latent_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim*2, self.latent_dim ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        """
        Forward process of discriminator
        Inputs:
            img : [tensor] input image
        Outputs；
            validity : [tensor] return the validity of the input image
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity



