"""
===========================================
DCGAN
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Alec Radford, Luke Metz and Soumith Chintala
Publihsed in ICLR 2016

===========================================
"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from pydpm.utils.utils import unnormalize_to_zero_to_one
from tqdm import tqdm
import os

class DCGAN(nn.Module):
    def __init__(self, args, device='cuda:0'):
        super(DCGAN, self).__init__()
        setattr(self, '_model_name', 'DCGAN')
        self.z_dim = args.z_dim
        self.generator = Generator(args.in_channels, z_dim=self.z_dim).to(device)
        self.discriminator = Discriminator(args.in_channels).to(device)
        self.adversarial_loss = torch.nn.BCELoss().to(device)
        self.in_channel = args.in_channels
        self.device = device
        self.Tensor = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            gen_imgs : [tensor] a batch of images
        """
        # Sample noise as generator input
        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.z_dim, 1, 1))).type(self.Tensor).to(self.device)
        # Generate a batch of images
        gen_imgs = self.generator(z)
        return gen_imgs

    def train_one_epoch(self, model_opt_G, model_opt_D, dataloader, sample_interval, epoch, n_epochs):
        '''
        Train for one epoch
        Inputs:
            model_opt_G     : Optimizer for generator
            model_opt_D     : Optimizer for discriminator
            dataloader      : Train dataset with form of dataloader
            sample_interval : interval betwen image samples while training
            epoch           : Current epoch on training stage
            n_epoch         : Total number of epochs on training stage
        '''
        G_loss_t, D_loss_t = 0, 0
        train_bar = tqdm(iterable=dataloader)
        for i, (imgs, _) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_bar.set_postfix(G_loss=G_loss_t / (i + 1), D_loss=D_loss_t / (i + 1))

            imgs = imgs.to(self.device)
            # Adversarial ground truths
            valid = Variable(self.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
            fake = Variable(self.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
            real_imgs = Variable(imgs.type(self.Tensor))

            gen_imgs = self.sample(imgs.shape[0])

            # Train Discriminator
            model_opt_D.zero_grad()
            real_loss = self.adversarial_loss(self.discriminator(real_imgs).view(imgs.shape[0], 1), valid)
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()).view(imgs.shape[0], 1), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            model_opt_D.step()
            D_loss_t += d_loss.item()

            # Train Generator
            model_opt_G.zero_grad()
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs).view(imgs.shape[0], 1), valid)
            g_loss.backward()
            model_opt_G.step()
            G_loss_t += g_loss.item()

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                sample_images = gen_imgs.data.cpu()[:25]
                sample_images = unnormalize_to_zero_to_one(sample_images)
                save_image(sample_images, "../../output/images/dcgan_%d.png" % batches_done, nrow=5, normalize=True)

    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/DCGAN.pth';
        """
        # Save the model
        torch.save({'state_dict': self.state_dict()}, model_path + '/' + self._model_name + '.pth')
        print('model has been saved by ' + model_path + '/' + self._model_name + '.pth')


    def load(self, model_path):
        """
        load model
        Inputs:
            model_path : [str] the path to load the model;
        """
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        # Load the model
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])

class Generator(torch.nn.Module):
    def __init__(self, in_channels, z_dim=100):
        '''
            in_channels : channels of input data
            z_dim : dimension of latent vector
        '''
        super().__init__()
        self.convT_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # [bs, 1024, 4, 4]
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # [bs, 512, 8, 8]
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # [bs, 256, 16, 16]
            nn.ConvTranspose2d(in_channels=256, out_channels=in_channels, kernel_size=4, stride=2, padding=1))
            # [bs, c, 32, 32]

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.convT_layers(x)
        x = self.output(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        '''
            in_channels : channels of input data
        '''
        super().__init__()
        self.conv_layers = nn.Sequential(
            # [bs, c, 32, 32]
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # [bs, 256, 16, 16]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # [bs, 512, 8, 8]
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # [bs, 1024, 4, 4]

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # [bs, 1, 1, 1]
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output(x)
        return x
