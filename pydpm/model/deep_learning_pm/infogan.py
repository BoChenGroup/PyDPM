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

import numpy as np
import torch.nn as nn
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
import os

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class InfoGAN(nn.Module):
    def __init__(self, args, device='cuda:0'):
        """
        The basic model for GAN
        Inputs:
            img_shape : [tuple] the size of input
            g_z_dim : [int] the noise dimension
            g_latent_dim : [list] generator latent dimension;
            d_latent_dim : [list] discriminator latent dimension
            device : [str] 'cpu' or 'gpu';
        """
        super(InfoGAN, self).__init__()
        setattr(self, '_model_name', 'InfoGAN')
        self.z_dim = args.z_dim
        self.dis_ch = args.dis_ch
        self.dis_ch_dim = args.dis_ch_dim
        self.con_ch = args.con_ch

        # Initialise the network
        self.generator = Generator(in_channels=(self.z_dim + self.dis_ch * self.dis_ch_dim + self.con_ch)).to(device)
        self.discriminator = Discriminator().to(device)
        self.netD = DHead().to(device)
        self.netQ = QHead().to(device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.netD.apply(weights_init)
        self.netQ.apply(weights_init)

        self.criterionD = torch.nn.BCELoss().to(device)
        self.criterionQ_dis = nn.CrossEntropyLoss().to(device)

        # 'self.criterionQ_con': Loss for continuous latent code will be used in training stage

        self.device = device
        self.Tensor = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor

    def criterionQ_con(self, x, mu, var):
        """
        Calculate the negative log likelihood
        of normal distribution.
        This needs to be minimised.

        Treating Q(cj | x) as a factored Gaussian.
        """
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

    def noise_sample(self, batch_size):
        """
        Sample random noise vector for training
        batch_size : Batch Size
            dis_ch : Number of discrete latent code
            dis_ch_dim : Dimension of discrete latent code
            con_ch : Number of continuous latent code
            z_dim : Dimension of incompressible noise
        """

        noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
        idx = np.zeros((self.dis_ch, batch_size))

        if self.dis_ch != 0:
            dis_c = torch.zeros(batch_size, self.dis_ch, self.dis_ch_dim, device=self.device)
            for i in range(self.dis_ch):
                # Encode discrete code with one-hot
                idx[i] = np.random.randint(self.dis_ch_dim, size=batch_size)
                dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

            dis_c = dis_c.view(batch_size, -1, 1, 1)
            noise = torch.cat((noise, dis_c), dim=1)

        if (self.con_ch != 0):
            # Random uniform between -1 and 1.
            con_c = torch.rand(batch_size, self.con_ch, 1, 1, device=self.device) * 2 - 1
            noise = torch.cat((noise, con_c), dim=1)

        return noise, idx

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            gen_imgs : [tensor] a batch of images
        """
        # Sample noise as generator input
        noise, idx = self.noise_sample(batch_size)
        # Generate a batch of images
        gen_imgs = self.generator(noise)

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
        G_loss_t, D_loss_t = 0., 0.
        train_bar = tqdm(iterable=dataloader)
        for i, (imgs, _) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_bar.set_postfix(G_loss=G_loss_t / (i + 1), D_loss=D_loss_t / (i + 1))

            bs = imgs.shape[0]
            # Adversarial ground truths
            valid = torch.full((bs, ), 1., requires_grad=False, device=self.device)
            fake = torch.full((bs, ), 0., requires_grad=False, device=self.device)
            real_imgs = imgs.to(self.device)

            noise, idx = self.noise_sample(bs)
            gen_imgs = self.generator(noise)

            # Train Discriminator and Head
            model_opt_D.zero_grad()
            probs_real = self.netD(self.discriminator(real_imgs))
            real_loss = self.criterionD(probs_real.view(-1), valid)
            real_loss.backward()

            probs_fake = self.netD(self.discriminator(gen_imgs.detach()))
            fake_loss = self.criterionD(probs_fake.view(-1), fake)
            fake_loss.backward()

            d_loss = real_loss + fake_loss
            # d_loss.backward()
            model_opt_D.step()
            D_loss_t += d_loss.item()

            model_opt_G.zero_grad()
            output = self.discriminator(gen_imgs)
            probs = self.netD(output)
            gen_loss = self.criterionD(probs.view(-1), valid)

            q_logits, q_mu, q_var = self.netQ(output)
            target = torch.LongTensor(idx).to(self.device)
            # Calculating loss for discrete latent code.
            dis_loss = 0.
            for j in range(self.dis_ch):
                dis_loss += self.criterionQ_dis(q_logits[:, j * self.dis_ch_dim: j * self.dis_ch_dim + self.dis_ch_dim], target[j])

            # Calculating loss for continuous latent code.
            con_loss = 0.
            if self.con_ch != 0:
                con_loss = 0.1 * self.criterionQ_con(noise[:, self.z_dim + self.dis_ch * self.dis_ch_dim:].view(-1, self.con_ch), q_mu, q_var)

            g_loss = gen_loss + dis_loss + con_loss
            g_loss.backward()
            model_opt_G.step()
            G_loss_t += g_loss.item()

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                sample_images = gen_imgs.data.cpu()[:25]
                save_image(sample_images[:25], "../../output/images/InfoGAN_%d.png" % batches_done, nrow=5, normalize=True)


    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/InfoGAN.pth';
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


class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(in_channels, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        img = torch.sigmoid(self.tconv4(x))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
