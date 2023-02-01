"""
===========================================
RBM
A Practical Guide to Training
Restricted Boltzmann Machines
Geoffrey Hinton
Publihsed in 2010
===========================================
"""
import numpy as np
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pydpm.model import RBM

batch_size = 128
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size)

rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(),0.1)

def train(epoch, final_epoch):
    loss_ = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data.view(-1, 784))
        sample_data = data.bernoulli()

        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.item())
        train_op.zero_grad()
        loss.backward()
        train_op.step()
    print('Train Epoch: {} Loss: {:.6f}'.format(epoch,np.mean(loss_)))
    if(epoch == final_epoch):
        print('saving!!! Please wait!!!')
        save_image(v.view(-1, 1, 28, 28), 'l_real_' + '.png')
        save_image(v1.view(-1, 1, 28, 28), 'l_generate_' + '.png')

epochs = 10
for epoch in range(epochs):
    train(epoch, final_epoch=9)


rbm.save(epoch=epochs, checkpoints='../save_models/')
rbm.load(epochs, '../save_models/')
