"""
===========================================
WHAI: WEIBULL HYBRID AUTOENCODING INFERENCE FOR DEEP TOPIC MODELING (Demo)
Hao Zhang, Bo Chen， Dandan Guo and Mingyuan Zhou
Published as a conference paper at ICLR 2018

===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio
from pydpm._model import WHAI
from pydpm._utils import *
from pydpm._metric import *
import torch
from sklearn.cluster import k_means

dataname = '20ng_20'
data_path = '../data/whai_20ng.pkl'
clc_num = 20
batch_size = 256
n_epochs = 500
train_loader, voc = dataloader(data_path, dataname=dataname, mode='train', batch_size=batch_size)
test_loader, _ = dataloader(data_path, dataname=dataname, mode='test', batch_size=batch_size)

K = [128, 64, 32]
H = [200, 200, 200]
# K = [100]
# H = [200]
phi_out = f'{dataname}/{K}'

model = WHAI(K=K, H=H, V=len(voc), device='cuda:0')
model.initial(voc=voc, cls=clc_num, batch_size=batch_size, n_epochs=n_epochs, MBratio=len(train_loader))

model = model.cuda()
model_opt = torch.optim.Adam(model.parameters())

# Training

for epoch in range(n_epochs):
    train_local_params = model.train_one_epoch(model_opt=model_opt, dataloader=train_loader, epoch=epoch)
    if epoch % 20 == 0:
        model.save_phi(phi_path=phi_out, epoch=epoch)
        test_theta, test_label, full_test_data = model.test_one_epoch(dataloader=test_loader)
        # calculate PPL
        x_hat = np.matmul(test_theta, np.transpose(model.global_params.Phi[0]))
        ppl = Perplexity(np.transpose(full_test_data), np.transpose(x_hat))
        # calculate NMI with train_local_params
        test_data_norm = standardization(train_local_params.theta)
        tmp = k_means(test_data_norm, clc_num)  # N*K
        predict_label = tmp[1]
        MI = NMI(train_local_params.label, predict_label)

# save the model after training
model.save()
torch.save({'state_dict':model.state_dict()}, './save_models/WHAI.pth')

whai = WHAI(K=K, H=H, V=len(voc), device='gpu')

whai.load('./save_models/WHAI.pth', './save_models/WHAI.npy')
test_theta, test_label, full_test_data = whai.test_one_epoch(dataloader=test_loader)
x_hat = np.matmul(test_theta, np.transpose(whai.global_params.Phi[0]))
ppl = Perplexity(np.transpose(full_test_data), np.transpose(x_hat))

