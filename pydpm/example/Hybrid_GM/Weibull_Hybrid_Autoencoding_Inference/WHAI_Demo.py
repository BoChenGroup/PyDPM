"""
===========================================
WHAI: WEIBULL HYBRID AUTOENCODING INFERENCE FOR DEEP TOPIC MODELING (Demo)
Hao Zhang, Bo Chen， Dandan Guo and Mingyuan Zhou
Published as a conference paper at ICLR 2018

===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import argparse
import numpy as np
import scipy.io as sio
from sklearn.cluster import k_means
from nltk.corpus import stopwords

from pydpm.model import WHAI
from pydpm.utils import *
from pydpm.metric import *
from pydpm.dataloader.text_data import Text_Processer, build_vocab_from_iterator

import torch
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS

parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0, help="the id of gpu to deploy")

# dataset
parser.add_argument("--dataset", type=str, default='AG_NEWS', help="the name of dataset")
parser.add_argument("--dataset_path", type=str, default='../../../dataset', help="the file path of dataset")

# network settings
parser.add_argument("--voc_size", type=int, default=20000, help="the length of vocabulary")
parser.add_argument("--z_dims", type=list, default=[128, 64, 32], help="the list of z dimension")
parser.add_argument("--hid_dims", type=list, default=[200, 200, 200], help="the list of hidden dimension")

# optimizer
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")

# training
parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--MBratio", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")

args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else f'cuda:{args.gpu_id}'

# Load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS(args.dataset_path, split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

# build vocabulary
stop_words = list(stopwords.words('english'))
vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True, stop_words=stop_words, max_tokens=args.voc_size)
vocab.set_default_index(vocab['<unk>'])
text_processer = Text_Processer(tokenizer=tokenizer, vocab=vocab)


# Get train/test label and data_file(tokens) from data_iter and convert them into clean file
train_files, train_labels = text_processer.file_from_iter(train_iter, tokenizer=tokenizer)
test_files, test_labels = text_processer.file_from_iter(test_iter, tokenizer=tokenizer)

# Take part of dataset for convenience
train_idxs = np.arange(7000)
np.random.shuffle(train_idxs)
train_files = [train_files[i] for i in train_idxs]
train_labels = [train_labels[i] for i in train_idxs]

test_idxs = np.arange(3000)
np.random.shuffle(test_idxs)
test_files = [test_files[i] for i in test_idxs]
test_labels = [test_labels[i] for i in test_idxs]

# =========================================== load data 1 ===================================================================== #
train_bows, train_labels = text_processer.bow_from_file(train_files, train_labels)
test_bows, test_labels = text_processer.bow_from_file(test_files, test_labels)

train_loader = DataLoader([train_data for train_data in zip(train_bows, train_labels)], batch_size=256, shuffle=False, num_workers=4, drop_last=True)
test_loader = DataLoader([test_data for test_data in zip(test_bows, test_labels)], batch_size=256, shuffle=False, num_workers=4, drop_last=True)

cls_num = len(np.unique(train_labels + test_labels))
model = WHAI(in_dim=args.voc_size, z_dims=args.z_dims, hid_dims=args.hid_dims, device=args.device)
model_opt = torch.optim.Adam(model.parameters())

for epoch in range(args.num_epochs):
    train_local_params = model.train_one_epoch(dataloader=train_loader, optim=model_opt, epoch_index=epoch, args=args, is_train=True)
    if (epoch+1) % 20 == 0:
        test_local_params = model.test_one_epoch(dataloader=test_loader)
        # calculate PPL
        x_hat = np.matmul(test_local_params.Theta[0], np.transpose(model.global_params.Phi[0]))
        ppl = Perplexity(test_local_params.data, x_hat)

        # calculate NMI with train_local_params
        test_theta_norm = standardization(test_local_params.Theta[0])
        tmp = k_means(test_theta_norm, cls_num)  # N*K
        predict_label = tmp[1] + 1 # Some label start with '1' not '0', there should be 'tmp[1] + 1'
        MI = NMI(test_local_params.label, predict_label)




