"""
===========================================
Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network
Zhibin Duan, Dongsheng Wang, Bo Chen, Chaojie Wang, Wenchao Chen, Yewen Li, Jie Ren and Mingyuan Zhou
Published as a conference paper at ICML 2021

===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse
import numpy as np
import scipy.io as sio
from sklearn.cluster import k_means
from nltk.corpus import stopwords

import torch
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS

from pydpm.model import SawETM
from pydpm.utils import *
from pydpm.metric import *
from pydpm.dataloader.text_data import Text_Processer, build_vocab_from_iterator

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0, help="the id of gpu to deploy")

# dataset
parser.add_argument("--dataset", type=str, default='AG_NEWS', help="the name of dataset")
parser.add_argument("--data_path", type=str, default='../../../dataset', help="the file path of dataset")

# model
parser.add_argument("--embed_size", type=int, default=64, help="the length of vocabulary")
parser.add_argument("--vocab_size", type=int, default=8000, help="the length of vocabulary")
parser.add_argument("--num_topics_list", type=list, default=[128, 64, 32], help="the list of z dimension")
parser.add_argument("--num_hiddens_list", type=list, default=[200, 200, 200], help="the list of hidden dimension")
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/SawETM.pth', help="the path of loading model")

# optim
parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="l2 regularization strength")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

# =========================================== Dataset ===================================================================== #
# load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS(args.data_path, split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

# build vocabulary
stop_words = list(stopwords.words('english'))
vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True, stop_words=stop_words, max_tokens=args.vocab_size)
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

train_bows, train_labels = text_processer.bow_from_file(train_files, train_labels)
test_bows, test_labels = text_processer.bow_from_file(test_files, test_labels)

train_loader = DataLoader([train_data for train_data in zip(train_bows, train_labels)], batch_size=256, shuffle=False, num_workers=4, drop_last=True)
test_loader = DataLoader([test_data for test_data in zip(test_bows, test_labels)], batch_size=256, shuffle=False, num_workers=4, drop_last=True)

# if args.pretrained_embeddings:
#     print('Using pretrained glove embeddings')
#     initial_embeddings = load_glove_embeddings(args.embed_size, vocab)
# else:
#     initial_embeddings = None
initial_embeddings = None
# =========================================== Model ===================================================================== #
model = SawETM(embed_size=args.embed_size, vocab_size=args.vocab_size, num_hiddens_list=args.num_hiddens_list, num_topics_list=args.num_topics_list, word_embeddings=initial_embeddings, device=args.device)
model.to(args.device)
model_opt = torch.optim.Adam(params=model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)


###############  Training  ################

for epoch_idx in range(args.num_epochs):
    _, _ = model.train_one_epoch(dataloader=train_loader, model_opt=model_opt, epoch_idx=epoch_idx, args=args)

    if (epoch_idx+1) % 20 == 0:
        theta, labels = model.test_one_epoch(dataloader=test_loader)

        # calculate NMI with train_local_params
        cls_num = len(np.unique(train_labels + test_labels))
        test_theta_norm = standardization(theta)
        tmp = k_means(test_theta_norm, cls_num)  # N*K
        predict_label = tmp[1] + 1  # Some label start with '1' not '0', there should be 'tmp[1] + 1'
        MI = NMI(labels, predict_label)
        purity = Purity(labels, predict_label)

# save
model.save(args.save_path)
# load
model.load(args.load_path)





