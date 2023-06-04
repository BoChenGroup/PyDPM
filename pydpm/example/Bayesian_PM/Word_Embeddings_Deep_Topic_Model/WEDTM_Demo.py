"""
===========================================
WEDTM Demo
Inter and Intra Topic Structure Learning with Word Embeddings
He Zhao, Lan Du, Wray Buntine, Mingyaun Zhou
Published in International Council for Machinery Lubrication 2018

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import argparse
import scipy.io as sio

import nltk
from nltk.corpus import stopwords

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from torchtext.datasets import AG_NEWS

from pydpm.model import WEDTM
from pydpm.metric import ACC
from pydpm.dataloader.text_data import Text_Processer, build_vocab_from_iterator

# =========================================== ArgumentParser ===================================================================== #
parser = argparse.ArgumentParser()

# device
parser.add_argument("--device", type=str, default='gpu')

# dataset
parser.add_argument("--data_path", type=str, default='../dataset/', help="the path of loading data")

# model
parser.add_argument("--save_path", type=str, default='../../save_models', help="the path of saving model")
parser.add_argument("--load_path", type=str, default='../../save_models/WEDTM.npy', help="the path of loading model")

parser.add_argument("--z_dim", type=int, default=100, help="number of topics in each layers")
parser.add_argument("--T", type=int, default=3, help="number of vertical layers")
parser.add_argument("--S", type=int, default=3, help="number of sub topics")

# optim
parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs of training")

args = parser.parse_args()

# =========================================== Dataset ===================================================================== #
# define transform for dataset and load orginal dataset
# Load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS(args.data_path, split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

# build vocabulary
# nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
vocab_size = 7000
vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), special_first=False, stop_words=stop_words, max_tokens=vocab_size)
text_processer = Text_Processer(tokenizer=tokenizer, vocab=vocab)

# Get train/test label and data_file(tokens) from data_iter and convert them into clean file
train_files, train_labels = text_processer.file_from_iter(train_iter, tokenizer=tokenizer)
test_files, test_labels = text_processer.file_from_iter(test_iter, tokenizer=tokenizer)

# Take part of dataset for convenience
train_idxs = np.arange(5000)
np.random.shuffle(train_idxs)
train_files = [train_files[i] for i in train_idxs]
train_labels = [train_labels[i] for i in train_idxs]

test_idxs = np.arange(1000)
np.random.shuffle(test_idxs)
test_files = [test_files[i] for i in test_idxs]
test_labels = [test_labels[i] for i in test_idxs]

# Build word embedding
vector = GloVe(name='6B', dim=50)
voc_embedding = vector.get_vecs_by_tokens(vocab.get_itos(), lower_case_backup=True)

# Dataloader
train_bow, train_labels = text_processer.bow_from_file(train_files, train_labels, to_sparse=False)
test_bow, test_labels = text_processer.bow_from_file(test_files, test_labels, to_sparse=False)

# Transpose dataset to fit the model and convert a tensor to numpy
train_data = np.asarray(train_bow).T.astype(int)
test_data = np.asarray(test_bow).T.astype(int)
voc_embedding = voc_embedding.numpy()

print('Data has been processed!')

# =========================================== Model ===================================================================== #
# create the model and deploy it on gpu or cpu
model = WEDTM(K=[args.z_dim] * args.T, device=args.device)
model.initial(train_data)  # use the shape of train_data to initialize the params of model

# train and evaluation
train_local_params = model.train(voc_embedding, train_data, args.S, num_epochs=args.num_epochs, is_initial_local=False)
train_local_params = model.test(voc_embedding, train_data, args.S, num_epochs=args.num_epochs)
test_local_params = model.test(voc_embedding, test_data, args.S, num_epochs=args.num_epochs)

# save the model after training
model.save(args.save_path)
# load the model
model.load(args.load_path)

# evaluate the model with classification accuracy
results = ACC(train_local_params.Theta, test_local_params.Theta, train_labels, test_labels, 'SVM')


# # load dataset (WS.mat from paper)
# dataset = sio.loadmat('../../dataset/WS.mat')
# train_data = np.asarray(dataset['doc'].todense()[:, dataset['train_idx'][0]-1])[:, ::10].astype(int)
# test_data = np.asarray(dataset['doc'].todense()[:, dataset['test_idx'][0]-1])[:, ::5].astype(int)
# train_label = dataset['labels'][dataset['train_idx'][0]-1][::10, :]
# test_label = dataset['labels'][dataset['test_idx'][0]-1][::5, :]

