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
import scipy.io as sio
import torch

from pydpm.metric import ACC
from pydpm.model import WEDTM

from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchtext.datasets import AG_NEWS
from pydpm.dataloader.text_data import file_from_iter, file_filter, BatchProcess

# Load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS('../dataset/', split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

# Get train/test label and data_file(tokens) from data_iter and convert them into clean file
stop_words = ['<unk>']
train_label, train_file = file_from_iter(train_iter, clean=True, tokenizer=tokenizer, stop_words=stop_words)
test_label, test_file = file_from_iter(test_iter, clean=True, tokenizer=tokenizer, stop_words=stop_words)
# Take part of dataset for convenience
train_label = train_label[:1000]
train_file = train_file[:1000]
test_label = test_label[:2000]
test_file = test_file[:2000]
data_file = train_file + test_file

# Build full vocabulary
specials_words = []
voc = build_vocab_from_iterator(data_file, specials=specials_words, special_first=False)

# Build part of vocabulary
voc_num = 1500
if voc_num > len(voc):
    voc_num = len(voc)
new_voc_list = voc.get_itos()[:voc_num]
train_file = file_filter(train_file, voc_list=new_voc_list)
test_file = file_filter(test_file, voc_list=new_voc_list)
data_file = train_file + test_file
voc = build_vocab_from_iterator(data_file, specials=specials_words, special_first=False)

# Build word embedding
vector = GloVe(name='6B', dim=50)
voc_embedding = vector.get_vecs_by_tokens(voc.get_itos(), lower_case_backup=True)

# Dataloader
batch_process = BatchProcess(tokenizer, voc)
train_bow = batch_process.bow_from_file(train_file, to_sparse=True)
test_bow = batch_process.bow_from_file(test_file, to_sparse=True)

# Transpose dataset to fit the model and convert a tensor to numpy
train_data = np.asarray(train_bow.todense()).T.astype(int)
test_data = np.asarray(train_bow.todense()).T.astype(int)
voc_embedding = voc_embedding.numpy()

print('Data has been processed!')

# # load dataset (WS.mat from paper)
# dataset = sio.loadmat('../../dataset/WS.mat')
# train_data = np.asarray(dataset['doc'].todense()[:, dataset['train_idx'][0]-1])[:, ::10].astype(int)
# test_data = np.asarray(dataset['doc'].todense()[:, dataset['test_idx'][0]-1])[:, ::5].astype(int)
# train_label = dataset['labels'][dataset['train_idx'][0]-1][::10, :]
# test_label = dataset['labels'][dataset['test_idx'][0]-1][::5, :]

# model = WEDTM(K, 'gpu')
# model.initial(dataset['doc'])  # use the shape of train_data to initialize the params of model
# train_local_params = model.train(dataset['embeddings'], S, 300, train_data)
# train_local_params = model.test(dataset['embeddings'], S, 300, train_data)
# test_local_params = model.test(dataset['embeddings'], S, 300, test_data)

# params of model
T = 3  # vertical layers
S = 3  # sub topics
K = [100] * T  # topics in each layers

# create the model and deploy it on gpu or cpu
model = WEDTM(K, 'gpu')
model.initial(train_bow.T)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(voc_embedding, S, 300, train_data)
train_local_params = model.test(voc_embedding, S, 300, train_data)
test_local_params = model.test(voc_embedding, S, 300, test_data)


# evaluate the model with classification accuracy
# the demo accuracy can achieve
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/WEDTM.npy')


