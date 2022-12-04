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
import nltk
import numpy as np
import scipy.io as sio
import torch

from pydpm.metric import ACC
from pydpm.model import WEDTM

from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from torchtext.datasets import AG_NEWS
from pydpm.dataloader.text_data import Text_Processer, build_vocab_from_iterator

# Load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS('../dataset/', split=('train', 'test'))
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

# params of model
T = 3  # vertical layers
S = 3  # sub topics
K = [100] * T  # topics in each layers

# create the model and deploy it on gpu or cpu
model = WEDTM(K, 'gpu')
model.initial(train_data)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(voc_embedding, train_data, S, iter_all=300, is_initial_local=False)
train_local_params = model.test(voc_embedding, train_data, S, iter_all=300)
test_local_params = model.test(voc_embedding, test_data, S, iter_all=300)


# evaluate the model with classification accuracy
results = ACC(train_local_params.Theta, test_local_params.Theta, train_labels, test_labels, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/WEDTM.npy')

# # load dataset (WS.mat from paper)
# dataset = sio.loadmat('../../dataset/WS.mat')
# train_data = np.asarray(dataset['doc'].todense()[:, dataset['train_idx'][0]-1])[:, ::10].astype(int)
# test_data = np.asarray(dataset['doc'].todense()[:, dataset['test_idx'][0]-1])[:, ::5].astype(int)
# train_label = dataset['labels'][dataset['train_idx'][0]-1][::10, :]
# test_label = dataset['labels'][dataset['test_idx'][0]-1][::5, :]

