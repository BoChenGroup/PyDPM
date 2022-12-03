"""
===========================================
Convolutional Poisson Factor Analysis
Chaojie Wang  Sucheng Xiao  Bo Chen  and  Mingyuan Zhou
Published in International Conference on Machine Learning 2019

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import numpy as np
import random
import scipy.io as sio
import _pickle as cPickle

from pydpm.metric import ACC
from pydpm.model import CPFA
from pydpm.dataloader.text_data import Text_Processer

from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchtext.datasets import AG_NEWS


# load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS('../../dataset/', split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

# build vocabulary
vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True, max_tokens=5000)
vocab.set_default_index(vocab['<unk>'])
text_processer = Text_Processer(tokenizer=tokenizer, vocab=vocab)

# Get train/test label and data_file(tokens) from data_iter and convert them into clean file
train_files, train_labels = text_processer.file_from_iter(train_iter, tokenizer=tokenizer, stop_words=None)
test_files, test_labels = text_processer.file_from_iter(test_iter, tokenizer=tokenizer, stop_words=None)

# Take part of dataset for convenience
train_idxs = np.arange(3000)
np.random.shuffle(train_idxs)
train_files = [train_files[i] for i in train_idxs]
train_labels = [train_labels[i] for i in train_idxs]

test_idxs = np.arange(1000)
np.random.shuffle(test_idxs)
test_files = [test_files[i] for i in test_idxs]
test_labels = [test_labels[i] for i in test_idxs]

# ===================================== mode 1, sparse input ====================================== #
# Build batch of word2index
train_sparse_batch, train_labels = text_processer.word_index_from_file(train_files, train_labels, to_sparse=True)
test_sparse_batch, test_labels = text_processer.word_index_from_file(test_files, test_labels, to_sparse=True)
print('Data has been processed!')

# create the model and deploy it on gpu or cpu
model = CPFA(200, 'gpu')

model.initial(train_sparse_batch, is_sparse=True)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(train_sparse_batch, is_sparse=True, iter_all=100)
train_local_params = model.test(train_sparse_batch, is_sparse=True, iter_all=100)
test_local_params = model.test(test_sparse_batch, is_sparse=True, iter_all=100)

train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T

# Score of test dataset's Theta: 0.682
results = ACC(train_theta, test_theta, train_labels, test_labels, 'SVM')
model.save()


# # Use custom dataset
# DATA = cPickle.load(open("../../dataset/TREC.pkl", "rb"), encoding='iso-8859-1')
#
# data_vab_list = DATA['Vocabulary']
# data_vab_count_list = DATA['Vab_count']
# data_vab_length = DATA['Vab_Size']
# data_label = DATA['Label']
# data_train_list = DATA['Train_Origin']
# data_train_label = np.array(DATA['Train_Label'])
# data_train_split = DATA['Train_Word_Split']
# data_train_list_index = DATA['Train_Word2Index']
# data_test_list = DATA['Test_Origin']
# data_test_label = np.array(DATA['Test_Label'])
# data_test_split = DATA['Test_Word_Split']
# data_test_list_index = DATA['Test_Word2Index']
# data_value = 25
#
# batch_len_tr, batch_rows_tr, batch_cols_tr, batch_file_index_tr, batch_value_tr, batch_label_tr = process_sparse_data(data_train_list_index, data_train_label)
# batch_len_te, batch_rows_te, batch_cols_te, batch_file_index_te, batch_value_te, batch_label_te = process_sparse_data(data_test_list_index, data_test_label)
#
# # create the model and deploy it on gpu or cpu
# model = CPFA(200, 'gpu')
# # mode 1, dense input
# model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
# train_local_params = model.train(500, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
# train_local_params = model.test(500, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
# test_local_params = model.test(500, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(data_test_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_te)])
#
# train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
# test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T
#
# train_theta[np.where(np.isinf((train_theta)))] = 0
#
# # Score of test dataset's Theta: 0.682
# results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
# model.save()



# mode 2, sparse input
# X_file_index, X_rows, X_cols = np.where(train_data)
# X_value = train_data[X_file_index, X_rows, X_cols]
# N, V, L = train_data.shape
# model = CPFA(kernel=100)
# model.initial([[X_file_index, X_rows, X_cols, X_value], [N, V, L]], dtype='sparse')
# model.train(iter_all=100)

# #pgcn demo
# train_data = sio.loadmat("./mnist_gray.mat")
# dataset = np.array(np.ceil(train_data['train_mnist'] * 5), order='C')[:,:999]  # 0-1
# dataset=np.transpose(dataset)
# dataset = np.reshape(dataset, [dataset.shape[0], 28, 28])
# #GPU only
#
# #dense input
# model=CPFA(kernel=100)
# model.initial(dataset)
# model.train(iter_all=100)
#
# #sparse input
# X_file_index, X_rows, X_cols = np.where(dataset)
# X_value = dataset[X_file_index, X_rows, X_cols]
# N,V,L = dataset.shape
# model = CPFA(kernel=100)
# model.initial([[X_file_index, X_rows, X_cols, X_value], [N,V,L]], dtype='sparse')
# model.train(iter_all=100)