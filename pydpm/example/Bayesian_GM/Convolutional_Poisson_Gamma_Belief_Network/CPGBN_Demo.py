"""
================================================
Convolutional Poisson Gamma Belief Network Demo
Chaojie Wang  Sucheng Xiao  Bo Chen  and  Mingyuan Zhou
Published in International Conference on Machine Learning 2019

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause
import numpy as np
import scipy.io as sio
import _pickle as cPickle
from pydpm.metric import ACC
from pydpm.model import CPGBN
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchtext.datasets import AG_NEWS
from pydpm.dataloader.text_data import file_from_iter, file_filter, BatchProcess

delete_count = 0
def process_sparse_data(file_index, label):
    delete_count = 0
    for i in range(len(file_index)):
        x_single = np.reshape(file_index[i], [len(file_index[i])]).astype(np.int32)
        x_len = x_single.shape[0]
        i_index = i - delete_count
        if i_index == 0:
            batch_len = np.array([x_len])
            batch_rows = x_single
            batch_cols = np.arange(x_len)
            batch_file_index = np.ones_like(x_single) * i_index
            batch_value = np.ones_like(x_single) * data_value
            batch_label = np.array([label[i]])
        else:
            batch_len = np.concatenate((batch_len, np.array([x_len])), axis=0)
            batch_rows = np.concatenate((batch_rows, x_single), axis=0)
            batch_cols = np.concatenate((batch_cols, np.arange(x_len)), axis=0)
            batch_file_index = np.concatenate((batch_file_index, np.ones_like(x_single) * i_index), axis=0)
            batch_value = np.concatenate((batch_value, np.ones_like(x_single) * data_value), axis=0)
            batch_label = np.concatenate((batch_label, np.array([label[i]])), axis=0)

    return batch_len, batch_rows, batch_cols, batch_file_index, batch_value, batch_label

# Load dataset (AG_NEWS from torchtext)
train_iter, test_iter = AG_NEWS('../dataset/', split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

# Get train/test label and data_file(tokens) from data_iter and convert them into clean file
stop_words = ['<unk>']# Defined by customer, as musch as possible
train_label, train_file = file_from_iter(train_iter, clean=True, tokenizer=tokenizer, stop_words=stop_words)
test_label, test_file = file_from_iter(test_iter, clean=True, tokenizer=tokenizer, stop_words=stop_words)
# Take part of dataset for convenience
train_label = train_label[:5000]
train_file = train_file[:5000]
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

# Build batch of word2index
batch_process = BatchProcess(tokenizer, voc)
train_file_index = batch_process.index_from_file(train_file)
test_file_index = batch_process.index_from_file(test_file)

print('Data has been processed!')
data_value = 25
batch_len_tr, batch_rows_tr, batch_cols_tr, batch_file_index_tr, batch_value_tr, batch_label_tr = process_sparse_data(train_file_index, train_label)
batch_len_te, batch_rows_te, batch_cols_te, batch_file_index_te, batch_value_te, batch_label_te = process_sparse_data(test_file_index, test_label)

# create the model and deploy it on gpu or cpu
model = CPGBN([200, 100, 50], 'gpu')
# mode 1, dense input
model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(train_file) - delete_count, len(voc), np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(train_file) - delete_count, len(voc), np.max(batch_len_tr)])
train_local_params = model.test(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(train_file) - delete_count, len(voc), np.max(batch_len_tr)])
test_local_params = model.test(100, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(test_file) - delete_count, len(voc), np.max(batch_len_te)])

train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T

# Score of test dataset's Theta: 0.682
results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
model.save()

# # Customer dataset

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
# def process_sparse_data(file_index, label):
#     delete_count = 0
#     for i in range(len(file_index)):
#         x_single = np.reshape(file_index[i], [len(file_index[i])]).astype(np.int32)
#         x_len = x_single.shape[0]
#         i_index = i - delete_count
#         if i_index == 0:
#             batch_len = np.array([x_len])
#             batch_rows = x_single
#             batch_cols = np.arange(x_len)
#             batch_file_index = np.ones_like(x_single) * i_index
#             batch_value = np.ones_like(x_single) * data_value
#             batch_label = np.array([label[i]])
#         else:
#             batch_len = np.concatenate((batch_len, np.array([x_len])), axis=0)
#             batch_rows = np.concatenate((batch_rows, x_single), axis=0)
#             batch_cols = np.concatenate((batch_cols, np.arange(x_len)), axis=0)
#             batch_file_index = np.concatenate((batch_file_index, np.ones_like(x_single) * i_index), axis=0)
#             batch_value = np.concatenate((batch_value, np.ones_like(x_single) * data_value), axis=0)
#             batch_label = np.concatenate((batch_label, np.array([label[i]])), axis=0)
#
#     return batch_len, batch_rows, batch_cols, batch_file_index, batch_value, batch_label
#
# batch_len_tr, batch_rows_tr, batch_cols_tr, batch_file_index_tr, batch_value_tr, batch_label_tr = process_sparse_data(train_file_index, train_label)
# batch_len_te, batch_rows_te, batch_cols_te, batch_file_index_te, batch_value_te, batch_label_te = process_sparse_data(test_file_index, test_label)
#
# # create the model and deploy it on gpu or cpu
# model = CPGBN([200, 100, 50], 'gpu')
# # mode 1, dense input
# model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
# train_local_params = model.train(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
# train_local_params = model.test(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
# test_local_params = model.test(100, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(data_test_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_te)])
#
# train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
# test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T
#
# # Score of test dataset's Theta: 0.682
# results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
# model.save()
