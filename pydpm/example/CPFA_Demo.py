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
import scipy.io as sio

from pydpm._metric import ACC
from pydpm._model import CPFA

import _pickle as cPickle

DATA = cPickle.load(open("./data/TREC.pkl", "rb"), encoding='iso-8859-1')

data_vab_list = DATA['Vocabulary']
data_vab_count_list = DATA['Vab_count']
data_vab_length = DATA['Vab_Size']
data_label = DATA['Label']
data_train_list = DATA['Train_Origin']
data_train_label = np.array(DATA['Train_Label'])
data_train_split = DATA['Train_Word_Split']
data_train_list_index = DATA['Train_Word2Index']
data_test_list = DATA['Test_Origin']
data_test_label = np.array(DATA['Test_Label'])
data_test_split = DATA['Test_Word_Split']
data_test_list_index = DATA['Test_Word2Index']
data_value = 25

# ======================= Preprocess =======================#
delete_count = 0

for i in range(len(data_train_list)):

    x_single = np.reshape(data_train_list_index[i], [len(data_train_list_index[i])]).astype(np.int32)
    x_len = x_single.shape[0]

    i_index = i - delete_count
    if i_index == 0:
        batch_len = np.array([x_len])
        batch_rows = x_single
        batch_cols = np.arange(x_len)
        batch_file_index = np.ones_like(x_single) * i_index
        batch_value = np.ones_like(x_single) * data_value
        batch_label = np.array([data_train_label[i]])
    else:
        batch_len = np.concatenate((batch_len, np.array([x_len])), axis=0)
        batch_rows = np.concatenate((batch_rows, x_single), axis=0)
        batch_cols = np.concatenate((batch_cols, np.arange(x_len)), axis=0)
        batch_file_index = np.concatenate((batch_file_index, np.ones_like(x_single) * i_index), axis=0)
        batch_value = np.concatenate((batch_value, np.ones_like(x_single) * data_value), axis=0)
        batch_label = np.concatenate((batch_label, np.array([data_train_label[i]])), axis=0)

batch_len_tr = batch_len
batch_rows_tr = batch_rows
batch_cols_tr = batch_cols
batch_file_index_tr = batch_file_index
batch_value_tr = batch_value
batch_label_tr = batch_label

# ======================= Preprocess =======================#
delete_count = 0

for i in range(len(data_test_list)):

    x_single = np.reshape(data_test_list_index[i], [len(data_test_list_index[i])]).astype(np.int32)
    x_len = x_single.shape[0]

    i_index = i - delete_count
    if i_index == 0:
        batch_len = np.array([x_len])
        batch_rows = x_single
        batch_cols = np.arange(x_len)
        batch_file_index = np.ones_like(x_single) * i_index
        batch_value = np.ones_like(x_single) * data_value
        batch_label = np.array([data_test_label[i]])
    else:
        batch_len = np.concatenate((batch_len, np.array([x_len])), axis=0)
        batch_rows = np.concatenate((batch_rows, x_single), axis=0)
        batch_cols = np.concatenate((batch_cols, np.arange(x_len)), axis=0)
        batch_file_index = np.concatenate((batch_file_index, np.ones_like(x_single) * i_index), axis=0)
        batch_value = np.concatenate((batch_value, np.ones_like(x_single) * data_value), axis=0)
        batch_label = np.concatenate((batch_label, np.array([data_test_label[i]])), axis=0)

batch_len_te = batch_len
batch_rows_te = batch_rows
batch_cols_te = batch_cols
batch_file_index_te = batch_file_index
batch_value_te = batch_value
batch_label_te = batch_label


# data = sio.loadmat('./mnist_gray')
# train_data = np.array(np.ceil(data['train_mnist']*25), order='C')[:, 0:999]
# test_data = np.array(np.ceil(data['train_mnist']*25), order='C')[:, 1000:1999]
# train_data = np.transpose(np.transpose(train_data).reshape([train_data.shape[1],  28, 28]), [0, 2, 1])
# test_data = np.transpose(np.transpose(test_data).reshape([test_data.shape[1], 28, 28]), [0, 2, 1])
# train_label = data['train_label'][:999]
# test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = CPFA(200, 'gpu')
# mode 1, dense input
model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
train_local_params = model.test(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
test_local_params = model.test(100, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(data_test_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_te)])

train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T

# train_theta[np.where(np.isinf((train_theta)))] = 0

# Score of test dataset's Theta: 0.682
results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
model.save()



# mode 2, sparse input
# X_file_index, X_rows, X_cols = np.where(train_data)
# X_value = train_data[X_file_index, X_rows, X_cols]
# N, V, L = train_data.shape
# model = CPFA(kernel=100)
# model.initial([[X_file_index, X_rows, X_cols, X_value], [N, V, L]], dtype='sparse')
# model.train(iter_all=100)







# #pgcn demo
# train_data = sio.loadmat("./mnist_gray.mat")
# data = np.array(np.ceil(train_data['train_mnist'] * 5), order='C')[:,:999]  # 0-1
# data=np.transpose(data)
# data = np.reshape(data, [data.shape[0], 28, 28])
# #GPU only
#
# #dense input
# model=CPFA(kernel=100)
# model.initial(data)
# model.train(iter_all=100)
#
# #sparse input
# X_file_index, X_rows, X_cols = np.where(data)
# X_value = data[X_file_index, X_rows, X_cols]
# N,V,L = data.shape
# model = CPFA(kernel=100)
# model.initial([[X_file_index, X_rows, X_cols, X_value], [N,V,L]], dtype='sparse')
# model.train(iter_all=100)