"""
===========================================
Bidirectional Convolutional Poisson Gamma Dynamical Systems Demo
Wenchao Chen, Chaojie Wang, Bo Cheny, Yicheng Liu, Hao Zhang
Published in Neural Information Processing Systems 2020

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Wei Zhao <13279389260@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause

import numpy as np
import scipy.io as sio
import _pickle as cPickle

from pydpm._metric import ACC
from pydpm._model import CPGDS


# load data
data = cPickle.load(open('./data/IMDB.pkl', 'rb'), encoding='iso-8859-1')

word2index = data['word2index']
index2word = data['index2word']
num_words = len(index2word)
index2word[num_words] = '<pad_zero>'
word2index['<pad_zero>'] = num_words
split_index = [word2index['.'], word2index['!'], word2index['?'], word2index['..'], word2index[';']]

train_doc_index = data['train_doc_index']
train_doc_label = np.array(data['train_doc_label'])
test_doc_index = data['test_doc_index']
test_doc_label = np.array(data['test_doc_label'])

# ======================= Preprocess: train =======================#
seq_max_len = 0

train_doc_split = []
train_doc_split_len = []
train_doc_len = []
for i in range(len(train_doc_index)):
    [seqs_len, seqs] = Seq_Split(train_doc_index[i], split_index, word2index['<pad_zero>'])
    train_doc_split.append(seqs)
    train_doc_split_len.append(seqs_len)
    tmp_max = Seq_Max_Len(seqs)
    if tmp_max > seq_max_len:
        seq_max_len = tmp_max
    train_doc_len.append(len(seqs))

def my_key(k):
    return train_doc_len[k]

sorted_train_index = sorted(range(len(train_doc_len)), key=my_key)  # 按长度进行排序
train_doc_split = [train_doc_split[i] for i in sorted_train_index]
train_doc_len = [train_doc_len[i] for i in sorted_train_index]
train_doc_label = [train_doc_label[i] for i in sorted_train_index]


# ======================= Preprocess: test  =======================#
seq_max_len = 0
test_doc_split = []
test_doc_split_len = []
test_doc_len = []
for i in range(len(test_doc_index)):
    [seqs_len, seqs] = Seq_Split(test_doc_index[i], split_index, word2index['<pad_zero>'])
    test_doc_split.append(seqs)
    test_doc_split_len.append(seqs_len)
    tmp_max = Seq_Max_Len(seqs)
    if tmp_max > seq_max_len:
        seq_max_len = tmp_max
    test_doc_len.append(len(seqs))

def my_key(k):
    return test_doc_len[k]

sorted_test_index = sorted(range(len(test_doc_len)), key=my_key)
test_doc_split = [test_doc_split[i] for i in sorted_test_index]
test_doc_len = [test_doc_len[i] for i in sorted_test_index]
test_doc_label = [test_doc_label[i] for i in sorted_test_index]


# create the model and deploy it on gpu or cpu
model = CPGDS(100, 'gpu')
train_data = [word2index, train_doc_split, train_doc_label, train_doc_len]
test_data = [word2index, test_doc_split, test_doc_label, test_doc_len]
model.initial(train_data)
train_local_params = model.train(20, *train_data)
train_local_params = model.test(20, *train_data)
test_local_params = model.test(20, *test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_doc_label, test_doc_label, 'SVM')

# save the model after training
model.save()
# model.load('./save_models/CDPGDS.npy')


