#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause
import copy
import pickle
import numpy as np

from scipy.sparse import csr_matrix, isspmatrix
from collections import Counter, OrderedDict

import torch
import torch.nn.utils.rnn as rnn

import torchtext.datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split



class Text_Processer(object):
    def __init__(self, tokenizer=None, vocab=None):

        assert tokenizer!=None, Warning('The tokenizer should be initialized')
        assert vocab!=None, Warning('The vocab should be initialized')

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_length = len(vocab)

    def read_file(self, file_path):
        with open(file_path, encoding='utf8') as f:
            text = f.readlines()
        return text

    def word2index(self, word):
        return self.vocab[word]

    def file_from_iter(self, iter: list, tokenizer=None, stop_words=None):
        '''
            Return: tokens
        '''
        files = []
        file_labels = []

        if stop_words:
            for label, data in iter:
                tokens = tokenizer(data)
                file = []
                for token in tokens:
                    if token not in stop_words:
                        file.append(token)
                files.append(tokens)
                file_labels.append(label)
        else:
            for label, data in iter:
                tokens = tokenizer(data)
                file = tokens
                files.append(file)
                file_labels.append(label)

        return files, file_labels

    def word_index_from_file(self, files: list, file_labels: list, min_tokens: int=0, to_sparse: bool=False):
        '''
            Return
        '''
        if not to_sparse:
            word_indices = []
            batch_labels = []
            for file, file_label in zip(files, file_labels):
                token_index = self.vocab(file)
                num_tokens = len(token_index)
                if num_tokens < min_tokens:
                    continue
                else:
                    word_indices.append(token_index)
                    batch_labels.append(file_label)

            return word_indices, batch_labels

        else:
            file_index = 0
            word_value = 25
            for file, file_label in zip(files, file_labels):
                token_index = self.vocab(file)
                num_tokens = len(token_index)
                if num_tokens < min_tokens:
                    continue
                else:
                    if file_index == 0:
                        batch_lens = np.array([num_tokens], dtype=np.int32)
                        batch_rows = np.array([token_index], dtype=np.int32).squeeze()
                        batch_cols = np.arange(num_tokens)
                        batch_file_indices = np.ones_like(token_index, dtype=np.int32) * file_index
                        batch_values = np.ones_like(token_index, dtype=np.int32) * word_value
                        batch_labels = np.array([file_label], dtype=np.int32)
                    else:
                        batch_lens = np.concatenate((batch_lens, np.array([num_tokens], dtype=np.int32)), axis=0)
                        batch_rows = np.concatenate((batch_rows, np.array([token_index], dtype=np.int32).squeeze()), axis=0)
                        batch_cols = np.concatenate((batch_cols, np.arange(num_tokens)), axis=0)
                        batch_file_indices = np.concatenate((batch_file_indices, np.ones_like(token_index, dtype=np.int32) * file_index), axis=0)
                        batch_values = np.concatenate((batch_values, np.ones_like(token_index, dtype=np.int32) * word_value), axis=0)
                        batch_labels = np.concatenate((batch_labels, np.array([file_label], dtype=np.int32)), axis=0)

                    file_index += 1  # file_index will increase if num_tokens is large than min_tokens

            sparse_data = [batch_rows, batch_cols, batch_file_indices, batch_values]
            sparse_shape = [len(batch_labels), len(self.vocab), np.max(batch_lens)]

            return [sparse_data, sparse_shape], batch_labels

    def tfidf(self, bow):
        if isspmatrix(bow):
            bow = bow.todense()

        def tf(bow):
            # tf_{j,v} = n_{j,v} / n_{j,.}
            return bow / (np.sum(bow, axis=1, keepdims=True) + 1e-10)  ### n,v

        def idf(bow):
            # idf_{v} = log(N / n_{v})
            N = bow.shape[0]
            bow_binary = np.array(bow, dtype=np.bool) * 1.0
            return np.log(1.0 * N / (np.sum(bow_binary, axis=0, keepdims=True) + 1e-10) + 1e-10)  ### v,1

        return tf(1.0 * bow) * idf(1.0 * bow)  # n,v

    def bow_from_file(self, files: list, file_labels: list, min_tokens: int=0, to_sparse=False):
        '''
            Input: tokens
        '''
        if not to_sparse:
            batch_bows = []
            batch_labels = []
            for file, file_label in zip(files, file_labels):
                token_index = self.vocab(file)
                num_tokens = len(token_index)
                if num_tokens < min_tokens:
                    continue

                file_bow = np.zeros(len(self.vocab))
                token_counts = Counter(file)
                for token in token_counts.keys():
                    file_bow[self.vocab[token]] = token_counts[token]

                batch_bows.append(file_bow)
                batch_labels.append(file_label)

            return batch_bows, batch_labels
        else:
            #TODO
            pass

            return

    # =====================================================================
    def collate_bow_batch(self, batch_data: list):
        '''
            batch_data: a list of [word_index, file_label]
        '''
        batch_bows = []
        batch_labels = []
        for word_index, file_label in batch_data:
            file_bow = np.zeros(len(self.vocab))
            word_counts = Counter(word_index)
            for word_key in word_counts.keys():
                file_bow[word_key] = word_counts[word_key]

            batch_bows.append(file_bow)
            batch_labels.append(file_label)

        return torch.tensor(batch_bows, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)

    def collate_sequence_batch(self, batch_data: list):
        '''
            batch_data: a list of [word_index, file_label]
        '''
        word_indices = []
        batch_labels = []
        batch_lens = []
        for word_index, file_label in batch_data:
            word_indices.append(torch.tensor([self.voc['<bos>']] + word_index + [self.voc['<eos>']]))
            batch_labels.append(file_label)
            batch_lens.append(len(word_index))

        batch_word_indices = rnn.pad_sequence(word_indices, padding_value=self.vocab['<pad>'], batch_first=True)

        return batch_word_indices, torch.tensor(batch_labels, torch.long), torch.tensor(batch_lens, torch.long)


# ======================================== CustomDataset ======================================================== #

# Dataset and dataloader designed by customer
class CustomDataset(Dataset):

    def __init__(self, data_file, dataname='20ng', mode='train'):
        self.mode = mode
        self.dataname = dataname
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        if dataname == '20ng_20':
            self.data = data['bow']
            self.label = np.squeeze(data['label20'])
            self.voc = data['voc']
            train_id = data['train_id']
            test_id = data['test_id']
            if mode == 'train':
                self.data = self.data[train_id]
                self.label = self.label[train_id]
            elif mode == 'test':
                self.data = self.data[test_id]
                self.label = self.label[test_id]

        if dataname == '20ng_6':
            self.data = data['bow']
            self.label = np.squeeze(data['label6'])
            self.voc = data['voc']
            train_id = data['train_id']
            test_id = data['test_id']
            if mode == 'train':
                self.data = self.data[train_id]
                self.label = self.label[train_id]
            elif mode == 'test':
                self.data = self.data[test_id]
                self.label = self.label[test_id]

        elif dataname == 'reuters':
            self.data = data['bow']
            self.voc = data['voc']
            self.label = np.zeros(self.data.shape[0])

        elif dataname == 'rcv2':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow']
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow']
                self.label = data['test_label']

        elif dataname == 'web':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow'].T
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow'].T
                self.label = data['test_label']

        elif dataname == 'tmn':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow'].T
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow'].T
                self.label = data['test_label']


        elif dataname == 'dp':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow'].T
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow'].T
                self.label = data['test_label']
        # try:
        #     self.tfidf = tfidf(self.dataset)
        # except:
        #     self.dataset = self.dataset.toarray()
        #     self.tfidf = tfidf(self.dataset)
        if self.dataname == 'dp':
            self.tfidf = self.data

        elif self.dataname == 'rcv2':
            self.data = self.data.toarray()
            self.tfidf = tfidf(self.data)

        else:
            try:
                self.tfidf = tfidf(self.data)
            except:
                self.data = self.data.toarray()
                self.tfidf = tfidf(self.data)

    def __getitem__(self, index):
        if self.dataname == 'dp':
            bow = np.squeeze(self.data[index].toarray())
            tfidf_data = bow
        else:
            bow = self.data[index]
            tfidf_data = self.tfidf[index]
        return bow, tfidf_data, np.squeeze(self.label[index])

    def __len__(self):
        return self.data.shape[0]

def text_dataloader(data_file, dataname='20ng_6', mode='train', batch_size=500, shuffle=True, drop_last=True, num_workers=4):
        dataset = CustomDataset(data_file, dataname=dataname, mode=mode)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                   drop_last=drop_last), dataset.voc








