#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

class ROC_AP_SCORE(object):

    def __init__(self, edges_pos, edges_neg, adj_orig, emb=None):

        self.edges_pos = edges_pos
        self.edges_neg = edges_neg
        self.adj_orig = adj_orig
        self.emb = emb

        self._get()

        # print(f'The AUC is:  {self._AUC:.4f} and AP is:   {self._AP:.4f}')

    def _get(self):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_decoder_a, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def beta(x):
            return 1 - np.exp(-x)

        # Predict on test set of edges
        adj_rec = np.dot(self.emb, self.emb.T)
        preds = []
        pos = []
        # print(adj_rec,'**************')
        for e in self.edges_pos:
            # preds.append(sigmoid(adj_rec[e[0], e[1]]))
            # preds.append(adj_rec[e[0], e[1]])
            preds.append(beta(adj_rec[e[0], e[1]]))
            pos.append(self.adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            # preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            # preds_neg.append(adj_rec[e[0], e[1]])
            preds_neg.append(beta(adj_rec[e[0], e[1]]))
            neg.append(self.adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        self._AUC = roc_score
        self._AP = ap_score
        # return roc_score, ap_score
