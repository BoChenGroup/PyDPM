#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np

class Purity(object):

	def __init__(self, y, ypred):
		"""
		Inputs:
			y: the ground_true, shape:(n_sample,)
			ypred: pred_label, shape:(n_sample,)
		Output:
			accuracy of cluster, in [0, 1]
		"""
		self.y = y
		self.ypred = ypred

		self._get()

		print(f'The cluster purity is: {self._purity:.4f}')

	def _get(self):

		clusters = np.unique(self.ypred)
		counts = []
		for c in clusters:
			indices = np.where(self.ypred == c)[0]
			max_votes = np.bincount(self.y[indices]).max()
			counts.append(max_votes)
		self._purity = sum(counts) / self.y.shape[0]
