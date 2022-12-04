"""
===========================================
per-held-word perplexity
===========================================

"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
from ..utils import *

class Perplexity(object):
	def __init__(self, x, x_hat):
		'''
		Inputs:
			x: [float] np.ndarray, N*V test matrix, the observations for test_data
			x_hat: [float] np.ndarray, N*V reconstruct matrix
		Outputs:
			PPL: [float], the perplexity score
		'''

		self.x = x
		self.x_hat = x_hat

		self._get()

		print(f'The PPL is: {self._PPL:.4f}')

	def _get(self):

		self.x_hat = self.x_hat / (np.sum(self.x_hat, axis=1, keepdims=True) + realmin)
		ppl = -1.0 * self.x * np.log(self.x_hat + realmin) / np.sum(self.x)
		ppl = np.exp(ppl.sum())

		self._PPL = ppl
