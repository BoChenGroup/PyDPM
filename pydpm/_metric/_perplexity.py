"""
===========================================
per-held-word perplexity
===========================================

"""

import numpy as np
from .._utils import *

class Perplexity(object):
	def __init__(self, x, x_hat):
		'''
		Inputs:
			x: [float] np.ndarray, V*N_test matrix, the observations for test_data
			x_hat: [float] np.ndarray, V*N_reconstruct matrix
		Outputs:
			PPL: [float], the perplexity score
		'''

		self.x = x
		self.x_hat = x_hat

		self._get()

		print(f'The PPL is: {self._PPL:.4f}')

	def _get(self):

		self.x_hat = self.x_hat / (self.x_hat.sum(0) + realmin)
		ppl = -1.0 * self.x * np.log(self.x_hat.T + realmin) / self.x.sum()
		ppl = ppl.sum().exp()

		self._PPL = ppl
