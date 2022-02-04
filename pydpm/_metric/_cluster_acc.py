import numpy as np
from scipy.optimize import linear_sum_assignment

# from sklearn.utils.linear_assignment_ import linear_assignment
# from sklearn.metrics.cluster import normalized_mutual_info_score as NMI, \
#         adjusted_mutual_info_score as AMI, adjusted_rand_score as AR, silhouette_score as SI, calinski_harabasz_score as CH

class Cluster_ACC(object):

	def __init__(self, y, ypred):
		'''
        Inputs:
            y: the ground_true, shape:(n_sample,)
               ypred: pred_label, shape:(n_sample,)

        Outputs:
            accuracy of cluster, in [0, 1]
		'''
		self.y = y
		self.ypred = ypred

		self._get()

		print(f'The cluster accuracy is: {self._cluster_acc:.4f}')

	def _get(self):
		s = np.unique(self.ypred)
		t = np.unique(self.y)

		N = len(np.unique(self.ypred))
		C = np.zeros((N, N), dtype=np.int32)
		for i in range(N):
			for j in range(N):
				idx = np.logical_and(self.ypred == s[i], self.y == t[j])
				C[i][j] = np.count_nonzero(idx)

		# convert the C matrix to the 'true' cost
		Cmax = np.amax(C)
		C = Cmax - C
		indices = linear_sum_assignment(C)
		row = indices[:][:, 0]
		col = indices[:][:, 1]
		# calculating the accuracy according to the optimal assignment
		count = 0
		for i in range(N):
			idx = np.logical_and(self.ypred == s[row[i]], self.y == t[col[i]])
			count += np.count_nonzero(idx)

		self._cluster_acc = 1.0 * count / len(self.y)

		# y_true = y_true.astype(np.int64)
		# assert y_pred.size == y_true.size
		# D = max(y_pred.max(), y_true.max()) + 1
		# w = np.zeros((D, D), dtype=np.int64)
		# for i in range(y_pred.size):
		#     w[y_pred[i], y_true[i]] += 1
		# from sklearn.utils.linear_assignment_ import linear_assignment
		# ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm
		# return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



