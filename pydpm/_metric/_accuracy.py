"""
===========================================
Metric to evaluate the performance of the classification
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

from sklearn import svm
import numpy as np

class ACC(object):

    def __init__(self, x_tr: np.ndarray, x_te: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray, model='SVM'):
        '''
        Inputs:
            x_tr : [np.ndarray] K*N_train matrix, N_train latent features of length K
            x_te : [np.ndarray] K*N_test matrix, N_test latent features of length K
            y_tr : [np.ndarray] N_train vector, labels of N_train latent features
            y_te : [np.ndarray] N_test vector, labels of N_test latent features

        Outputs:
            accuracy: [float] scalar, the accuracy score

        '''
        self.x_tr = x_tr
        self.x_te = x_te
        self.y_tr = y_tr
        self.y_te = y_te

        if model == 'SVM':
            self._svm()
        else:
            print("Please input metric model correctly. Options: 'SVM'")

        print(f'The classification accuracy with {model} is: {self._accuracy:.4f}')


    def _svm(self):

        self.model = svm.SVC()
        self.model.fit(self.x_tr.T, self.y_tr)
        print(f'Optimization Finished')
        self._accuracy = self.model.score(self.x_te.T, self.y_te)

