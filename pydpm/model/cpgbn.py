"""
===========================================
Convolutional Poisson Gamma Belief Network
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from pydpm.utils.Metric import *


# GPU only
class CPGBN(object):

    def __init__(self, K, device='gpu'):

        self.K = K
        #self.T = self.K.size
        if device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Crt_Matrix = Model_Sampler_GPU.Crt_Matrix_GPU
            self.Crt_Multirnd_Matrix = Model_Sampler_GPU.Crt_Multirnd_Matrix_GPU
            self.Conv_Multi_Matrix = Model_Sampler_GPU.conv_multi_sample
        else:
            raise Exception('device type error')


    def initial(self, data ,dtype="dense"):

        # data: N*V*L
        if dtype == 'dense':
            self.batch_file_index, self.batch_rows, self.batch_cols = np.where(data)
            self.batch_value = data[self.batch_file_index, self.batch_rows, self.batch_cols]
            N, V, L = data.shape
        elif dtype == 'sparse':
            self.batch_file_index, self.batch_rows, self.batch_cols, self.batch_value = data[0]
            N, V, L = data[1]
        else:
            raise Exception('data type error')

        self.batch_len= L
        self.Setting = {}
        self.Setting['N_train'] = N
        # 1-th layer
        self.Setting['K1'] = self.K[0]
        self.Setting['K1_V1'] = V
        self.Setting['K1_V2'] = L + 2  # padding
        self.Setting['K1_S3'] = V
        self.Setting['K1_S4'] = 3
        self.Setting['K1_S1'] = self.Setting['K1_V1'] + 1 - self.Setting['K1_S3']
        self.Setting['K1_S2'] = self.Setting['K1_V2'] + 1 - self.Setting['K1_S4']
        # 2-th layer
        self.Setting['K2'] = self.K[1]
        self.Setting['K2_V1'] = self.Setting['K1_S1']
        self.Setting['K2_V2'] = self.Setting['K1_S2'] + 2  # padding
        self.Setting['K2_S3'] = 1
        self.Setting['K2_S4'] = 3
        self.Setting['K2_S1'] = self.Setting['K2_V1'] + 1 - self.Setting['K2_S3']
        self.Setting['K2_S2'] = self.Setting['K2_V2'] + 1 - self.Setting['K2_S4']
        # 3-th layer
        self.Setting['K3'] = self.K[2]
        self.Setting['K3_V1'] = self.Setting['K2_S1']
        self.Setting['K3_V2'] = self.Setting['K2_S2'] + 2  # padding
        self.Setting['K3_S3'] = 1
        self.Setting['K3_S4'] = 3
        self.Setting['K3_S1'] = self.Setting['K3_V1'] + 1 - self.Setting['K3_S3']
        self.Setting['K3_S2'] = self.Setting['K3_V2'] + 1 - self.Setting['K3_S4']

        # ======================= self.SuperParams =======================#
        self.SuperParams = {}
        self.SuperParams['gamma0'] = 0.1  # r
        self.SuperParams['c0'] = 0.1
        self.SuperParams['a0'] = 0.1  # p
        self.SuperParams['b0'] = 0.1
        self.SuperParams['e0'] = 0.1  # c
        self.SuperParams['f0'] = 0.1
        self.SuperParams['eta'] = 0.05  # Phi
        # ======================= self.Params =======================#
        self.Params = {}

        # 1-th layer
        self.Params['D1_k1'] = np.random.rand(self.Setting['K1'], self.Setting['K1_S3'], self.Setting['K1_S4'])
        for k1 in range(self.Setting['K1']):
            self.Params['D1_k1'][k1, :, :] = self.Params['D1_k1'][k1, :, :] / np.sum(self.Params['D1_k1'][k1, :, :])
        self.Params['W1_nk1'] = np.random.rand(self.Setting['N_train'], self.Setting['K1'], self.Setting['K1_S1'], self.Setting['K1_S2'])
        self.Params['W1_nk1_Pooling'] = np.sum(np.sum(self.Params['W1_nk1'], axis=3), axis=2)

        self.Params['c2_n'] = 1 * np.ones([self.Setting['N_train']])
        self.Params['p2_n'] = 1 / (1 + self.Params['c2_n'])

        # 2-th layer
        self.Params['Phi_2'] = 0.2 + 0.8 * np.random.rand(self.Setting['K1'], self.Setting['K2'])
        self.Params['Phi_2'] = self.Params['Phi_2'] / np.sum(self.Params['Phi_2'], axis=0)
        self.Params['Theta_2'] = np.random.rand(self.Setting['N_train'], self.Setting['K2'])

        self.Params['c3_n'] = 1 * np.ones([self.Setting['N_train']])
        tmp = -log_max(1 - self.Params['p2_n'])
        self.Params['p3_n'] = (tmp / (tmp + self.Params['c3_n']))  # pj_3 - pj_T+1

        # 3-th layer
        self.Params['Phi_3'] = 0.2 + 0.8 * np.random.rand(self.Setting['K2'], self.Setting['K3'])
        self.Params['Phi_3'] = self.Params['Phi_3'] / np.sum(self.Params['Phi_3'], axis=0)
        self.Params['Theta_3'] = np.random.rand(self.Setting['N_train'], self.Setting['K3'])

        self.Params['c4_n'] = 1 * np.ones([self.Setting['N_train']])
        tmp = -log_max(1 - self.Params['p3_n'])
        self.Params['p4_n'] = (tmp / (tmp + self.Params['c4_n']))  # pj_3 - pj_T+1

        self.Params['Gamma'] = np.ones([self.Setting['K3'], 1]) / self.Setting['K3']

    def train(self, iter_all=200):

        from scipy.special import gamma
        self.Setting['Burinin'] = 0.75 * iter_all
        self.Setting['Collection'] = iter_all - self.Setting['Burinin']

        # Collection
        W_train_1 = np.zeros([self.Setting['N_train'], self.Setting['K1']])
        W_train_2 = np.zeros([self.Setting['N_train'], self.Setting['K2']])
        W_train_3 = np.zeros([self.Setting['N_train'], self.Setting['K3']])

        import time
        Iter_time = []
        Iter_lh = []

        # ========================== Gibbs ==========================#
        for t in range(iter_all):

            start_time = time.time()

            # ========================== 1st layer Augmentation ==========================#
            self.Params['D1_k1_Aug'] = np.zeros_like(self.Params['D1_k1'])  # Augmentation on D 
            self.Params['W1_nk1_Aug'] = np.zeros_like(self.Params['W1_nk1'])  # Augmentation on w


            W1_nk1 = np.array(self.Params['W1_nk1'], dtype='float32', order='C')
            D1_k1 = np.array(self.Params['D1_k1'], dtype='float32', order='C')

            W1_nk1_Aug, D1_k1_Aug = self.Conv_Multi_Matrix(self.batch_file_index, self.batch_rows, self.batch_cols, self.batch_value, W1_nk1, D1_k1, self.Setting)


            self.Params['W1_nk1_Aug'] = np.array(W1_nk1_Aug, dtype='float64')  # N*K1*S1*S2
            self.Params['D1_k1_Aug'] = np.array(D1_k1_Aug, dtype='float64')  # K1*S3*S4
            self.Params['W1_nk1_Aug_Pooling'] = np.sum(np.sum(self.Params['W1_nk1_Aug'], axis=3), axis=2)  # N*K1

            # ========================== 2nd layer Augmentation ==========================#
            M1_tmp = np.array(np.transpose(np.round(self.Params['W1_nk1_Aug_Pooling'])), dtype='float64', order='C')
            Theta2_tmp = np.array(np.transpose(self.Params['Theta_2']), dtype='float64', order='C')
            Xt_to_t1_2, WSZS_2 = self.Crt_Multirnd_Matrix(M1_tmp, self.Params['Phi_2'], Theta2_tmp)

            # ========================== 3rd layer Augmentation ==========================#
            M2_tmp = np.array(np.round(Xt_to_t1_2), dtype='float64', order='C')
            Theta3_tmp = np.array(np.transpose(self.Params['Theta_3']), dtype='float64', order='C')
            Xt_to_t1_3, WSZS_3 = self.Crt_Multirnd_Matrix(M2_tmp, self.Params['Phi_3'], Theta3_tmp)

            # ====================== Parameters Update ======================#
            # Update D,Phi
            for k1 in range(self.Setting['K1']):
                X_k1_34 = self.Params['D1_k1_Aug'][k1, :, :]
                X_k1_34_tmp = np.random.gamma(X_k1_34 + self.SuperParams['eta'])
                D1_k1_s = X_k1_34_tmp / np.sum(X_k1_34_tmp)
                self.Params['D1_k1'][k1, :, :] = D1_k1_s

            Phi_2_tmp = np.random.gamma(WSZS_2 + self.SuperParams['eta'])
            self.Params['Phi_2'] = Phi_2_tmp / np.sum(Phi_2_tmp, axis=0)

            Phi_3_tmp = np.random.gamma(WSZS_3 + self.SuperParams['eta'])
            self.Params['Phi_3'] = Phi_3_tmp / np.sum(Phi_3_tmp, axis=0)

            # Update c_j,p_j
            self.Params['c2_n'] = np.random.gamma(
                self.SuperParams['e0'] + np.sum(np.dot(self.Params['Phi_2'], self.Params['Theta_2'].T), 0))
            self.Params['c2_n'] = self.Params['c2_n'] / (self.SuperParams['f0'] + np.sum(self.Params['W1_nk1_Pooling'], axis=1))
            self.Params['p2_n'] = 1 / (self.Params['c2_n'] + 1)

            self.Params['c3_n'] = np.random.gamma(
                self.SuperParams['e0'] + np.sum(np.dot(self.Params['Phi_3'], self.Params['Theta_3'].T), 0))
            self.Params['c3_n'] = self.Params['c3_n'] / (self.SuperParams['f0'] + np.sum(self.Params['Theta_2'], axis=1))
            tmp = -log_max(1 - self.Params['p2_n'])
            self.Params['p3_n'] = tmp / (self.Params['c3_n'] + tmp)

            self.Params['c4_n'] = np.random.gamma(self.SuperParams['e0'] + np.sum(self.Params['Gamma']))
            self.Params['c4_n'] = self.Params['c4_n'] / (self.SuperParams['f0'] + np.sum(self.Params['Theta_3'], axis=1))
            tmp = -log_max(1 - self.Params['p3_n'])
            self.Params['p4_n'] = tmp / (self.Params['c4_n'] + tmp)

            # Update w_j
            W_k3_sn = np.random.gamma(self.Params['Gamma'] + Xt_to_t1_3) / (
                        -log_max(1 - self.Params['p3_n']) + self.Params['c4_n'])  # V*N
            self.Params['Theta_3'] = np.transpose(W_k3_sn)

            shape2 = np.dot(self.Params['Phi_3'], self.Params['Theta_3'].T)
            W_k2_sn = np.random.gamma(shape2 + Xt_to_t1_2) / (-log_max(1 - self.Params['p2_n']) + self.Params['c3_n'])  # V*N
            self.Params['Theta_2'] = np.transpose(W_k2_sn)

            shape1 = np.dot(self.Params['Phi_2'], self.Params['Theta_2'].T)  # V*N
            W_k1_sn = np.random.gamma(shape1 + self.Params['W1_nk1_Aug_Pooling'].T) / (1 + self.Params['c2_n'])  # V*N
            self.Params['W1_nk1_Pooling'] = np.transpose(W_k1_sn)

            for k1 in range(self.Setting['K1']):
                self.Params['W1_nk1'][:, k1, 0, :] = (self.Params['W1_nk1_Aug'][:, k1, 0, :] / (
                            self.Params['W1_nk1_Aug_Pooling'][:, k1:k1 + 1] + 0.0001)) * self.Params['W1_nk1_Pooling'][:,
                                                                                    k1:k1 + 1]

            if t >= self.Setting['Burinin']:
                W_train_1 = W_train_1 + np.sum(self.Params['W1_nk1'][:, :, 0, :], axis=2)
                W_train_2 = W_train_2 + self.Params['Theta_2']
                W_train_3 = W_train_3 + self.Params['Theta_3']

            end_time = time.time()

            if t == 0:
                Iter_time.append(end_time - start_time)
            else:
                Iter_time.append(end_time - start_time + Iter_time[-1])

            print("epoch " + str(t) + " takes " + str(end_time - start_time) + " seconds")


        W_train_1 = W_train_1 / self.Setting['Collection']
        W_train_2 = W_train_2 / self.Setting['Collection']
        W_train_3 = W_train_3 / self.Setting['Collection']


