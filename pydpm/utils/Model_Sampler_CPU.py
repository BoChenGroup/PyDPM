"""
===========================================
Model Sampler implemented on CPU
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
import numpy.ctypeslib as npct
import ctypes  
from ctypes import *
import os

realmin = 2.2e-10

array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='C')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C')
array_int = npct.ndpointer(dtype=np.int32, ndim=0, flags='C')
ll = ctypes.cdll.LoadLibrary   
try:
    import platform
    if platform.system()=="Windows":
        Multi_lib = ll(os.path.dirname(__file__)+"/libMulti_Sample.dll")
        Crt_lib = ll(os.path.dirname(__file__)+"/libCrt_Sample.dll")
        Crt_Multi_lib = ll(os.path.dirname(__file__)+"/libCrt_Multi_Sample.dll")
    else:
        Multi_lib = ll(os.path.dirname(__file__) + "/libMulti_Sample.so")
        Crt_lib = ll(os.path.dirname(__file__) + "/libCrt_Sample.so")
        Crt_Multi_lib = ll(os.path.dirname(__file__) + "/libCrt_Multi_Sample.so")
except:
    try:
        Multi_lib = ll(os.path.dirname(__file__)+"/libMulti_Sample.so")
        Crt_lib = ll(os.path.dirname(__file__)+"/libCrt_Sample.so")
        Crt_Multi_lib = ll(os.path.dirname(__file__)+"/libCrt_Multi_Sample.so")
    except:
        raise Exception("can not load cpu lib")
Multi_lib.Multi_Sample.restype = None
Multi_lib.Multi_Sample.argtypes = [array_2d_double, array_2d_double, array_2d_double, array_2d_double, array_2d_double, c_int, c_int, c_int]

Crt_Multi_lib.Crt_Multi_Sample.restype = None
Crt_Multi_lib.Crt_Multi_Sample.argtypes = [array_2d_double, array_2d_double, array_2d_double, array_2d_double, array_2d_double, c_int, c_int, c_int]

Crt_lib.Crt_Sample.restype = None
Crt_lib.Crt_Sample.argtypes = [array_2d_double, array_2d_double, array_2d_double, c_int, c_int]
def Multrnd_Matrix(X_t, Phi_t, Theta_t):

    X_t = np.array(X_t, order='C').astype('double')
    Phi_t = np.array(Phi_t, order='C').astype('double')
    Theta_t = np.array(Theta_t, order='C').astype('double')

    V = X_t.shape[0]
    J = X_t.shape[1]
    K = Theta_t.shape[0]
    Xt_to_t1_t = np.zeros([K, J], order='C').astype('double')
    WSZS_t = np.zeros([V, K], order='C').astype('double')
    Multi_lib.Multi_Sample(X_t, Phi_t, Theta_t, WSZS_t, Xt_to_t1_t, V, K, J)

    return Xt_to_t1_t, WSZS_t



def Crt_Matrix(Xt_to_t1_t, p):

    Xt_to_t1_t = np.array(Xt_to_t1_t, order='C')
    p = np.array(p, order='C')

    K_t = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    X_t1 = np.zeros([K_t, J], order='C').astype('double')

    Crt_lib.Crt_Sample(Xt_to_t1_t, p, X_t1, K_t, J)

    return X_t1



def Crt_Multirnd_Matrix(Xt_to_t1_t, Phi_t1, Theta_t1):

    Xt_to_t1_t = np.array(Xt_to_t1_t, order='C').astype('double')
    Phi_t1 = np.array(Phi_t1, order='C').astype('double')
    Theta_t1 = np.array(Theta_t1, order='C').astype('double')

    K_t = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    K_t1 = Theta_t1.shape[0]
    Xt_to_t1_t1 = np.zeros([K_t1, J], order='C').astype('double')
    WSZS_t1 = np.zeros([K_t, K_t1], order='C').astype('double')
    
    Crt_Multi_lib.Crt_Multi_Sample(Xt_to_t1_t, Phi_t1, Theta_t1, WSZS_t1, Xt_to_t1_t1, K_t, K_t1, J)

    return Xt_to_t1_t1, WSZS_t1


def Sample_Pi(WSZS, Eta):
    Phi = np.random.gamma(WSZS + Eta)
    tmp = np.sum(Phi, axis=0)
    temp_dex = np.where(tmp > 0)
    temp_dex_no = np.where(tmp <= 0)
    Phi[:, temp_dex] = Phi[:, temp_dex] / tmp[temp_dex]
    Phi[:, temp_dex_no] = 0
    return Phi


def Sample_Delta(X_train, Theta, eps, Station):
    delta = np.ones([Theta.shape[1], 1])
    if (Station == 0):
        shape = eps + np.sum(X_train, axis=0)
        scale = eps + np.sum(Theta, axis=0)
        delta = np.random.gamma(shape) / scale
    else:
        shape = eps + np.sum(X_train)
        scale = eps + np.sum(Theta)
        delta = np.random.gamma(shape) / scale

    return delta
