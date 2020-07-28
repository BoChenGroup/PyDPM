"""
===========================================
Convolutional Poisson Factor Analysis Demo
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
import scipy.io as sio

from pydpm.utils.Metric import *
from pydpm.model import CPFA
#pgcn demo
train_data = sio.loadmat("./mnist_gray.mat")
data = np.array(np.ceil(train_data['train_mnist'] * 5), order='C')[:,:999]  # 0-1
data=np.transpose(data)
data = np.reshape(data,[data.shape[0], 28, 28])
#GPU only
#dense input
model=CPFA(kernel=100)
model.initial(data)
model.train(iter_all=100)

#sparse input
X_file_index,X_rows,X_cols=np.where(data)
X_value = data[X_file_index,X_rows,X_cols]
N,V,L=data.shape
model=CPFA(kernel=100)
model.initial([[X_file_index,X_rows,X_cols,X_value],[N,V,L]], dtype='sparse')
model.train(iter_all=100)