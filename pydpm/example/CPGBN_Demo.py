"""
================================================
Convolutional Poisson Gamma Belief Network Demo
================================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0
from pydpm.model import CPGBN
import numpy as np
import scipy.io as sio

train_data = sio.loadmat("./mnist_gray.mat")
data = np.array(np.ceil(train_data['train_mnist'] * 5), order='C')[:,:999]  # 0-1
data=np.transpose(data)
data = np.reshape(data,[data.shape[0],28,28])
#GPU only
#dense input
model=CPGBN([200,100,50])
model.initial(data)
model.train(100)

#sparse input
X_file_index,X_rows,X_cols=np.where(data)
X_value = data[X_file_index,X_rows,X_cols]
N,V,L=data.shape
model=CPGBN([200,100,50])
model.initial([[X_file_index,X_rows,X_cols,X_value],[N,V,L]],'sparse')
model.train(100)