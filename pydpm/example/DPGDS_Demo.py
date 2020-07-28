"""
===========================================
Deep Poisson Gamma Dynamical Systems Demo
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
import scipy.io as sio
data = sio.loadmat('./mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
from pydpm.model import DPGDS
model=DPGDS([200, 100, 50],'gpu')
#model=DPGDS([200, 100, 50],'cpu')
model.initial(train_data)
model.train(200)


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(model.Likelihood, label="pt")
plt.legend(loc='lower right')
plt.show()
# plt.figure(2)
# plt.plot(test_model.Reconstruct_Error, label="gpu")
# plt.plot(test_model_2.Reconstruct_Error, label="cpu")
# plt.plot(pgbn.layerlist[1].Reconstruct_Error, label="nn")
# #plt.plot(ER2,label="CPU")
# plt.legend(loc='lower right')
# plt.show()