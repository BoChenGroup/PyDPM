"""
===========================================
Poisson Gamma Belief Network Demo
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
import scipy.io as sio
data = sio.loadmat('./mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]   #0-1

#PGBN demo
from pydpm.model import PGBN

test_model = PGBN([128, 64, 32], device='gpu')
#test_model = PGBN([128, 64, 32], device='cpu')
test_model.initial(train_data)
test_model.train(100)


#PGBN demo by layer
from pydpm.layer import data_base,prob_layer,model

data = data_base('./mnist_gray')
layer1 = prob_layer(128)
layer2 = prob_layer(64)
layer3 = prob_layer(32)
pgbn = model([data, layer1, layer2, layer3],'gpu')
pgbn.train(iter=100)


# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(test_model.Likelihood, label="gpu")
# plt.legend(loc='lower right')
# plt.show()
# plt.figure(2)
# plt.plot(test_model.Reconstruct_Error, label="gpu")
# plt.legend(loc='lower right')
# plt.show()