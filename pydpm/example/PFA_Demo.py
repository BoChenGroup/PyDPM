"""
===========================================
Poisson Factor Analysis Demo
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
import scipy.io as sio
data = sio.loadmat('./mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]   #0-1


#PFA demo
from pydpm.model import PFA

test_model = PFA(128, device='gpu')
#test_model = PFA(128, device='cpu')
test_model.initial(train_data)
test_model.train(100)
