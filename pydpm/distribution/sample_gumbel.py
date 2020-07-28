"""
===========================================
Gumbel Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.gumbel(0.5, 0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.gumbel(0.5, 0.5, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=50, density=True)
plt.title('DSG.beta')
plt.show()

plt.figure()
plt.hist(b, bins=50, density=True)
plt.title('numpy.random.beta')
plt.show()

"""


import numpy as np
from pycuda.compiler import SourceModule
from .pre_process import para_preprocess
import pycuda.autoinit
import pydpm.distribution.compat


Sampler = SourceModule("""
#include <stdio.h>

__global__ void rand_Gumbel(float* randomseed, float* target, int* matrix_scale, float* miu, float* beta)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        float uniform = randomseed[idx]+(1e-20);
        int current_index = idx/matrix_scale[1];
        target[idx]= miu[current_index] - beta[current_index] * log(-log(uniform));
    }
}

""")

def gumbel(miu=0, beta=1, times=1,device='cpu'):

    miu, beta, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, miu, beta)
    func = Sampler.get_function('rand_Gumbel')
    func(randomseed, output, matrix_scale, miu, beta, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output