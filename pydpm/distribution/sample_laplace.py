"""
===========================================
Laplace Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.laplace(2, 0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.laplace(2, 0.5, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=50, density=True)
plt.title('DSG.laplace')
plt.show()

plt.figure()
plt.hist(b, bins=50, density=True)
plt.title('numpy.random.laplace')
plt.show()

"""


import numpy as np
from pycuda.compiler import SourceModule
from .pre_process import para_preprocess
import pycuda.autoinit
import pydpm.distribution.compat


Sampler = SourceModule("""
#include <stdio.h>

__device__ int cudarand(long long seed)
{
    if (seed == 0)
    {
        seed = 1;
    }
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

__global__ void rand_Laplace(float *randomseed, float *target, int * matrix_scale,float *miu, float *gamma)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int current_index = idx/matrix_scale[1];
        float a = miu[current_index];
        float b = gamma[current_index];
        float uniform = randomseed[idx]-0.5;
        int sign = uniform==0?0:uniform/abs(uniform);
        target[idx]=a - b*sign*log(1-2*abs(uniform));
    }
}

""")

def laplace(miu, gamma, times=1,device='cpu'):

    miu, gamma, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, miu, gamma)
    func = Sampler.get_function('rand_Laplace')
    func(randomseed, output, matrix_scale, miu, gamma, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output