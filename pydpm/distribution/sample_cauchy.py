"""
===========================================
Cauchy Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

need to be checked

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.cauchy(0.5, 0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

plt.figure()
plt.hist(a, bins=50, density=True)
plt.title('cauchy')
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

__global__ void rand_Cauchy(float* randomseed, float* target, int* matrix_scale, float* x, float* gamma)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        float uniform = randomseed[idx] - 0.5;
        int current_index=idx/matrix_scale[1];
        float pi=3.1415926;
        target[idx]=x[current_index]+tan(uniform*pi)*gamma[current_index]; // rows first
    }
}

""")

def cauchy(x=0, gamma=1,times=1,device='cpu'):

    x, gamma, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, x, gamma)
    func = Sampler.get_function('rand_Cauchy')

    func(randomseed, output, matrix_scale, x, gamma,
         grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output

def standard_cauchy(size=1,device='cpu'):

    x = np.zeros(size)
    gamma = np.ones(size)
    x, gamma, output, matrix_scale, randomseed, partition, single_number = para_preprocess(1, np.float32, np.float32, x, gamma)

    func = Sampler.get_function('rand_Cauchy')
    func(randomseed, output, matrix_scale, x, gamma, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output