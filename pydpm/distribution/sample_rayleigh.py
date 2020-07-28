"""
===========================================
Rayleigh Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.rayleigh(0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.rayleigh(0.5, 1000000)
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

__device__ int cudarand(long long seed)
{
    if (seed == 0)
    {
        seed = 1;
    }
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

__global__ void rand_Rayleigh(float* randomseed, float* target, int* matrix_scale, float* scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        float U, V;
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        int current_index=idx/matrix_scale[1];
        float theta=scale[current_index];
        float z,x;
        seed=cudarand(seed);
        U = seed / 2147483647.0;
        seed=cudarand(seed);
        V = seed / 2147483647.0;
        z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V)*theta;
        
        seed=cudarand(seed);
        U = seed / 2147483647.0;
        seed=cudarand(seed);
        V = seed / 2147483647.0;
        x = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V)*theta;
        //x = sqrt(-2.0 * log(V))* cos(2.0 * 3.141592654 * U)*theta;
        
        target[idx]=sqrt(z*z+x*x);
    }
}

""")

def rayleigh(scale=1.0, times=1, device='cpu'):

    scale, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, scale)
    func = Sampler.get_function('rand_Rayleigh')
    func(randomseed, output, matrix_scale, scale,
         grid=(partition[0], 1, 1), block=(partition[1], 1, 1))
    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output

