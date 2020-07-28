"""
===========================================
Geometric Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.geometric(0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.geometric(0.5, 1000000)
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

__global__ void rand_Geometric(float* randomseed, int* target, int* matrix_scale, float* prob)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        int current_index=idx/matrix_scale[1];
        float p=prob[current_index];
        int rnd=0; 
        int seed= randomseed[idx]*2147483647.0;
        while(++rnd)
        {
            seed = cudarand(seed);
            if (seed/2147483647.0 < p)
            {
                target[idx]=rnd;
                return;
            }
            if (rnd>1000)
            {
                target[idx]=rnd;
                return;
            }
        }
    }
}


""")

def geometric(p, times=1,device='cpu'):

    p, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.int32, p)
    func = Sampler.get_function('rand_Geometric')
    func(randomseed, output, matrix_scale, p,
         grid=(partition[0], 1, 1), block=(partition[1], 1, 1))
    if device=='cpu':
        output=output.get()
    if single_number:
        return output[0]
    return output
