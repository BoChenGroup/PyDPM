"""
===========================================
Poisson Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.poisson(0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.poisson(0.5, 1000000)
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
import pycuda.autoinit
import pydpm.distribution.compat
from pydpm.distribution.pre_process import para_preprocess

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
__device__ int cudaseed(float seed ,long idx)
    {
        clock_t start = clock();
        int iseed = int(seed*2147483647)%2147483647;
        long long nseed = iseed*idx%2147483647;
        nseed = nseed*(abs(start+idx)%2069)%2147483647;
        long long temp=(48271 * nseed + 0) % 2147483647;
        return temp;
    }
__global__ void rand_Poisson(float *randomseed, int *target,  int* matrix_scale, float *lambda)
{
    int k = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        float Lambda = lambda[int(idx / matrix_scale[1])];
        float p = 1.0;
        float l = exp(-Lambda);
        int seed = randomseed[idx]*2147483647;
        seed = cudarand(seed);
        while (p >= l)
        {
            seed = cudarand(seed);
            float u = seed / 2147483647.0;
            p *= u;
            k++;
        }
        target[idx] = k-1;
    }
}
""")

def poisson(Lambda, times=1,device='cpu'):

    Lambda, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.int32, Lambda)
    func = Sampler.get_function('rand_Poisson')
    func(randomseed, output, matrix_scale, Lambda, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output
