"""
===========================================
Lognormal Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.lognormal(0.5, 0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.lognormal(0.5, 0.5, 1000000)
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
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
import pydpm.distribution.compat
from pycuda import gpuarray
cuda_generator = XORWOWRandomNumberGenerator()

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

__global__ void rand_Lognormal(float* randomseed, float * target, int * matrix_scale, float * mean, float * std)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        float U, V;
        float z;
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        int current_index = idx/matrix_scale[1];
        seed=cudarand(seed);
        U = seed / 2147483647.0;
        seed=cudarand(seed);
        V = seed / 2147483647.0;
        z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V);
        target[idx]=exp(z*std[current_index]+mean[current_index]); // rows first
    }
}

""")



Sampler = SourceModule("""
#include <stdio.h>

__global__ void calculate(float* target,int* matrix_scale , float* mean,float* std)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        int current_index=idx/matrix_scale[1];
        target[idx]=exp(target[idx]*std[current_index]+mean[current_index]);
    }
}
""")

def lognormal(mean=0, std=1, times=1,device='cpu'):

    mean = np.array(mean, dtype=np.float32, order='C')
    std = np.array(std, dtype=np.float32, order='C')
    single_number = False
    if mean.size == 1 and std.size != 1:
        mean = np.array(np.full(std.shape, mean), dtype=np.float32, order='C')
    if std.size == 1 and mean.size != 1:
        std = np.array(np.full(mean.shape, std), dtype=np.float32, order='C')
    mean = gpuarray.to_gpu(mean)
    std = gpuarray.to_gpu(std)
    if times > 1:
        output_scale = mean.shape + (times,)
    else:
        output_scale = mean.shape
        if output_scale == ():
            single_number = True
            output_scale = (1)
    output = cuda_generator.gen_normal(output_scale,np.float32)

    nx = output.size
    matrix_scale = gpuarray.to_gpu(np.array([nx, times], dtype=np.int32, order='C'))

    block = int(500)
    grid = int(nx/block)+1

    func = Sampler.get_function("calculate")
    func(output, matrix_scale, mean, std, grid=(grid, 1, 1), block=(block, 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output
