"""
===========================================
Chisquare Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.chisquare(5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.chisquare(5, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=100, density=True)
plt.title('DSG.chisquare')
plt.show()

plt.figure()
plt.hist(b, bins=100, density=True)
plt.title('numpy.random.chisquare')
plt.show()

start_time = time.time()
a = DSG.noncentral_chisquare(5, 5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.noncentral_chisquare(5, 5, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=100, density=True)
plt.title('DSG.noncentral_chisquare')
plt.show()

plt.figure()
plt.hist(b, bins=100, density=True)
plt.title('numpy.random.noncentral_chisquare')
plt.show()

"""

import numpy as np
from pycuda.compiler import SourceModule
from pydpm.distribution.pre_process import para_preprocess
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

__device__ float single_normal(float rand1, float rand2)
{
    float U, V;
    float z;

    U = rand1 / 2147483647.0;
    V = rand2 / 2147483647.0;
    z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V);
    return z;
}

__global__ void rand_Chisquare(float * randomseed, float * target,int * matrix_scale , int * n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int N=n[int(idx/matrix_scale[1])];
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        seed=cudarand(seed);
        float sum=0;
        for (int i=0;i<N;i++)
        {
            float r1 = seed;
            float r2 = cudarand(seed);
            seed = cudarand(r2);
            float x=pow(double(single_normal(r1, r2)),2);
            sum+=x;
        }
        target[idx]=sum;
    }
}

__global__ void rand_Noncentral_Chisquare(float * randomseed, float * target,int * matrix_scale , int * n , float * delta , float * var)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int current_index=idx/matrix_scale[1];
        int N=n[current_index];
        float mean=sqrt(delta[current_index]/N);
        float Var=var[current_index];
        
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        seed=cudarand(seed);
        float sum=0;
        for (int i=0;i<N;i++)
        {
            float r1 = seed;
            float r2 = cudarand(seed);
            seed = cudarand(r2);
            float x=pow(double(single_normal(r1, r2)*Var+mean),2);
            sum+=x;
        }
        target[idx]=sum;
    }
}

""")

def chisquare(n, times=1,device='cpu'):

    n, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.int32, np.float32, n)
    func = Sampler.get_function('rand_Chisquare')
    func(randomseed, output, matrix_scale, n, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))
    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output

def noncentral_chisquare(n , delta, times=1 , var=1,device='cpu'):

    # n,delta,var must be same shape

    n = np.array(n, dtype=np.int32, order='C')
    delta, var, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, delta , var)
    func = Sampler.get_function('rand_Noncentral_Chisquare')
    func(randomseed, output, matrix_scale, n,delta, var, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output
