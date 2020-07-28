"""
===========================================
F Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.f(10, 10, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.f(10, 10, 1000000)
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

__device__ float single_normal(float rand1, float rand2)
{
    float U, V;
    float z;

    U = rand1 / 2147483647.0;
    V = rand2 / 2147483647.0;
    z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V);
    return z;
}

__global__ void rand_F(float * randomseed, float * target,int * matrix_scale , int * n1 , int * n2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int N1=n1[int(idx/matrix_scale[1])];
        int N2=n2[int(idx/matrix_scale[1])];
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        
        seed = cudarand(seed);
        float sum1=0;
        float sum2=0;
        for (int i=0;i<N1;i++)
        {
            float r1 = seed;
            float r2 = cudarand(seed);
            seed = cudarand(r2);
            float x=pow(double(single_normal(r1, r2)),2);
            sum1+=x;
        }
        
        seed=cudarand(seed);
        for (int i=0;i<N2;i++)
        {
            float r1 = seed;
            float r2 = cudarand(seed);
            seed = cudarand(r2);
            float x=pow(double(single_normal(r1, r2)),2);
            sum2+=x;
        }
        
        target[idx]=(sum1/N1)/(sum2/N2);
    }
}

__global__ void Noncentral_F(float* randomseed, float* target, int* matrix_scale, int* n1, int* n2, float* delta, float* std)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int current_index=idx/matrix_scale[1];
        int N1=n1[current_index];
        int N2=n2[current_index];
        float mean=sqrt(delta[current_index]/N1);
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        
        seed=cudarand(seed);
        float sum1=0;
        float sum2=0;
        for (int i=0;i<N1;i++)
        {
            float r1 = seed;
            float r2 = cudarand(seed);
            seed = cudarand(r2);
            float x=pow(double(single_normal(r1, r2)*std[current_index]+mean),2);
            sum1+=x;
        }
        
        seed=cudarand(seed);
        for (int i=0;i<N2;i++)
        {
            float r1 = seed;
            float r2 = cudarand(seed);
            seed = cudarand(r2);
            float x=pow(double(single_normal(r1, r2)),2);
            sum2+=x;
        }
        
        target[idx]=(sum1/N1)/(sum2/N2);
    }
}


""")


def f(n1, n2, times=1,device='cpu'):

    if np.sum(n1<1):
        raise Exception('parameter n1 should be lager than 1')

    if np.sum(n2<1):
        raise Exception('parameter n2 should be lager than 1')

    n1, n2, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.int32, np.float32, n1, n2)
    func = Sampler.get_function('rand_F')
    func(randomseed, output, matrix_scale, n1, n2, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output

def noncentral_f(n1, n2, delta, times=1, std=1,device='cpu'):

    if np.sum(n1<1):
        raise Exception('parameter n1 should be lager than 1')

    if np.sum(n2<1):
        raise Exception('parameter n2 should be lager than 1')

    delta, std, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, delta, std)
    n1 = np.array(n1, dtype=np.int32, order='C')
    n2 = np.array(n2, dtype=np.int32, order='C')
    func = Sampler.get_function('Noncentral_F')
    func(randomseed, output, matrix_scale, n1, n2, delta, std, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output


