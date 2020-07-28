"""
===========================================
Zipf Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.zipf(10, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.zipf(10, 1000000)
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
from pydpm.distribution.pre_process import cuda_generator
from pycuda import gpuarray
import pycuda.autoinit
import pydpm.distribution.compat

Sampler = SourceModule("""
#include <stdio.h>

__global__ void caculate_prob(float* target, int* matrix_scale, float* a , float* sum)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[3])
    {   
        int current_index=idx*matrix_scale[2];
        float sum=0.0;
        for (int i=0;i<matrix_scale[2];i++)
        {
            sum+=float(1/pow(double(i+1),double(a[idx])));
            target[current_index+i]=sum;
        }   
        for (int i=0;i<matrix_scale[2];i++)
        {
            target[current_index+i]/=sum;
        }   
    }
}

__global__ void rand_Zipf(float* randomseed, int* target, int* matrix_scale, float* prob)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[3]*matrix_scale[2])
    {
        int current_index=idx/matrix_scale[1];
        int prob_offset=current_index*matrix_scale[2];    
        for (int i=0;i<matrix_scale[2];i++)
        {
            if (randomseed[idx] < prob[i+prob_offset])
            {
                target[idx]=i+1;
                return;
            }
        }
    }
}
""")

def preprocess(times, in_type, out_type, a, range):

    a = gpuarray.to_gpu(np.array(a, dtype=in_type, order='C'))
    single_number = False

    if times > 1:
        output_scale = a.shape + (times,)
    else:
        # single_sample
        output_scale = a.shape
        # output 1 number
        if output_scale == ():
            single_number = True
            output_scale = (1)

    output = gpuarray.empty(output_scale, dtype=out_type)
    prob_nx = range*a.size
    nx = output.size  # total elements
    matrix_scale = gpuarray.to_gpu(np.array([nx, times,range,prob_nx], dtype=np.int32, order='C'))

    randomseed = cuda_generator.gen_uniform([nx],dtype=np.float32)

    block_x = int(500)
    grid_x = int(np.floor(nx / block_x) + 1)
    partition = (grid_x, block_x)

    return a, output, matrix_scale, randomseed, partition, single_number ,prob_nx

def zipf(a, times=1 , Range=100, device='cpu'):

    a, output, matrix_scale, randomseed, partition, single_number, prob_nx = preprocess(times, np.float32, np.int32, a ,Range)
    prob_matrix = gpuarray.empty(a.shape+(Range,), dtype=np.float32)
    func = Sampler.get_function('caculate_prob')
    block = int(500)
    grid = int(a.size/block)+1
    func(prob_matrix, matrix_scale, a, grid=(grid,1,1), block=(block,1,1))

    func = Sampler.get_function('rand_Zipf')
    func(randomseed, output, matrix_scale, prob_matrix, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output



