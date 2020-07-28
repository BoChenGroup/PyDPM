"""
===========================================
Normal Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0
"""

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pydpm.distribution.pre_process import cuda_generator

Sampler = SourceModule("""
#include <stdio.h>

__global__ void calculate(float* target,int* matrix_scale , float* mean,float* std)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        int current_index=idx/matrix_scale[1];
        target[idx]=target[idx]*std[current_index]+mean[current_index];
    }
}
""")


def normal(mean=0, std=1, times=1, device='cpu'):

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

    output = cuda_generator.gen_normal(output_scale, np.float32)

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

def standard_normal(size=1,device='cpu'):

    output = cuda_generator.gen_normal(size, np.float32)
    if device == 'cpu':
        output = output.get()
    return output