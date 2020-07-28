"""
===========================================
Pre_process for input
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from pycuda import gpuarray
import pycuda.autoinit
import pydpm.distribution.compat
from pycuda.curandom import XORWOWRandomNumberGenerator
cuda_generator = XORWOWRandomNumberGenerator()

def para_preprocess(times, in_type, out_type, *args):

    if len(args) == 1:

        a = gpuarray.to_gpu(np.array(args[0], dtype=in_type, order='C'))
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

    if len(args) == 2:

        a = np.array(args[0], dtype=in_type, order='C')
        b = np.array(args[1], dtype=in_type, order='C')
        single_number = False

        if a.size == 1 and b.size != 1:
            a = np.array(np.full(b.shape, a), dtype=in_type, order='C')
        if b.size == 1 and a.size != 1:
            b = np.array(np.full(a.shape, b), dtype=in_type, order='C')

        a = gpuarray.to_gpu(a)
        b = gpuarray.to_gpu(b)

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

    nx = output.size  # total elements
    matrix_scale = gpuarray.to_gpu(np.array([nx, times], dtype=np.int32, order='C'))

    randomseed = cuda_generator.gen_uniform([nx], dtype=np.float32)

    block_x = int(500)
    grid_x = int(np.floor(nx / block_x) + 1)
    partition = (grid_x, block_x)


    if len(args) == 1:
        return a, output, matrix_scale, randomseed, partition, single_number
    if len(args) == 2:
        return a, b, output, matrix_scale, randomseed, partition, single_number


