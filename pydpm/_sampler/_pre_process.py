import numpy as np
import time
import ctypes


def para_preprocess(times=1, in_type=np.float32, out_type=np.float32, *args):
    """
    preprocess the input parameters in sampling the distribution with gpu
    Inputs:
        times   : [int] repeat times
        in_type : [np.dtype] or list of np.dtype the dtype of the input parameters
        out_type: [np.dtype] the dtype of the output sampling results
        args[0] : [np.ndarray] the first variable in the target distribution
        args[1] : [np.ndarray] the second variable in the target distribution
    Outputs:
        para_a      : [pycuda.gpuarray] the input matrix for the first parameter
        para_b      : [pycuda.gpuarray] the input matrix for the second parameter
        output      : [pycuda.gpuarray] the matrix on gpu to store the sampling result
        para_scale  : [list] a list including the number of element and repeat times in the resulting matrix
        para_seed   : [pycuda.gpuarray] seed matrix on gpu
        partition   : [list] a list including
        scalar_flag : [bool] if the resulting matrix is a scalar
    """
    assert len(args) <= 2, 'Value Error: the number of the input parameter in the sampling distribution should not be larger than 2'

    if len(args) == 1:
        para_a = np.array(args[0], dtype=in_type, order='C')
        # assert len(para_a.shape) <= 2, 'Shape Error: the dimension of the input parameter a in the sampling distribution shoud not be larger than 2'

        # obtain the output_scale and judge if the para_a is a scalar
        if times > 1:
            output_scale = para_a.shape + (times,)
            scalar_flag = False
        else:
            output_scale = (1,) if para_a.shape == () else para_a.shape
            scalar_flag = True if para_a.shape == () else False

    elif len(args) == 2:
        if type(in_type) == type:
            in_type = [in_type, in_type]
        assert (type(in_type) == list and len(in_type) == 2)

        para_a = np.array(args[0], dtype=in_type[0], order='C')
        para_b = np.array(args[1], dtype=in_type[1], order='C')
        # assert len(para_a.shape) <= 2, 'Shape Error: the dimension of the input parameter a in the sampling distirbution shoud not be larger than 2'
        # assert len(para_b.shape) <= 2, 'Shape Error: the dimension of the input parameter b in the sampling distirbution shoud not be larger than 2'

        # make sure the sizes of para_a and para_b are equal
        if para_a.size == 1 and para_b.size != 1:
            para_a = np.array(np.full(para_b.shape, para_a), dtype=in_type[0], order='C')
        if para_b.size == 1 and para_a.size != 1:
            para_b = np.array(np.full(para_a.shape, para_b), dtype=in_type[1], order='C')

        # obtain the output_scale and judge if the para_a is a scalar
        if times > 1:
            output_scale = para_a.shape + (times,)
            scalar_flag = False
        else:
            output_scale = (1,) if para_a.shape == () else para_a.shape
            scalar_flag = True if para_a.shape == () else False

    matrix_scale = para_a.size
    nElems = para_a.size * times  # output_scale multi.

    # output
    output = np.empty(output_scale, dtype=out_type, order='C')

    if len(args) == 1:
        return matrix_scale, nElems, para_a, output, output_scale, scalar_flag
    elif len(args) == 2:
        return matrix_scale, nElems, para_a, para_b, output, output_scale, scalar_flag