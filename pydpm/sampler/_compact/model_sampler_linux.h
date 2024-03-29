#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>

// status
#define blockDimX 32
#define blockDimY 4  // blockDimX * Y should be multiples of 32, and no more than 1024
#define gridDimX 128
#define nStatus (blockDimX * blockDimY * gridDimX)

// const
#define one_third 0.333333333333333
#define Pi 3.141592654

// gamma
#define nThreads_x 32
#define nThreads_y 4
#define nThreads (nThreads_x * nThreads_y)

