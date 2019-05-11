#ifndef _forward_h_
#define _forward_h_

#include <cuda_runtime.h>

#include "model.cuh"

/*
    Run forward algorithm with sub portion of the given model which has layers in range [start_layer,  end_layer]

    input: input vector to put into start layer
*/
void run_forward(SubModel *submodel, float *input, cudaStream_t stream);

#endif

