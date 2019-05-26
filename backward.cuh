#ifndef _backward_h_
#define _backward_h_

#include <cuda_runtime.h>

#include "model.cuh"

/*
    Run backward algorithm with the given submodel

    number_of_upper_nodes: number of nodes of upper submodel's first layer
    upper_values: previous forward values to upper submodel
    upper_grads: gradient from upper submodel
    batch_size: the size of mini batch
    learning_weight: -(learning_weight / batch_size)
    stream: stream to schedule forward algorithm
    one: float type 1 in device memory
    zero: float type 0 in device memory
*/

void run_backward(SubModel *submodel, int number_of_upper_nodes, float *upper_values, float *upper_grads, unsigned int batch_size, float *learning_weight, cudaStream_t stream, float *one, float* zero);


#endif