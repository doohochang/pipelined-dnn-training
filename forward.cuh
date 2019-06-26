#ifndef _forward_h_
#define _forward_h_

#include <cuda_runtime.h>

#include "model.cuh"
#include "hparams.cuh"

/*
    Run forward algorithm with the given submodel

    input: input matrix to put into first layer (batch_size * first_layer_size)
    batch_size: the size of mini batch
    stream: stream to schedule forward algorithm
    one: float type 1 in device memory
*/
void run_forward(SubModel *submodel, float *input, unsigned int batch_size, cudaStream_t stream, float *one, OutputLayer outputlayer, float* loss, int* label, float *batch_size_buffer, float* host_loss);



/*
    Run output layer calculation

    layer: output layer specification
    input: input matrix to put into output layer (batch_size * layer_size)
    batch_size: the size of mini batch
    answers: answers of each sample in mini-batch
    loss: float pointer to store loss output
    grad_input: float array pointer to store gradient matrix (batch_size * layer_size)
    stream: stream to schedule calculation
    ones: batch size float array pointer which is set one
    batch_size_buffer: float array pointer which has batch_size
*/
//void run_output_layer(OutputLayer layer, float *input, unsigned int batch_size, int *answers, float *loss, float *grad_input, cudaStream_t stream, cublasHandle_t handle, float *ones, float *batch_size_buffer);

#endif

