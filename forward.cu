#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "model.cuh"
#include "activation.cuh"

/*
    Run forward algorithm with sub portion of the given model which has layers in range [start_layer,  end_layer]

    input: input vector to put into start layer
    output: forward output vectors of each layer
*/
void run_forward(
    Model model, 
    unsigned int start_layer, 
    unsigned int end_layer, 
    float *input, 
    float **output,
    cudaStream_t stream
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    float *current_input = input;
    for (int i = start_layer, o = 0; i <= end_layer; i++, o++){
        // TODO: Implement whole foward logics

        //cudaMalloc(&current_output, sizeof(float) * model.spec.hidden_layers[i].number_of_nodes);
        
    }
}

void run_forward_step(
    cublasHandle_t handle,
    cudaStream_t stream,
    Activation activation,
    float *input, unsigned int input_size,
    float *weight_matrix, float *bias,
    float *output, unsigned int output_size
) {
    // weight_matrix: input_size * output_size
    // bias: output_size

    float *one;
    cudaMalloc(&one, sizeof(float));
    *one = 1.0;

    cublasScopy(
        handle, output_size,
        bias, 1,
        output, 1
    );

    cublasSgemv(
        handle, CUBLAS_OP_T,
        input_size, output_size,
        one,
        weight_matrix, input_size,
        input, 1,
        one,
        output, 1
    );

    switch (activation) {
        case ACTIVATION_LINEAR:
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_kernel<<<256, (output_size + 255) / 256>>>(output, output_size);
            break;
        case ACTIVATION_RELU:
            relu_kernel<<<256, (output_size + 255) / 256>>>(output, output_size);
            break;
    }
}

