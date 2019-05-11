#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "forward.cuh"
#include "model.cuh"
#include "activation.cuh"

#define THREAD_NUM 256

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
            sigmoid_kernel<<<THREAD_NUM, (output_size + THREAD_NUM - 1) / THREAD_NUM>>>(output, output_size);
            break;
        case ACTIVATION_RELU:
            relu_kernel<<<THREAD_NUM, (output_size + THREAD_NUM - 1) / THREAD_NUM>>>(output, output_size);
            break;
    }
}

void run_forward(SubModel *submodel, float *input, cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    run_forward_step(
        handle, stream, submodel->spec.layers[0].activation,
        input, submodel->spec.number_of_input_nodes,
        submodel->weight_matrices[0], submodel->biases[0],
        submodel->forward_values[0], submodel->spec.layers[0].number_of_nodes
    );

    for (int i = 1; i < submodel->spec.number_of_layers; i++){
        run_forward_step(
            handle, stream, submodel->spec.layers[i].activation,
            submodel->forward_values[i - 1], submodel->spec.layers[i - 1].number_of_nodes,
            submodel->weight_matrices[i], submodel->biases[i],
            submodel->forward_values[i], submodel->spec.layers[i].number_of_nodes
        );
    }
}

