#include <cuda_runtime.h>
#include <curand.h>
#include "model.cuh"

SubModel::SubModel(SubModelSpec spec) {
    this->spec = spec;

    // Alloc weight matrices
    cudaMalloc(&(this->weight_matrices), sizeof(float *) * spec.number_of_layers);

    cudaMalloc(&(this->weight_matrices[0]), sizeof(float) * spec.number_of_input_nodes * spec.layers[0].number_of_nodes);

    for (int i = 1; i < spec.number_of_layers; i++) {
        cudaMalloc(&(this->weight_matrices[i]), sizeof(float) * spec.layers[i - 1].number_of_nodes * spec.layers[i].number_of_nodes);
    }

    // Alloc biases & forward values
    cudaMalloc(&(this->biases), sizeof(float *) * spec.number_of_layers);
    cudaMalloc(&(this->forward_values), sizeof(float *) * spec.number_of_layers);

    for (int i = 0; i < spec.number_of_layers; i++) {
        cudaMalloc(&(this->biases[i]), sizeof(float) * spec.layers[i].number_of_nodes);
        cudaMalloc(&(this->forward_values[i]), sizeof(float) * spec.layers[i].number_of_nodes);
    }
}

SubModel::~SubModel() {
    for (int i = 0; i < spec.number_of_layers; i++) {
        cudaFree(this->weight_matrices[i]);
        cudaFree(this->biases[i]);
        cudaFree(this->forward_values[i]);
    }

    cudaFree(this->weight_matrices);
    cudaFree(this->biases);
    cudaFree(this->forward_values);
}
