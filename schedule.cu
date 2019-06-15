#include <cuda_runtime.h>

#include "schedule.cuh"
#include "model.cuh"

SubModelSpec *generate_submodel_specs(int num_devices, ModelSpec model_spec) {
    int base_num_layers = model_spec.number_of_hidden_layers / num_devices;
    int num_remainder = model_spec.number_of_hidden_layers - base_num_layers * num_devices;
    
    SubModelSpec *submodels = (SubModelSpec *)malloc((sizeof(SubModelSpec) * num_devices));

    // Set number of layers to each submodel specs
    for (int i = 0; i < num_devices; i++)
        submodels[i].number_of_layers = base_num_layers;

    for (int i = num_devices - 2; i > num_devices - 2 - num_remainder;  i--)
        submodels[i].number_of_layers += 1;

    // Assign layers to submodel specs
    int num_input_nodes = model_spec.number_of_input_nodes;
    HiddenLayer *layers_to_assign = model_spec.hidden_layers;
    for (int i = 0; i < num_devices; i++) {
        submodels[i].number_of_input_nodes = num_input_nodes;
        submodels[i].layers = layers_to_assign;
        layers_to_assign += submodels[i].number_of_layers;
        num_input_nodes = (layers_to_assign - 1)->number_of_nodes;
    }

    return submodels;
}

void schedule_training(
    HyperParams params,
    int data_length, int input_dim, float **inputs, void *answers
) {
    // Generate submodel specs
    SubModelSpec *submodel_specs = generate_submodel_specs(params.number_of_devices, params.model_spec);

    // Initialize submodels
    // Each device has [params.number_of_devices] submodels
    SubModel **submodels;
    submodels = (SubModel **)malloc(sizeof(SubModel *) * params.number_of_devices);
    for (int i = 0; i < params.number_of_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc(submodels + i, sizeof(SubModel) * params.number_of_devices);
        for (int j = 0; j < params.number_of_devices; j++) {
            submodels[i][j] = SubModel(submodel_specs[i]);
        }
    }
    
    // Initialize Streams for each device
    cudaStream_t *calc_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * params.number_of_devices);
    cudaStream_t *transfer_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * params.number_of_devices);
    for (int i = 0; i < params.number_of_devices; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&calc_streams[i]);
        cudaStreamCreate(&transfer_streams[i]);
    }

    // Get max layer size
    int max_layer_size = params.batch_size;
    if (max_layer_size < params.model_spec.number_of_input_nodes)
        max_layer_size = params.model_spec.number_of_input_nodes;
    for (int i = 0; i < params.model_spec.number_of_hidden_layers; i++) {
        if (max_layer_size < params.model_spec.hidden_layers[i].number_of_nodes)
            max_layer_size = params.model_spec.hidden_layers[i].number_of_nodes;
    }

    // Init another memory spaces for each device
    float **ones = (float **)malloc(sizeof(float *) * params.number_of_devices);
    float **zero = (float **)malloc(sizeof(float *) * params.number_of_devices);
    float **batch_size_buffers = (float **)malloc(sizeof(float *) * params.number_of_devices);
    float **in_spaces = (float **)malloc(sizeof(float *) * params.number_of_devices);
    float **out_spaces = (float **)malloc(sizeof(float *) * params.number_of_devices);
    for (int i = 0; i < params.number_of_devices; i++) {
        cudaSetDevice(i);

        cudaMalloc(ones + i, sizeof(float) * max_layer_size);
        for (int j = 0; j < max_layer_size; j++) ones[i][j] = 1;

        cudaMalloc(zero + i, sizeof(float));
        *zero[i] = 0;

        cudaMalloc(batch_size_buffers + i, sizeof(float) * max_layer_size);

        cudaMalloc(in_spaces + i, sizeof(float) * params.batch_size * max_layer_size);
        cudaMalloc(out_spaces + i, sizeof(float) * params.batch_size * max_layer_size);
    }
}
