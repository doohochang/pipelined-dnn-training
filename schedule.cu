#include <cuda_runtime.h>

#include "schedule.cuh"
#include "model.cuh"

SubModelSpec *generate_submodel_specs(int num_devices, ModelSpec model) {
    int base_num_layers = model.number_of_hidden_layers / num_devices;
    int num_remainder = model.number_of_hidden_layers - base_num_layers * num_devices;
    
    SubModelSpec *submodels = (SubModelSpec *)malloc((sizeof(SubModelSpec) * num_devices));

    // Set number of layers to each submodel specs
    for (int i = 0; i < num_devices; i++)
        submodels[i].number_of_layers = base_num_layers;

    for (int i = num_devices - 2; i > num_devices - 2 - num_remainder;  i--)
        submodels[i].number_of_layers += 1;

    // Assign layers to submodel specs
    int num_input_nodes = model.number_of_input_nodes;
    HiddenLayer *layers_to_assign = model.hidden_layers;
    for (int i = 0; i < num_devices; i++) {
        submodels[i].number_of_input_nodes = num_input_nodes;
        submodels[i].layers = layers_to_assign;
        layers_to_assign += submodels[i].number_of_layers;
        num_input_nodes = (layers_to_assign - 1)->number_of_nodes;
    }

    return submodels;
}

void schedule(int num_devices, SubModelSpec *submodel_specs) {
    SubModel **submodels;

    // Initialize submodels
    submodels = (SubModel **)malloc(sizeof(SubModel *) * num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc(submodels + i, sizeof(SubModel) * num_devices);
        for (int j = 0; j < num_devices; j++) {
            submodels[i][j] = SubModel(submodel_specs[i]);
        }
    }


}
