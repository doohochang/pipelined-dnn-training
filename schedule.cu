#include <cuda_runtime.h>
#include <queue>

#include "schedule.cuh"
#include "model.cuh"
#include "forward.cuh"
#include "backward.cuh"

using namespace std;

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

enum TaskType {
    TASK_TYPE_FORWARD,
    TASK_TYPE_BACKWARD
};

typedef struct _Task {
    int step;
    TaskType type;
} Task;

bool is_schedule_finished(Task *last_task, int num_devices, int num_steps) {
    // TODO: Implement
    return false;
}

// inputs, answers are assumed to be in host memory
void schedule_training(
    HyperParams params,
    int data_length, int input_dim, float *inputs, void *answers
) {
    int num_devices = params.number_of_devices;

    // Generate submodel specs
    SubModelSpec *submodel_specs = generate_submodel_specs(num_devices, params.model_spec);

    // Initialize submodels
    // Each device has [num_devices] submodels
    SubModel **submodels;
    submodels = (SubModel **)malloc(sizeof(SubModel *) * num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc(submodels + i, sizeof(SubModel) * num_devices);
        for (int j = 0; j < num_devices; j++) {
            submodels[i][j] = SubModel(submodel_specs[i]);
        }
    }
    
    // Initialize Streams for each device
    cudaStream_t *calc_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_devices);
    cudaStream_t *transfer_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_devices);
    for (int i = 0; i < num_devices; i++) {
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
    float **ones = (float **)malloc(sizeof(float *) * num_devices);
    float **zero = (float **)malloc(sizeof(float *) * num_devices);
    float **batch_size_buffers = (float **)malloc(sizeof(float *) * num_devices);
    float **f_in_spaces = (float **)malloc(sizeof(float *) * num_devices);
    float **b_in_spaces = (float **)malloc(sizeof(float *) * num_devices);
    float **out_spaces = (float **)malloc(sizeof(float *) * num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);

        cudaMalloc(ones + i, sizeof(float) * max_layer_size);
        for (int j = 0; j < max_layer_size; j++) ones[i][j] = 1;

        cudaMalloc(zero + i, sizeof(float));
        *zero[i] = 0;

        cudaMalloc(batch_size_buffers + i, sizeof(float) * max_layer_size);

        cudaMalloc(f_in_spaces + i, sizeof(float) * params.batch_size * max_layer_size);
        cudaMalloc(b_in_spaces + i, sizeof(float) * params.batch_size * max_layer_size);
        cudaMalloc(out_spaces + i, sizeof(float) * params.batch_size * max_layer_size);
    }

    // Define tasks for scheduling
    Task *last_task = (Task *)malloc(sizeof(Task) * num_devices);
    Task *next_task = (Task *)malloc(sizeof(Task) * num_devices);
    Task task_none = {-1, TASK_TYPE_FORWARD};

    // Define prepared step number to schedule
    int *prepared_f_step = (int *)malloc(sizeof(int) * num_devices);
    int *prepared_b_step = (int *)malloc(sizeof(int) * num_devices);

    // Store recorded events for each devices
    queue<cudaEvent_t> *forward_ready_events = new queue<cudaEvent_t> [num_devices];
    queue<cudaEvent_t> *backward_ready_events = new queue<cudaEvent_t> [num_devices];

    int num_steps = (data_length + params.batch_size - 1) / params.batch_size;
    for (int epoch = 0; epoch < params.epoch; epoch++) {
        // Initialize variables
        for (int i = 0; i < num_devices; i++) {
            last_task[i] = task_none;
            prepared_f_step[i] = (i == 0) ? 0 : -1;
            prepared_b_step[i] = -1;
        }

        // Iterate scheduling timestamp
        for (int t = 0; !is_schedule_finished(last_task, num_devices, num_steps); t++) {
            // Get next tasks to schedule
            for (int dev = 0; dev < num_devices; dev++) {
                if (last_task[dev].step < 0) {
                    next_task[dev].step = 0;
                    next_task[dev].type = TASK_TYPE_FORWARD;
                    continue;
                }

                next_task[dev] = task_none;
                int step;
                switch (last_task[dev].type) {
                    case TASK_TYPE_BACKWARD:
                        step = last_task[dev].step + num_devices - dev;
                        if (step < num_steps) {
                            // There are still remaining forward tasks
                            next_task[dev].step = step;
                            next_task[dev].type = TASK_TYPE_FORWARD;
                        } else if (last_task[dev].step + 1 < num_steps) {
                            // There's no remaining forward task
                            next_task[dev].step = last_task[dev].step + 1;
                            next_task[dev].type = TASK_TYPE_BACKWARD;
                        }
                        break;

                    case TASK_TYPE_FORWARD:
                        step = last_task[dev].step - (num_devices - dev - 1);
                        if (step >= 0) {
                            next_task[dev].step = step;
                            next_task[dev].type = TASK_TYPE_BACKWARD;
                        }
                        break;
                }
            }

            // Schedule next task if it's ready
            for (int dev = 0; dev < num_devices; dev++) {
                if (next_task[dev].step < 0) continue; // There's no futher task to schedule
                SubModel *model;
                switch (next_task[dev].type) {
                    case TASK_TYPE_FORWARD:
                        if (prepared_f_step[dev] < next_task[dev].step) break;
                        cudaSetDevice(dev);
                        model = submodels[dev] + (next_task[dev].step % num_devices);
                        if (dev == 0) {
                            run_forward(
                                model,
                                inputs + next_task[dev].step * params.batch_size * params.model_spec.number_of_input_nodes,
                                params.batch_size,
                                calc_streams[dev],
                                ones[dev]
                            );
                        } else {
                            
                        }
                        break;
                    case TASK_TYPE_BACKWARD:
                        if (prepared_b_step[dev] < next_task[dev].step) break;
                        cudaSetDevice(dev);
                        model = submodels[dev] + (next_task[dev].step % num_devices);
                        // TODO: Implement
                        break;
                }
            }
        }
    }
}
