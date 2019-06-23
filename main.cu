#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "hparams.cuh"
#include "model.cuh"
#include "forward.cuh"
#include "backward.cuh"

#define TRAIN_CASE 55000
#define TEST_CASE 10000
#define NUM_DEVICE 1
#define THREAD_LEN 256
#define N_HIDDEN 4
#define D_INPUT 784
#define D_HIDDEN_1 1000
#define D_HIDDEN_2 1000
#define D_HIDDEN_3 1000
#define D_HIDDEN_4 10
#define D_OUTPUT 10
#define LEARNIG_RATE 0.01
#define BATCH_SIZE 100
#define MERGE_EPOCH 1
#define EPOCH 1000

int main(int argc, char** argv) {
    
    HiddenLayer* hiddenlayers = (HiddenLayer*)malloc(sizeof(HiddenLayer)* N_HIDDEN);

    hiddenlayers[0].number_of_nodes = D_HIDDEN_1;
    hiddenlayers[0].activation = ACTIVATION_RELU;

    hiddenlayers[1].number_of_nodes = D_HIDDEN_2;
    hiddenlayers[1].activation = ACTIVATION_RELU;
    
    hiddenlayers[1].number_of_nodes = D_HIDDEN_3;
    hiddenlayers[1].activation = ACTIVATION_RELU;
    
    hiddenlayers[1].number_of_nodes = D_HIDDEN_4;
    hiddenlayers[1].activation = ACTIVATION_LINEAR;

    OutputLayer outputlayer;
    outputlayer.number_of_input_nodes = D_OUTPUT;
    outputlayer.loss = LOSS_SOFTMAX_CROSS_ENTROPY;

    ModelSpec modelspec;
    modelspec.number_of_input_nodes = D_INPUT;
    modelspec.number_of_hidden_layers = N_HIDDEN;
    modelspec.hidden_layers = hiddenlayers;
    modelspec.output_layer = outputlayer;
    
    HyperParams hyperparmeters;
    hyperparmeters.number_of_devices = NUM_DEVICE;
    hyperparmeters.model_spec = modelspec;
    hyperparmeters.epoch = EPOCH;
    hyperparmeters.merge_period_epoch = MERGE_EPOCH;
    hyperparmeters.batch_size = BATCH_SIZE;
    hyperparmeters.learning_rate = LEARNIG_RATE;
    
    SubModelSpec submodelspec;
    submodelspec.number_of_layers = N_HIDDEN;
    submodelspec.number_of_input_nodes = D_INPUT;
    submodelspec.layers = hiddenlayers;
    
    SubModel submodel(submodelspec);
/*    
    cudaSetDevice(0);
    srand(time(NULL));
    
    float *host_train_input, *host_test_input *train_input, *test_input;
    int *host_train_label, *host_test_label *train_label, *test_label;
    
    host_train_input = (float*)malloc(sizeof(float) * D_INPUT * TRAIN_CASE);
    host_test_input = (float*)malloc(sizeof(float) * D_INPUT * TEST_CASE);
    cudaMalloc(&train_input, sizeof(float) * D_INPUT * TRAIN_CASE);
	cudaMalloc(&test_input, sizeof(float) * D_INPUT * TEST_CASE);
    
    host_train_label = (int*)malloc(sizeof(int) * TRAIN_CASE);
    host_test_label = (int*)malloc(sizeof(int) * TEST_CASE);
    cudaMalloc(&train_label, sizeof(int) * TRAIN_CASE);
	cudaMalloc(&test_label, sizeof(int) * TEST_CASE);
    
    FILE *train_image_path, *test_image_path;
	FILE *train_label_path, *test_label_path;

	train_image_path = fopen("./data/train_image.txt", "r");
	train_label_path = fopen("./data/train_label.txt", "r");
	test_image_path = fopen("./data/test_image.txt", "r");
	test_label_path = fopen("./data/test_label.txt", "r");
    
    int buffer_size = 0;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		for(int m = 0; m < D_INPUT; m++)
		{
			fscanf(train_image_path, "%f", &host_train_input[buffer_size++]);
		}
	}
	
	//get train_label
	buffer_size = 0;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		fscanf(train_label_path, "%d", &host_train_label[buffer_size++]);
	}
    
    float fshuffle[D_INPUT];
	int ishuffle;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		int idx = rand() % (TRAIN_CASE - n) + n;
	
		ishuffle = tint[idx];
		tint[idx] = tint[n];
		tint[n] = ishuffle;
			
		memcpy(fshuffle, &tfloat[idx*D_INPUT], sizeof(float) * D_INPUT);
		memcpy(&tfloat[idx*D_INPUT], &tfloat[n*D_INPUT], sizeof(float) * D_INPUT);
		memcpy(&tfloat[n*D_INPUT], fshuffle, sizeof(float) * D_INPUT);
	}
    
    cudaMemcpy(train_input, tfloat, sizeof(float) * D_INPUT * TRAIN_CASE, cudaMemcpyHostToDevice);
	cudaMemcpy(train_label, tint, sizeof(int) * TRAIN_CASE, cudaMemcpyHostToDevice);
    
    //get test_input
	buffer_size = 0;
	for(int n = 0; n < TEST_CASE; n++)
	{
		for(int m = 0; m < D_INPUT; m++)
		{
			fscanf(test_image_path, "%f", &tfloat[buffer_size++]);
		}
	}

	cudaMemcpy(test_input, tfloat, sizeof(float) * D_INPUT * TEST_CASE, cudaMemcpyHostToDevice);

	//get test_label
	buffer_size = 0;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		fscanf(test_label_path, "%d", &tint[buffer_size++]);
	}

	fclose(train_image_path);
	fclose(train_label_path);
	fclose(test_image_path);
	fclose(test_label_path);
    
    float *input;
	int *label;
    
	cudaMalloc(&input, sizeof(float) * D_INPUT * BATCH_SIZE);
	cudaMalloc(&label, sizeof(int) * BATCH_SIZE);
    
    cudaStream_t stream;
	cudaStreamCreate(&stream);
    
    float *one;
    float *zero;
    float *batch_size_buffer;
    float lr = LEARNIG_RATE;
    float *learning_rate;
    
    cudaMalloc(&one, sizeof(float) * D_HIDDEN_1 * BATCH_SIZE);
    cudaMemset(one, 1,  sizeof(float) * D_HIDDEN_1 * BATCH_SIZE);
    
	cudaMalloc(&zero, sizeof(float) * D_HIDDEN_1 * BATCH_SIZE);
    cudaMemset(zero, 0,  sizeof(float) * D_HIDDEN_1 * BATCH_SIZE);
    
    cudaMalloc(&batch_size_buffer, sizeof(float) * BATCH_SIZE);
    cudaMalloc(&learning_rate, sizeof(float));
    
    float * loss;
    cudaMalloc(&loss, sizeof(float));
    
    cudaMemcpyAsync(learning_rate, &lr, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    //start = clock();
	for(int epoch = 0; epoch < EPOCH; epoch++)
	{
		for(int n = 0; n < TRAIN_CASE/BATCH_SIZE; n++)
		{
			cudaMemcpyAsync(input, &train_input[n*BATCH_SIZE*D_INPUT], sizeof(float) * BATCH_SIZE * D_INPUT, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(label, &train_label[n*BATCH_SIZE], sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice, stream);

            run_forward(&submodel, input, BATCH_SIZE, stream, one);
            run_output_layer(outputlayer, submodel.forward_values[1], BATCH_SIZE, label, loss, submodel.gradients[1], stream, one, batch_size_buffer);
            run_backward(&submodel, D_OUTPUT, submodel.forward_values[1], submodel.gradients[1], BATCH_SIZE, learning_rate, stream, one, zero);
            
		}

		//test
    }
 */
    return 0;

}

