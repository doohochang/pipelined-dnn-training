#ifndef _hparams_h_
#define _hparams_h_

enum Activation {
    ACTIVATION_LINEAR,
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU
};

typedef struct _HiddenLayer {
    int number_of_nodes;
    Activation activation;
} HiddenLayer;

enum Loss {
    LOSS_SOFTMAX_CROSS_ENTROPY
};

typedef struct _OutputLayer {
    int number_of_nodes;
    Loss loss;
} OutputLayer;

typedef struct _ModelSpec {
    unsigned int number_of_input_nodes;
    unsigned int number_of_hidden_layers;
    HiddenLayer *hidden_layers;
    OutputLayer output_layer;
} ModelSpec;

typedef struct _SubModelSpec {
    unsigned int number_of_layers;
    unsigned int number_of_input_nodes;
    HiddenLayer *layers;
} SubModelSpec;

typedef struct _HyperParams {
    int number_of_devices;
    ModelSpec model_spec;
    int epoch;
    int merge_period_epoch;
    float learning_rate;
} HyperParams;

#endif

