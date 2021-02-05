#ifndef __schedule_h__
#define __schedule_h__

#include "hparams.cuh"

SubModelSpec *generate_submodel_specs(int num_devices, ModelSpec model);

#endif
