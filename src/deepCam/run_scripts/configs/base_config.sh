#!/bin/bash

export WIREUP_METHOD="nccl-slurm"

# hyperparameters
export LOCAL_BATCH_SIZE=2
export START_LR=0.004
export OPTIMIZER="LAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="1100 4096"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=200
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=1
export SEED=999

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# auxiliary parameters
export LOGGING_FREQUENCY=10

# misc args
export ADDITIONAL_ARGS="--enable_jit --disable_comm_overlap --enable_graph"

# system parameters
export DGXNGPU=4
export DGXNNODES=${SLURM_NNODES}
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:30:00
