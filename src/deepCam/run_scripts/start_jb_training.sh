#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd  /p/project/jb_benchmark/mlperf/training_results_v0.7/NVIDIA/benchmarks/resnet/implementations/mxnet \
cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

#export DGXNNODES=$SLURM_NNODES
SRUN_PARAMS=(
  --mpi            pspmix
#  --cpu-bind       none
  --label
)

SINGULATIRY_PARAMS=(
  /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/nvidia-deepcam.sif

  "bash"

)

export TRAIN_DATA_PREFIX="/p/largedata/datasets/MLPerf/MLPerfHPC/deepcam_v1.0/"
export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0/run-logs/"

RUN_COMMANDS=(
  "source configs/base_config.sh"
  "bash /workspace/run_and_time.sh"
)

#srun "${SRUN_PARAMS[@]}" bash -c "singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash jb_train.sh"
srun "${SRUN_PARAMS[@]}" singularity exec --nv \
      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/nvidia-deepcam.sif "${RUN_COMMANDS[@]}"

#/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/run_scripts/jb_train.sh

#bash -c "singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash jb_train.sh"
