#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd  /p/project/jb_benchmark/mlperf/training_results_v0.7/NVIDIA/benchmarks/resnet/implementations/mxnet \
cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

#export DGXNNODES=$SLURM_NNODES
SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
)

#srun "${SRUN_PARAMS[@]}" bash -c "singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash jb_train.sh"
srun "${SRUN_PARAMS[@]}" --pty singularity exec --nv \
      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif \
      "bash /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/run_scripts/jb_train.sh"

#bash -c "singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash jb_train.sh"

# old version
#srun  \
#    --mpi=pspmix --cpu-bind=none \
#    bash -c "\
#      source /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/deepCam/run_scripts/config_DGXA100_common.sh; \
#      source config_DGXA100_multi_230x8x35.sh; \
#      export CUDA_VISIBLE_DEVICES="0,1,2,3";  \
#      singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash run_and_time.sh"
