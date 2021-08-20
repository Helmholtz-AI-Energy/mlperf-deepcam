#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
  --label
)

export SLURM_CPU_BIND_USER_SET="none"


export TRAIN_DATA_PREFIX="/p/largedata/datasets/MLPerf/MLPerfHPC/deepcam_v1.0/"
export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0/run-logs/"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

SCRIPT_DIR="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts/"
SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/nvidia-deepcam.sif"

#srun "${SRUN_PARAMS[@]}" bash -c "singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash jb_train.sh"
srun "${SRUN_PARAMS[@]}" singularity exec --nv ${SINGULARITY_FILE} \
      bash -c "\
      	source ${SCRIPT_DIR}configs/base_config.sh; \
      	bash run_and_time.sh"
