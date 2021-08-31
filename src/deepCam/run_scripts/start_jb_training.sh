#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
  #--label
)

export SLURM_CPU_BIND_USER_SET="ldoms"

export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/deepCam/"

export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0/run-logs"

export WIREUP_METHOD="nccl-slurm"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

SCRIPT_DIR="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts/"
SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/nvidia-deepcam.sif"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}":/data ${SINGULARITY_FILE} \
    bash -c "\
      source ${SCRIPT_DIR}configs/base_config.sh; \
	    export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
