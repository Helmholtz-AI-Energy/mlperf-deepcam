#!/bin/bash
# This file is the first things to be done with srun

ml purge

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  #--cpu-bind="ldoms"
  --label
)

export SLURM_CPU_BIND_USER_SET="ldoms"
export DATA_DIR_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"
export OUTPUT_ROOT="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
export DEEPCAM_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/image-src/"

SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/src/deepCam/run_scripts/"
SINGULARITY_FILE="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/docker/nvidia-optimized-torch.sif"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}":/data,${SCRIPT_DIR},${OUTPUT_ROOT} ${SINGULARITY_FILE} \
    bash -c "\
      source ${SCRIPT_DIR}configs/base_horeka.sh; \
	    export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
