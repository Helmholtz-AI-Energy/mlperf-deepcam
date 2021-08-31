#!/bin/bash
# This file is the first things to be done with srun

ml purge

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  #--cpu-bind="ldoms"
#  --label
)

export SLURM_CPU_BIND_USER_SET="ldoms"
export DATA_DIR_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"
export OUTPUT_ROOT="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"

export WIREUP_METHOD="nccl-slurm-pmi"
export SEED="50"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
export DEEPCAM_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/image-src/"

SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/src/deepCam/run_scripts/"
SINGULARITY_FILE="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/docker/nvidia-optimized-torch.sif"
CONFIG_FILE="${SCRIPT_DIR}configs/best_configs/config_DGXA100_128GPU_BS128_graph.sh"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT},${DEEPCAM_DIR} ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
