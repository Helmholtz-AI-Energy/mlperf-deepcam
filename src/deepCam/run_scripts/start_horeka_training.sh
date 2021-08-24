#!/bin/bash
# This file is the first things to be done with srun

ml purge

#export DGXNNODES=$SLURM_NNODES
SRUN_PARAMS=(
  --mpi               "pspmix"
  --cpu-bind          "none"
  --cibtainer-image   "nvidia-torch"
  --container-mount   "/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/:/work,/hkfs/home/dataset/datasets/deepcam/:/data"
)

export SLURM_CPU_BIND_USER_SET="ldoms"

export TRAIN_DATA_PREFIX="/hkfs/home/datasets/deepcam/"
export OUTPUT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"
export OUTPUT_ROOT="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"

#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

export SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/src/deepCam/run_scripts/"

# NOTE: horeka containers require a GPU (need to acquire a node with one)
#export ENROOT_DATA_PATH=/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/
#
#ENROOT_PARAMS=(
#  --env        "NVIDIA_DRIVER_CAPABILITIES"
#  --mount      "/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/:/work"
#  --mount      "/hkfs/home/dataset/datasets/deepcam/:/data"
#  --rw
#)

srun "${SRUN_PARAMS[@]}" bash -c "\
      	source ${SCRIPT_DIR}configs/base_config.sh; \
	      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      	bash run_and_time.sh"

#srun "${SRUN_PARAMS[@]}" enroot exec "${ENROOT_PARAMS[@]}" mlperf-torch \
#  bash /work/src/deepCam/run_scripts/horeka_train.sh

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
