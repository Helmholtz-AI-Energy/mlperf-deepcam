#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

SRUN_PARAMS=(
  --mpi            pspmix
#  --cpu-bind       none
  #--label
)

export SLURM_CPU_BIND_USER_SET="ldoms"

export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/deepCam/"

export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0/run-logs"

export WIREUP_METHOD="nccl-slurm"
export SEED="50"

export DEEPCAM_DIR="/opt/deepCam/"
#export DEEPCAM_DIR="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/image-src/"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

SCRIPT_DIR="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts/"
SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/nvidia-optimized-image-2.sif"
#nvidia-optimized-master.sif"
#nvidia-deepcam.sif"

if [ -n "${CONFIG_FILE}" ]
  then
    export CONFIG_FILE="${SCRIPT_DIR}configs/best_configs/config_DGXA100_512GPU_BS1024_graph.sh"
fi
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT} ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
