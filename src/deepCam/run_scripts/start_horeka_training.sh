#!/bin/bash
# This file is the first things to be done with srun

ml purge

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  #--cpu-bind="ldoms"
  --label
  #--container-name="nvidia-torch"
  #--container-mounts="/etc/slurm/:/etc/slurm:r/,/scratch:/scratch:rw,/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/:/work:rw,/hkfs/work/workspace/scratch/qv2382-mlperf_data/:/data:r"
  #--container-mounts="/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,/scratch:/scratch"
  #--container-mounts="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/:/work"
  #--container-mounts="/hkfs/home/dataset/datasets/deepcam/:/data:ro"
  #--container-writable
)


export SLURM_CPU_BIND_USER_SET="ldoms"


export TRAIN_DATA_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"

export OUTPUT_ROOT="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/src/deepCam/run_scripts/"
SINGULARITY_FILE="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/docker/nvidia-optimized-torch.sif"

#srun "${SRUN_PARAMS[@]}" bash -c "singularity run --nv \
#      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
#      bash jb_train.sh"
srun "${SRUN_PARAMS[@]}" singularity exec --nv --bind "${TRAIN_DATA_PREFIX}":/data,${SCRIPT_DIR},${OUTPUT_ROOT} ${SINGULARITY_FILE} \
      bash -c "\
      	source ${SCRIPT_DIR}configs/base_horeka.sh; \
	export SLURM_CPU_BIND_USER_SET=\"none\"; \
      	bash run_and_time.sh"


# /usr/lib64/slurm/spank_pyxis.so
#export TRAIN_DATA_PREFIX="/hkfs/home/datasets/deepcam/"
#export OUTPUT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"
#export OUTPUT_ROOT="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"

#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
#export NVIDIA_VISIBLE_DEVICES="0,1,2,3"
#echo cuda devices
#echo $CUDA_AVAILABLE_DEVICES
#export NVIDIA_DRIVER_CAPABILITIES="utility,compute"
#echo $NVIDIA_DRIVER_CAPABILITIES

#export SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/src/deepCam/run_scripts/"

#export TRAIN_DATA_PREFIX="/data/"
#export OUTPUT_DIR="/work/run-logs"
#export OUTPUT_ROOT="/work/run-logs/"

#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

#export SCRIPT_DIR="/work/src/deepCam/run_scripts/"


#srun "${SRUN_PARAMS[@]}" bash -c "\
#        sleep 0.25;\
#	pwd"
#	
#	#cd ${SCRIPT_DIR}; \
#      	#source configs/base_horeka.sh; \
#	#bash run_and_time.sh"

