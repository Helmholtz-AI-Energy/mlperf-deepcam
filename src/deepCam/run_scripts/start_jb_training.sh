#!/bin/bash
#SBATCH -A jb_benchmark
#SBATCH -J mlperf_train
#SBATCH --ntasks-per-node 4
## #SBATCH -p booster
#SBATCH --partition largebooster
#SBATCH --time=00:10:00
#SBATCH --gres gpu:4
#SBATCH --output /p/project/jb_benchmark/MLPerf-1.0/slurm-logs-%N-%J.out
#SBATCH --error /p/project/jb_benchmark/MLPerf-1.0/slurm-logs-%N-%J.err
#SBATCH --reservation bench-2021-04-09

# this will call the srun
# script tree:
#   1. Machine selection
#   2. set configs  <--- we are here
#   3. call srun on the run script

TOTAL_TASKS =

ml purge
#ml GCC OpenMPI
#cd  /p/project/jb_benchmark/mlperf/training_results_v0.7/NVIDIA/benchmarks/resnet/implementations/mxnet \
cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/

export DGXNNODES=$SLURM_NNODES

# todo: set up the configs to be correct
# NOTE
srun  \
    --mpi=pspmix --cpu-bind=none \
    bash -c "\
      source /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/deepCam/run_scripts/config_DGXA100_common.sh; \
      singularity run --nv \
      /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif  \
      bash jb_train.sh"

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
