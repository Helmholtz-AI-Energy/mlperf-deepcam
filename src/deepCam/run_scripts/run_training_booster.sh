#!/bin/bash
#SBATCH -A jb_benchmark
#SBATCH -J mlperf_train
#SBATCH --ntasks-per-node 4
#SBATCH --nodes=512
## #SBATCH -p booster
#SBATCH --partition largebooster
#SBATCH --time=00:10:00
#SBATCH --gres gpu:4
#SBATCH --output /p/project/jb_benchmark/MLPerf-1.0/slurm-logs-%N-%J.out
#SBATCH --error /p/project/jb_benchmark/MLPerf-1.0/slurm-logs-%N-%J.err
#SBATCH --reservation bench-2021-04-09


# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# load stuff
conda activate mlperf_deepcam
module load pytorch/v1.4.0
export PROJ_LIB=/global/homes/t/tkurth/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/t/tkurth/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:${PYTHONPATH}

#ranks per node
rankspernode=1
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

#parameters
run_tag="deepcam_train-booster"
data_dir_prefix="/p/largedata/datasets/MLPerf/MLPerfHPC/deepcam_v1.0"
output_dir="/p/project/jb_benchmark/MLPerf-1.0/run-logs/${run_tag}"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

#run training
srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 256 / ${rankspernode} )) --cpu_bind=none \
     python ../train_hdf5_ddp.py \
     --wireup_method "mpi" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 0 \
     --model_prefix "classifier" \
     --optimizer "Lamb" \
     --start_lr 1e-3 \
     --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor $(( ${SLURM_NNODES} / 8 )) \
     --weight_decay 1e-2 \
     --validation_frequency 200 \
     --training_visualization_frequency 200 \
     --validation_visualization_frequency 40 \
     --max_validation_steps 50 \
     --logging_frequency 0 \
     --save_frequency 400 \
     --max_epochs 200 \
     --amp_opt_level O1 \
     --local_batch_size 2 |& tee -a ${output_dir}/train.out




~
