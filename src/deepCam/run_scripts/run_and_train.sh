#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   sbatch run_and_time.sh --system SYSTEM_NAME --nnodes NUMBER OF NODES
while test $# -gt 0; do
  case "$1" in
    -s)
      shift
      if test $# -gt 0; then
        export TRAINING_SYSTEM=$1
      else
        echo "no process specified"
        exit 1
      fi
      shift
      ;;
    --system*)
      export TRAINING_SYSTEM=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
  esac
done

if [ "$TRAINING_SYSTEM" != "booster" ] && [ "$TRAINING_SYSTEM" != "horeka" ]
  then
    echo "must specify system that we are running on! give as first unnamed parameter"
    exit 128
fi
# ============= above is new stuff

# export configs:
## System config params
export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export NCCL_SOCKET_IFNAME="ib0"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# JB
if [ "$TRAINING_SYSTEM" == "booster" ]
  then
    # JB
    export DGXNGPU=2
    export DGXNSOCKET=2
    export DGXSOCKETCORES=24 # 76 CPUs / DGXNSOCKET
    export DGXHT=2  # HT is on is 2, HT off is 1
    export TRAIN_DATA_PREFIX="/p/largedata/datasets/MLPerf/MLPerfHPC/deepcam_v1.0/"
    export OUTPUT_DIR="/p/project/jb_benchmark/MLPerf-1.0/run-logs/"
#    DISTRIBUTED="srun --mpi=pspmix --cpu-bind=none "

elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    # horeka
    export DGXNGPU=2
    export DGXNSOCKET=2
    export DGXSOCKETCORES=36 # 76 CPUs / DGXNSOCKET
    export DGXHT=2  # HT is on is 2, HT off is 1
    # TODO: define the output directory for horeka
#    DISTRIBUTED="mpirun --bind-to none --np ${NGPU}"

else
  # this is never hit, is this needed?
  echo "must specify system that we are running on! give as first unnamed parameter"
fi

# ===================== finished config exports =====================

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
readonly global_rank=${SLURM_PROCID:-}
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

echo "running benchmark"
export NGPUS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

if [[ ${PROFILE} -ge 1 ]]; then
    export TMPDIR="/profile_result/"
fi
RUN_TAG="$SLURM_JOBID"
# todo define: run_tag, output_dir, data_dir_prefix

#GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')

# these parameters are taken from the run scripts from other model runs
# commented out parameters are the defaults, can be anabled here
PARAMS=(
       --wireup_method                       "nccl-openmpi"
       --run_tag                             "${run_tag}"
       --output_dir                          "${output_dir}"
#       --checkpoint                          "None"
       --data_dir_prefix                     "${TRAIN_DATA_PREFIX}"
#       --max_inter_threads                   "1"
       --max_epochs                          "200"
       --save_frequency                      "400"
       --validation_frequency                "200"
       --max_validation_steps                "50"
       --logging_frequency                   "0"
       --training_visualization_frequency    "200"
       --validation_visualization_frequency  "40"
       --local_batch_size                    "2"
#       --channels                            "{[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}"
       --optimizer                           "LAMB"
       --start_lr                            "1e-3"
#       --adam_eps                            "1e-8"
       --weight_decay                        "1e-2"
#       --loss_weight_pow                     "-0.125"
       --lr_warmup_steps                     "0"
       --lr_warmup_factor                    "1.0"
       --lr_schedule                         "{\"type\":\"multistep\",\"milestones\":\"15000 25000\",\"decay_rate\":\"0.1\"}"
#       --target_iou                          "0.82"
       --model_prefix                        "classifier"
       --amp_opt_level                       "O1"
#       --enable_wandb
#       --resume_logging
#       |& tee
#       -a ${output_dir}/train.out
)

# TODO: fix the profiler
PROFILE_COMMAND=""
if [[ ${PROFILE} == 1 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 ]] || [[ ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
            PROFILE_COMMAND="nvprof --profile-child-processes --profile-api-trace all --demangling on --profile-from-start on  --force-overwrite --print-gpu-trace --csv --log-file /results/rn50_v1.5_${BATCHSIZE}.%h.%p.data --export-profile /results/rn50_v1.5_${BATCHSIZE}.%h.%p.profile "
        fi
    fi
fi

if [[ ${PROFILE} == 2 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 ]] || [[ ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
        PROFILE_COMMAND="nsys profile --trace=cuda,nvtx --force-overwrite true --export=sqlite --output /results/${NETWORK}_b${BATCHSIZE}_%h_${local_rank}_${global_rank}.qdrep "
        fi
    fi
fi

DISTRIBUTED="srun --mpi=pspmix --cpu-bind=none "
# $SINGULARLAUNCHER should be defined in the start_training_run.sh script

if [[ ${PROFILE} -ge 1 ]]; then
    TMPDIR=/results ${DISTRIBUTED} ${PROFILE_COMMAND} python3 train_imagenet.py "${PARAMS[@]}"; ret_code=$?
else
    ${DISTRIBUTED} "$SINGULARLAUNCHER" python ../train_hdf5_ddp.py "${PARAMS[@]}"; ret_code=$?
fi

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
export PROFILE=0
