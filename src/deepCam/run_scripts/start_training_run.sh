#!/bin/bash

# hooray for stack overflow...
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for DeepCam on either HoreKa or Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-s, --system              the HPC machine to use [horeka, booster]"
      echo "-N, --nnodes              number of nodes to compute on"
      echo "-t, --time                compute time limit"
      exit 0
      ;;
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
    -N)
      shift
      if test $# -gt 0; then
        export SLURM_NNODES=$1
      else
        echo "no output dir specified"
        exit 1
      fi
      shift
      ;;
    --nnodes*)
      export SLURM_NNODES=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    *)
      break
      ;;
    -t)
      shift
      if test $# -gt 0; then
        export TIMELIMIT=$1
      else
        echo "no output dir specified"
        exit 1
      fi
      shift
      ;;
    --time*)
      export TIMELIMIT=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    *)
      break
      ;;
  esac
done

# todo: define the SBATCH params here

PARAMS_SBATCH=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "4"
  --time               "${TIMELIMIT}"
  --gres               "gpu:4"
  --job-name           "deepcam-mlperf"
  --time               "${TIMELIMIT}"
  --system             "${TRAINING_SYSTEM}"
)

if [ "$TRAINING_SYSTEM" == "booster" ]
  then
    # JB
    export TRAIN_DATA_PREFIX="/p/largedata/datasets/MLPerf/MLPerfHPC/deepcam_v1.0/"
    export OUTPUT_DIR="/p/project/jb_benchmark/MLPerf-1.0/run-logs/"

    SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/mlperf-torch.sif"
    export SINGULARLAUNCHER="singularity run --nv ${SINGULARITY_FILE}"

    PARAMS_SBATCH+=(
      --partition     "booster"
      --output        "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.err"
    )
elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    # this is the horeka case
    export TRAIN_DATA_PREFIX="/hkfs/home/datasets/deepcam/"
    export OUTPUT_DIR=""

    export SINGULARLAUNCHER="enroot"

    PARAMS_SBATCH+=(
      --partition     "booster"
      --output        "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.err"
    )
else
  echo "must specify system that we are running on! give as first unnamed parameter"
fi

#

sbatch $PARAMS_SBATCH  run_and_train.sh