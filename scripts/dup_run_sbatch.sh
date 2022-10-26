#!/bin/bash

# This is a wrapper for `sbatch_run.sh` to run repeated experiments
# It will duplicate the same params file for several times and run them all

#######################################################################
# An example usage:
#     GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh \
#       rtx6000 test-sbatch test.py ddp params.py --fp16 --ddp --cudnn
#######################################################################

# read args from command line
GPUS=${GPUS:-1}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
MEM_PER_CPU=${MEM_PER_CPU:-5}
QOS=${QOS:-normal}
REPEAT=${REPEAT:-3}
TIME=${TIME:-0}

PY_ARGS=${@:6}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3
DDP=$4
PARAMS=$5

for repeat_idx in $(seq 1 $REPEAT)
do
    params="${PARAMS:0:(-3)}-dup${repeat_idx}.py"
    cp $PARAMS $params
    job_name="${JOB_NAME}-dup${repeat_idx}"
    cmd="./scripts/sbatch_run.sh $PARTITION $job_name $PY_FILE $DDP --params $params $PY_ARGS"
    echo $cmd
    eval $cmd
done
