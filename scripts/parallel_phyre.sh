#!/bin/bash

# PHYRE-related jobs using multiple threads

#######################################################################
# An example usage:
#     CPUS_PER_TASK=4 ./scripts/parallel_phyre.sh rtx6000,t4v2,t4v1,p100 \
#         test.py params.py weight.pth SPLIT_NUM
#######################################################################

# read args from command line
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
QOS=${QOS:-normal}

PY_ARGS=${@:6}
PARTITION=$1
PY_FILE=$2
PARAMS=$3
WEIGHT=$4
TOTAL_SPLIT=$5

for split in $(seq 0 $((TOTAL_SPLIT-1)))
do
    job_name="phyre_slots-split${split}"
    cmd="GPUS=1 CPUS_PER_TASK=$CPUS_PER_TASK MEM_PER_CPU=4 QOS=$QOS ./scripts/sbatch_run.sh $PARTITION $job_name $PY_FILE none --params $PARAMS --weight $WEIGHT --split $split --total_split $TOTAL_SPLIT --cpus $CPUS_PER_TASK $PY_ARGS"
    echo $cmd
    eval $cmd
done
