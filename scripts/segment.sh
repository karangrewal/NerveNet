#!/bin/bash

NET_TYPE=$1
BATCH_SIZE=$2
NUM_EPOCHS=$3
LEARNING_RATE=$4

# call python script to initiate learning

python train.py $NET_TYPE $BATCH_SIZE $NUM_EPOCHS $LEARNING_RATE