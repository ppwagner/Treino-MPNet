#!/bin/bash

# export p2p
export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 train.py \
    --dataset 10B \
    --model_size l12 \
    --position_encoding alibi \
    --batch_size 32 \
    --sequence_length 512 \
    --tokens_per_step $((512*1024)) \
    --num_iterations 16000 \
    --learning_rate 1e-3 \
    --val_loss_every 64 \
    --tensorcores \
    --compile
