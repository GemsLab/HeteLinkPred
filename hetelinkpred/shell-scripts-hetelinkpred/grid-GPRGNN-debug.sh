#!/usr/bin/env bash
set -x

# python graphsage_wo_exclude_train_target.py \
#     --model_name GPRGNN --dataset=grid-dense --grid_idx 9 \
#     --num_layers 2 \
#     --batch_size 65536 --n_epochs 10 "$@"


python graphsage_wo_exclude_train_target.py \
    --model_name GPRGNN --dataset=esci --grid_idx 9 \
    --num_layers 2 \
    --batch_size 65536 --n_epochs 10 "$@"