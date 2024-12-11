#!/usr/bin/env bash
set -x

python graphsage_wo_exclude_train_target.py \
    --model_name SAGE --dataset=grid-dense --grid_idx 4 \
    --num_layers 2 \
    --batch_size 65536 --n_epochs 1500 --runs 3 "$@"
