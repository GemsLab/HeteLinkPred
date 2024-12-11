#!/usr/bin/env bash
set -x

python graphsage_wo_exclude_train_target.py \
    --model_name SAGE_DistMultS --dataset=grid-dense --grid_idx 0 \
    --num_layers 2 \
    --batch_size 65536 --n_epochs 304 "$@"
