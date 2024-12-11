#!/usr/bin/env bash
set -x
for i in {0..4}; do
    python graphsage_wo_exclude_train_target.py \
        --model_name SAGE --dataset=grid-dense --grid_idx $i \
        --num_layers 2 \
        --batch_size 65536 --n_epochs 150 --runs 3
done