#!/usr/bin/env bash
set -x

DATASET=esci
{
    for model in GPRGNN GPRGNN_DOT GPRGNN_DistMultS; do
        python graphsage_wo_exclude_train_target.py \
        --model_name $model --dataset="$DATASET" \
        --num_layers 2 \
        --batch_size 32768 --n_epochs 1000 --runs 3 --log_steps 10 \
        --exclude_target_degree 10 \
        --checkpoint_folder output/$DATASET/ $@
    done
}