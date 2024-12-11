#!/usr/bin/env bash
set -x

DATASET=ogbl-collab
{
    for model in GPRGNN GPRGNN_DOT GPRGNN_DistMultS; do
        python graphsage_wo_exclude_train_target.py \
        --model_name $model --dataset="$DATASET" \
        --num_layers 2 \
        --batch_size 65536 --n_epochs 640 --runs 3 --log_steps 8 \
        --checkpoint_folder output/$DATASET/ $@
    done
}