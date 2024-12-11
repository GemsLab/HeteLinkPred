#!/usr/bin/env bash
set -x

DATASET=attributed-ppi
{
    for model in GCN SAGE GCN_DOT SAGE_DOT GCN_DistMultS SAGE_DistMultS GPRGNN \
                 GPRGNN_DOT GPRGNN_DistMultS MLPDecoder; do
        python graphsage_wo_exclude_train_target.py \
        --model_name $model --dataset="$DATASET" \
        --num_layers 2 \
        --batch_size 65536 --n_epochs 10000 --runs 3 --log_steps 50 \
        --checkpoint_folder output/$DATASET/ $@
    done
}

DATASET=attributed-facebook
{
    for model in GCN SAGE GCN_DOT SAGE_DOT GCN_DistMultS SAGE_DistMultS GPRGNN \
                 GPRGNN_DOT GPRGNN_DistMultS MLPDecoder; do
        python graphsage_wo_exclude_train_target.py \
        --model_name $model --dataset="$DATASET" \
        --num_layers 2 \
        --batch_size 65536 --n_epochs 10000 --runs 3 --log_steps 50 \
        --checkpoint_folder output/$DATASET/ $@
    done
}

