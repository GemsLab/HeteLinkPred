#!/usr/bin/env bash
set -x

{
    for DATASET in amazon-computer; do
        for model in GPRGNN_DOT GPRGNN_DistMultS; do
            for alpha in 0.9; do
                python graphsage_wo_exclude_train_target.py \
                --model_name $model --dataset="$DATASET" \
                --num_layers 2 \
                --batch_size 65536 --n_epochs 1000 --runs 3 --log_steps 25 \
                --checkpoint_folder output/$DATASET/ --alpha $alpha $@
            done
        done
    done
}
