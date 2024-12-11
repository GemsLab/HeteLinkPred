#!/usr/bin/env bash
set -x
{
    for model in SAGE; do
        for i in 0 ; do
            for deg_threshold in 0; do
                python graphsage_wo_exclude_train_target.py \
                    --model_name $model --dataset=grid-dense --grid_idx $i \
                    --num_layers 2 \
                    --batch_size 65536 --n_epochs 600 --runs 3 \
                    --exclude_target_degree $deg_threshold \
                    # --alt_training_steps 20 --alt_training_ratio 0.5
            done
        done
    done
}