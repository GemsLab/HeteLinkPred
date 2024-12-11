#!/usr/bin/env bash
set -x
{
    # for i in 0 9 1 8 2 7 3 6 4 5; do
    #     python graphsage_wo_exclude_train_target.py \
    #         --model_name SAGE --dataset=grid-dense --grid_idx $i \
    #         --num_layers 2 \
    #         --batch_size 65536 --n_epochs 300 --runs 3
    # done

    for model in SAGE; do
        for i in 0 9 1 8 2 7 3 6 4 5; do
            for deg_threshold in -0.25 -0.5; do
                python graphsage_wo_exclude_train_target.py \
                    --model_name $model --dataset=grid-dense --grid_idx $i \
                    --num_layers 2 \
                    --batch_size 65536 --n_epochs 300 --runs 3 \
                    --exclude_target_degree $deg_threshold
            done
        done
    done
}