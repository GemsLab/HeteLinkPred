#!/usr/bin/env bash
set -x
{
    for model in GCN; do
        python graphsage_wo_exclude_train_target.py \
            --model_name $model --dataset="esci" \
            --num_layers 2 \
            --batch_size 32768 --n_epochs 1000 --runs 3 --log_steps 50 \
            --exclude_target_degree 10 \
            --checkpoint_folder output/esci/
    done
}
