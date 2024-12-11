#!/usr/bin/env bash
set -x
{
    for model in GCN; do
        for i in 0 9 1 8; do
            for deg_threshold in 0; do
                for num_layers in 1 2 3; do
                    for hidden_dim in 64 128 256; do
                        for lr in 5e-5 5e-4; do
                            python graphsage_wo_exclude_train_target.py \
                                --model_name $model --dataset=grid-dense --grid_idx $i \
                                --num_layers $num_layers --hidden_dim $hidden_dim \
                                --lr $lr \
                                --batch_size 65536 --n_epochs 300 --runs 3 \
                                --exclude_target_degree $deg_threshold \
                                --checkpoint_folder /root/hetelinkpred_new_xuehao/GCN-tunning/
                        done
                    done
                done
            done
        done
    done
}
