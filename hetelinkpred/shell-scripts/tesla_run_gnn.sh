#!/bin/bash

model_name=SAGE
echo "Dataset: "
read dataset_name
echo "Exclude target or not: "
read exclude
echo "Full neighbor or not: "
read full_neighbor
echo "Number of layers: "
read num_layers
echo "Training batch size: "
read batch_size
echo "CUDA number: "
read cuda
cd ../

for hidden_dim in 256
do
  for lr in 0.0001
    do
      identifier=${model_name}_${dataset_name}_hdim${hidden_dim}_nlayers${num_layers}_lr${lr}_batch_size${batch_size}_CUDA${cuda}_exclude_target${exclude}_full_neighbor${full_neighbor}
      echo ${identifier}
      if [ ${full_neighbor} -eq 0 ]
      then
        if [ ${exclude} -eq 0 ]
        then
          CUDA_VISIBLE_DEVICES=${cuda}, python3 graphsage_wo_exclude_train_target.py --dataset ${dataset_name} --lr ${lr} --hidden_dim ${hidden_dim} --num_layers ${num_layers} --batch_size ${batch_size} --runs 3 --n_epochs 150 --log_steps 5 1> logs/${identifier}.txt
        else
          CUDA_VISIBLE_DEVICES=${cuda}, python3 graphsage_wo_exclude_train_target.py --dataset ${dataset_name} --lr ${lr} --hidden_dim ${hidden_dim} --num_layers ${num_layers} --batch_size ${batch_size} --exclude_target 1 --runs 3 --n_epochs 150 --log_steps 5 1> logs/${identifier}.txt
        fi
      else
        if [ ${exclude} -eq 0 ]
        then
          CUDA_VISIBLE_DEVICES=${cuda}, python3 graphsage_wo_exclude_train_target.py --dataset ${dataset_name} --lr ${lr} --hidden_dim ${hidden_dim} --num_layers ${num_layers} --batch_size ${batch_size} --full_neighbor 1 --runs 3 --n_epochs 150 --log_steps 5 1> logs/${identifier}.txt
        else
          CUDA_VISIBLE_DEVICES=${cuda}, python3 graphsage_wo_exclude_train_target.py --dataset ${dataset_name} --lr ${lr} --hidden_dim ${hidden_dim} --num_layers ${num_layers} --batch_size ${batch_size} --full_neighbor 1 --exclude_target 1 --runs 3 --n_epochs 150 --log_steps 5 1> logs/${identifier}.txt
        fi
      fi
    done
done