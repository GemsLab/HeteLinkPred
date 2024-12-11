#!/usr/bin/env bash

echo "Dataset: "
read dataset_name
echo "Model: "
read model_name
echo "Exclude target degree: "
read exclude_target_degree
echo "Full neighbor or not: "
read full_neighbor
echo "Number of layers: "
read num_layers
echo "Training batch size: "
read batch_size
echo "Inference mode: "
read inference_mode
echo "CUDA number: "
read cuda
cd ../

for hidden_dim in 256
do
  for lr in 0.0005
    do
      identifier=${model_name}_${dataset_name}_hdim${hidden_dim}_nlayers${num_layers}_lr${lr}_batch_size${batch_size}_CUDA${cuda}_exclude_target_degree${exclude_target_degree}_full_neighbor${full_neighbor}_inference_mode${inference_mode}
      echo ${identifier}
      if [ ${full_neighbor} -eq 0 ]
      then
          CUDA_VISIBLE_DEVICES=${cuda} python3 graphsage_inference_target.py --dataset ${dataset_name} --model_name ${model_name} --lr ${lr} --hidden_dim ${hidden_dim} --num_layers ${num_layers} --exclude_target_degree ${exclude_target_degree} --batch_size ${batch_size} --inference_mode ${inference_mode} --full_neighbor 0 --runs 3 --n_epochs 150 --log_steps 5 1> logs/${identifier}.txt
      else
          CUDA_VISIBLE_DEVICES=${cuda} python3 graphsage_inference_target.py --dataset ${dataset_name} --model_name ${model_name} --lr ${lr} --hidden_dim ${hidden_dim} --num_layers ${num_layers} --exclude_target_degree ${exclude_target_degree} --batch_size ${batch_size} --inference_mode ${inference_mode} --runs 3 --full_neighbor 1 --n_epochs 150 --log_steps 5 1> logs/${identifier}.txt
      fi
    done
done
