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
echo "Hidden dim"
read hidden_dim
echo "Inference mode: "
read inference_mode


for lr in 0.0005
  do
    identifier=${model_name}_${dataset_name}_hdim${hidden_dim}_nlayers${num_layers}_lr${lr}_batch_size${batch_size}_exclude_target_degree${exclude_target_degree}_full_neighbor${full_neighbor}_inference_mode${inference_mode}
    echo ${identifier}
    if [ ${full_neighbor} -eq 0 ]
    then
      if [ "$inference_mode" = "train" ]
      then
        sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_gnn_model_greatlakes.sh "graphsage_wo_exclude_train_target.py" ${dataset_name} ${lr} ${hidden_dim} ${num_layers} ${batch_size} ${model_name} ${exclude_target_degree} ${inference_mode}
      else
        sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_gnn_model_greatlakes.sh "graphsage_inference_target.py" ${dataset_name} ${lr} ${hidden_dim} ${num_layers} ${batch_size} ${model_name} ${exclude_target_degree} ${inference_mode}
      fi
    else
      if [ "$inference_mode" = "train" ]
      then
        sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_gnn_model_greatlakes.sh "graphsage_wo_exclude_train_target.py" ${dataset_name} ${lr} ${hidden_dim} ${num_layers} ${batch_size} ${model_name} ${exclude_target_degree} ${inference_mode} "--full_neighbor 1"
      else
        sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_gnn_model_greatlakes.sh "graphsage_inference_target.py" ${dataset_name} ${lr} ${hidden_dim} ${num_layers} ${batch_size} ${model_name} ${exclude_target_degree} ${inference_mode} "--full_neighbor 1"
      fi
    fi
  done