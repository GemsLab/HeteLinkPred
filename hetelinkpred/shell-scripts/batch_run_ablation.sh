#!/usr/bin/env bash

echo "Dataset: "
read dataset_name
echo "Model: "
read model_name
echo "Training batch size: "
read batch_size
echo "Hidden dim"
read hidden_dim


identifier=${model_name}_${dataset_name}_hdim${hidden_dim}_batch_size${batch_size}_ablation
echo ${identifier}
sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_ablation_study_greatlakes.sh ${dataset_name} ${hidden_dim} ${batch_size} ${model_name}