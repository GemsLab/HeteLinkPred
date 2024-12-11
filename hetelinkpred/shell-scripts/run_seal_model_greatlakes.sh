#!/bin/bash
#SBATCH --mail-user=tonyzhou@umd.edu
#SBATCH --mail-type=ALL
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:1
#SBATCH --time=04-00:00:00
#SBATCH --account=dkoutra1
#SBATCH --partition=spgpu
#

source ~/.bashrc
conda activate gnn_pitfall
cd ../seal-dataset/


script_fn=${1}
dataset_name=${2}
exclude_degree=${3}
inference_mode=${4}
eval_steps=5
runs=3


if [ ${dataset_name} == 'ogbl-collab' ]
then
  train_percent=15
  hidden_channels=256
  batch_size=32
  python ${script_fn} --runs ${runs} --batch_size ${batch_size} --dataset ${dataset_name} --train_percent ${train_percent} --hidden_channels ${hidden_channels} --eval_steps ${eval_steps} --exclude_target_degree ${exclude_degree} --use_valedges_as_input --inference_mode ${inference_mode}
elif [ ${dataset_name} == 'USAir' ]
then
  train_percent=100
  hidden_channels=32
  batch_size=1
  python ${script_fn} --runs ${runs} --batch_size ${batch_size} --dataset ${dataset_name} --train_percent ${train_percent} --hidden_channels ${hidden_channels} --eval_steps ${eval_steps} --exclude_target_degree ${exclude_degree} --inference_mode ${inference_mode}
elif [ ${dataset_name} == 'ogbl-citation2' ]
then
  train_percent=2
  python ${script_fn} --dataset ogbl-citation2 --num_hops 1 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 1 --test_percent 1 --runs 3 --exclude_target_degree ${exclude_degree} --inference_mode ${inference_mode}
fi

conda deactivate