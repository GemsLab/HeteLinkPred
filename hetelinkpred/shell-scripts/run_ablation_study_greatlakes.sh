#!/bin/bash
#SBATCH --mail-user=tonyzhou@umd.edu
#SBATCH --mail-type=ALL
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:1
#SBATCH --time=04-00:00:00
#SBATCH --account=dkoutra1
#SBATCH --partition=spgpu
#

source ~/.bashrc
conda activate gnn_pitfall
cd ../

dataset_name=${1}
hidden_dim=${2}
batch_size=${3}
model_name=${4}


if [ ${dataset_name} == 'USAir' ]
then
  epoch_number=100
  accum_iter_number=1
else
  epoch_number=150
  accum_iter_number=1
fi

#for exclude_degree in 5 10 13 15 18 20 25
for exclude_degree in 18 23 25
do
  identifier=${dataset_name}_${model_name}_${exclude_degree}_experiment
  echo ${identifier}
  python graphsage_wo_exclude_train_target.py --runs 1 --n_epochs ${epoch_number} --log_steps 5 --dataset ${dataset_name} --lr 0.0005 --batch_size ${batch_size} --num_layers 3 --hidden_dim ${hidden_dim} --model_name ${model_name} --accum_iter_number 1 --exclude_target_degree ${exclude_degree} --full_neighbor 1 --inference_mode train
done
conda deactivate