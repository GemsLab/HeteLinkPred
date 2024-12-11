#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --time=30:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=default                                           # set QOS, this will determine what resources can be requested
#SBATCH --mem=20g
#SBATCH --gres=gpu:1

source /fs/clip-emoji/tonyzhou/anaconda3/etc/profile.d/conda.sh
conda activate gnn_pitfall
cd ../

script_fn=${1}
dataset_name=${2}
lr=${3}
hidden_dim=${4}
num_layers=${5}
batch_size=${6}
model_name=${7}
exclude_degree=${8}
inference_mode=${9}
full_neighbor=${10}


if [ ${dataset_name} == 'USAir' ]
then
  epoch_number=100
  accum_iter_number=1
else
  epoch_number=150
  accum_iter_number=1
fi

# srun --pty --ntasks=1 --cpus-per-task=4 --mem-per-cpu=6000 --gres=gpu:1 --time=02-00:00:00 --partition=spgpu --account=dkoutra1 nvidia-smi
python ${script_fn} --runs 3 --n_epochs ${epoch_number} --log_steps 5 --dataset ${dataset_name} --lr ${lr} --batch_size ${batch_size} --num_layers ${num_layers} --hidden_dim ${hidden_dim} --model_name ${model_name} --accum_iter_number ${accum_iter_number} --exclude_target_degree ${exclude_degree} ${full_neighbor} --inference_mode ${inference_mode}
conda deactivate