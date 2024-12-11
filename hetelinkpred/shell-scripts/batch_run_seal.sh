#!/usr/bin/env bash
echo "Dataset: "
read dataset_name
echo "Exclude target degree: "
read exclude_target_degree
echo "Inference mode: "
read inference_mode

identifier=SEAL_${dataset_name}_exclude_target_degree${exclude_target_degree}_inference_mode${inference_mode}
echo ${identifier}
if [ "$inference_mode" = "train" ]
  then
    if [ "$dataset_name" = "ogbl-citation2" ]
    then
      sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_seal_model.sh "seal_main_citation2.py" ${dataset_name} ${exclude_target_degree} ${inference_mode}
    else
      sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_seal_model.sh "seal_main.py" ${dataset_name} ${exclude_target_degree} ${inference_mode}
    fi
  else
    if [ "$dataset_name" = "ogbl-citation2" ]
    then
      sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_seal_model.sh "seal_inference_citation2.py" ${dataset_name} ${exclude_target_degree} ${inference_mode}
    else
      sbatch --job-name ${identifier} --output ../logs/${identifier}.txt run_seal_model.sh "seal_inference_target.py" ${dataset_name} ${exclude_target_degree} ${inference_mode}
    fi
fi