set -x

{
    for DATASET in attributed-ppi attributed-facebook amazon-computer; do

    python -m src.runners.run_heuristics_esci --dataset_name $DATASET

    done
}