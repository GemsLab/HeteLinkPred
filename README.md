# On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks

Jiong Zhu, Gaotang Li, Yao-An Yang, Jing Zhu, Xuehao Cui, Danai Koutra. 2024. *On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks*. Advances in Neural Information Processing Systems 38 (2024).

[[Paper]](https://arxiv.org/abs/2409.17475)

## Run Experiments

To install the dependecies in a new conda environment, run
```
$ conda create --name <env> --file hetelinkpred/conda_env/dgl.txt
```
Scripts used to get results for BUDDY can be found in /subgraph-sketching/scripts. \
Other scripts can be found in /hetelinkpred/shell-scripts-hetelinkpred.

## GNN Encoders Supported
- GraphSAGE
- GCN
- BUDDY 

## Decoders Supported
- MLP
- DistMult
- Dot product

## Heuristics Supported 
- Common Neighbor
- Adamic Adar
- Resource Allocation
- Personalized Page Rank

## Datasets
- Synthetic dataset of varying heterophily level
- ogbl-collab
- ogbl-citation2
- E-Commerce
- Attributed-PPI
- Attributed-Facebook

Please following the script below to unzip the synthetic and e-commerce datasets.
```bash
# sudo apt-get install p7zip-full # install 7z for ubuntu.
# brew install p7zip # install 7z for mac
# For windows, download 7z from https://www.7-zip.org/download.html

cd hetelinkped
7z x dataset.7z.part.001
7z x dataset.7z
```
Other datasets can be downloaded from their repective official sites, and processed with /hetelinkpred/generate_split.py

## Contact

Please contact Jiong Zhu (jiongzhu@umich.edu) in case you have any questions.

## Citation

Please cite our paper if you make use of this code in your own work:

```bibtex
@article{zhu2024impactfeatureheterophilylink,
    title={On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks}, 
    author={Zhu, Jiong and  Li, Gaotang and Yang, Yao-An and Zhu, Jing and Cui, Xuehao and Koutra, Danai},
    journal={Advances in Neural Information Processing Systems},
    volume={38},
    year={2024}
}
```