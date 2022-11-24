# Certified Graph Unlearning
Certified (approximate) machine unlearning for simplified graph convolutional networks (SGCs) with theoretical guarantees [(paper)](https://openreview.net/pdf?id=wCxlGc9ZCwi).

# Package Information
```
pytorch=1.10.0
torch-geometric=2.0.4
ogb
```

# Example
Our method supports three types of removing requests that could possibly arise in graph unlearning: node feature unlearning, node unlearning, and edge unlearning. We also include the comparison with complete retraining and one method designed for unstructured unlearning by [Guo et al](https://proceedings.mlr.press/v119/guo20c.html).

Here are three examples for `Cora` dataset. Detailed description of each input can be found in the code.

**Node feature unlearning**
```
python sgc_feature_node_unlearn.py --dataset='cora' --std=0.1 --num_removes=800 --train_mode='ovr' --prop_step=2 --lr=0.5 --removal_mode='feature' \
                                   --disp=100 --trails=1 --compare_retrain --compare_guo --optimizer='LBFGS' --data_dir='../PyG_datasets'
```

**Node unlearning**
```
python sgc_feature_node_unlearn.py --dataset='cora' --std=0.1 --num_removes=800 --train_mode='ovr' --prop_step=2 --lr=0.5 --removal_mode='node' \
                                   --disp=100 --trails=1 --compare_retrain --compare_guo --optimizer='LBFGS' --data_dir='../PyG_datasets'
```

**Edge unlearning**
```
python sgc_edge_unlearn.py --dataset='cora' --std=0.1 --num_removes=2000 --train_mode='ovr' --prop_step=2 --lr=0.5 --removal_mode='edge' \
                           --disp=100 --trails=1 --compare_retrain --optimizer='LBFGS' --data_dir='../PyG_datasets'
```

# Contact
Please contact Chao Pan (chaopan2@illinois.edu), Eli Chien (ichien3@illinois.edu) if you have any question.

# Citation
If you find our code or work useful, please consider citing our paper:
```
@inproceedings{
chien2022certified,
title={Certified Graph Unlearning},
author={Eli Chien and Chao Pan and Olgica Milenkovic},
booktitle={NeurIPS 2022 Workshop: New Frontiers in Graph Learning},
year={2022},
url={https://openreview.net/forum?id=wCxlGc9ZCwi}
}
```
