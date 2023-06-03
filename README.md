# HIT


This project is the official implementation of our paper 
[Holistic Interaction Transformer Network for Action Detection](https://arxiv.org/abs/2210.12686) (**WACV 2023**), authored
by Gueter Josmy Faure, Min-Hung Chen and Shang-Hong Lai. 

### What makes this different from the [original Repo](https://github.com/joslefaure/HIT)?
- The code is simplified and customized for the AVA dataset
- This implementation outperforms the original (on JHMDB and UCF) with only person features (no hands, pose and objects)
- (Caution) I made sure the code works for AVA, without training and testing on the whole AVA dataset, therefore I don't know how good (bad) it is compared to the original implementation.

## Installation


You need first to install this project, please check [INSTALL.md](INSTALL.md)

## Data Preparation

To do training or inference on AVA, please check [DATA.md](DATA.md)
for data preparation instructions. Instructions for other datasets coming soon.

## Model Zoo

Please see [MODEL_ZOO.md](MODEL_ZOO.md) for downloading models.

## Training and Inference

To do training or inference with HIT, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).


## Citation

If this project helps you in your research or project, please cite
this paper:

```
@InProceedings{Faure_2023_WACV,
    author    = {Faure, Gueter Josmy and Chen, Min-Hung and Lai, Shang-Hong},
    title     = {Holistic Interaction Transformer Network for Action Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3340-3350}
}
```

## Acknowledgement
We are very grateful to the authors of [AlphAction](https://github.com/MVIG-SJTU/AlphAction) for open-sourcing their code from which this repository is heavily sourced. If your find this research useful, please consider citing their paper as well.

```
@inproceedings{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2020}
}
```
