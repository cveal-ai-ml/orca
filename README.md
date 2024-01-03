# Open set Recognition using Contrastive Anchors (ORCA)

This repo will demonstrate ORCA for OSR and knowledge discovery

## Installation / Setup

Dockerfile

- `docker build --platform linux/x86_64 --rm -t orca .`
- `docker run --platform linux/x86_64 -it -v data:/develop/data -v results:/develop/results -v code:/develop/code orca
- `--platform linux/x86_64` flag is only required for MacOS

Miniconda

- `conda create -n orca python=3.10`
- `conda activate orca`
- `pip install dash dash-core-components plotly scikit-image scikit-learn-extra lightning`
- `pip install --no-dependencies qudida albumentations`
- `pip install --no-dependencies efficientnet-pytorch==0.7.1 timm==0.9.2 pretrainedmodels==0.7.4 segmentation_models_pytorch`

## Datasets

[MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Run Example


