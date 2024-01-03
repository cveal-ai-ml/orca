FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop

RUN pip install dash dash-core-components plotly scikit-image lightning
RUN pip install --no-dependencies qudida albumentations
RUN pip install --no-dependencies efficientnet-pytorch==0.7.1 timm==0.9.2 pretrainedmodels==0.7.4 segmentation_models_pytorch

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code
