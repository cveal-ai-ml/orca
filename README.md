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

### Kubernetes

To stream the output from kubenetes commands, examples are below:

- `watch "kubectl get pods"`
- `watch "kubectl get jobs"`

Create storage for code, data, and results

- `cd orca/examples/kubernetes/storage`
- `kubectl apply -f code.yaml`
- `kubectl apply -f data.yaml`
- `kubectl apply -f results.yaml`

Confirm storage creation 

- `kubectl get pvc`

Create a pod to interact to with storage

- `cd orca/examples/kubernetes/examples`
- `kubectl apply -f monitor.yaml`

While pod is building, you can monitor its progress

- `kubectl describe pod orca-monitor`

Enter the pod and download code to code storage

- The pod entrypoint is `/develop/code`
- `kubectl exec -it orca-monitor -- /bin/bash`
- `git clone https://github.com/cveal-ai-ml/orca.git`
- `exit`

Launch job to train ORCA

- `kubectl apply -f train.yaml`
- Monitor its job creation via `kubectl describe job orca-train`
