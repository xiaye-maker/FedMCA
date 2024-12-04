# Personalized Federated Learning with Multiple Classifier Aggregation

[//]: # ([![DOI]&#40;https://zenodo.org/badge/292225878.svg&#41;]&#40;https://zenodo.org/badge/latestdoi/292225878&#41;)

## Models
- for CIFAR10 and CIFAR100 
    2. FedAvgCNN()
    3. ResNet18, AlexNet, MobileNet, GoogleNet, etc.

- for Flowers102 and Tiny-ImageNet
  
  1. DNN(3\*64\*64)
  2. FedAvgCNN()
  3. AlexNet, ResNet18, MobileNet, GoogleNet, etc.

## Environments

With installed [python3.9](wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) to home

```bash
tar -zxf Python-3.9.0.tgz
 
cd Python-3.9.0
 
./configure --prefix=/usr/local/python3
 
#please run ./configure --enable-optimizations
./configure --enable-optimizations

make && make install

python

exit()
```

[//]: # (Install [CUDA]&#40;https://developer.nvidia.com/cuda-11-6-0-download-archive&#41; first. )

With the installed [conda](https://mirror.tuna.tsinghua.edu.cn/anaconda/archive/)

```bash
bash Anaconda XXX Linux-x86_64.sh
```
we can run in a conda virtual environment called *fl_torch*. 

```bash
conda env create -f env_cuda_latest.yaml
```

## Train and evaluate the model:
```bash
cd ./system
python main.py -data CIFAR10 -dim 3*32*32 -indim 1600 -hete Hete1 -algo FedRG -gr 200 -id 0
```
