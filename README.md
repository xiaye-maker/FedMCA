# Personalized Federated Learning with Multiple Classifier Aggregation

## Models
- for PathMNIST, CIFAR10 and CIFAR100 
    1. FedAvgCNN()
    2. ResNet18, MobileNet.

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
python -u main.py -nc 20 -nb 10 -data cifar10 -m cnn -algo FedMCA -gr 1000 -K 1 -mu 0.05 -lam 1 -did 4 -go cnn
```
