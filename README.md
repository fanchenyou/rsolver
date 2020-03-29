# Optimization on Stiefel Manifold



## Requirements

Create python3 Pytorch environment

```
conda create -n py37 python=3.7
source activate py37
conda install -c conda-forge opencv 
pip install -r requirements.txt
```

## Train

MNIST:
```
[Riemann-SGD-Adaptive] 
CUDA_VISIBLE_DEVICES='0' python main.py --save ./models/CayleySGD_Adaptive --model resnet --depth 10 --width 1 --optim_method Cayley_SGD_ADP --lr 0.1 --lrg 5.0 --lr_decay_ratio 0.2 --dataset MNIST --epochs 20

[Riemann-SGD] 
CUDA_VISIBLE_DEVICES='1' python main.py --save ./models/CayleySGD --model resnet --depth 10 --width 1 --optim_method Cayley_SGD --lr 0.1 --lrg 0.01 --lr_decay_ratio 0.2 --dataset MNIST --epochs 20

```

CIFAR-10:
```

[Riemann-SGD-Adaptive] 
CUDA_VISIBLE_DEVICES='0' python main.py --save ./models/CayleySGD_Adaptive --model resnet --depth 16 --width 5 --optim_method Cayley_SGD_ADP --lr 0.8 --lrg 0.1 --lr_decay_ratio 0.2 

[Cayley-SGD] 
CUDA_VISIBLE_DEVICES='1' python main.py --save ./models/CayleySGD --model resnet --depth 16 --width 5 --optim_method Cayley_SGD --lr 0.8 --lrg 0.1 --lr_decay_ratio 0.2


```