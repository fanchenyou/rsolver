# Optimization on Stiefel Manifold via Cayley Transform



## Requirements

The script depends on opencv python bindings, easily installable via conda:

```
conda install -c conda-forge opencv 
```

After that and after installing pytorch do:

```
pip install -r requirements.txt
```

## Train
The commands below are examples.

CIFAR-10:
```
[Cayley-SGD] 
CUDA_VISIBLE_DEVICES='0' python main.py --save ./models/CayleySGD --model resnet --depth 16 --width 5 --optim_method Cayley_SGD --lr 0.8 --lrg 0.1 --lr_decay_ratio 0.2

[Cayley-SGD-Adaptive] 
CUDA_VISIBLE_DEVICES='1' python main.py --save ./models/CayleySGD_Adaptive --model resnet --depth 16 --width 5 --optim_method Cayley_SGD_ADP --lr 0.8 --lrg 0.1 --lr_decay_ratio 0.2 
```