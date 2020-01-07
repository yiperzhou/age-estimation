# **distribution_loss** branch for the age estimation  

use the [Kullback-Leibler divergence loss](https://pytorch.org/docs/stable/nn.html?highlight=kldivloss#torch.nn.KLDivLoss) as the loss function

## introduction

### mathmatical formuation

$$
D_{\mathrm{KL}}(P \| Q)=\sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right)
$$

### code modification

```python
train_valid/age_losses_methods.py
```

### steps  
1. construct the ground truth age distribution with using the strategy from the paper, **Deep Label Distribution Learning With Label Ambiguity[1]**

## experimental results  


## reference  
1. [Gao, Bin-Bin, et al. "Deep label distribution learning with label ambiguity." IEEE Transactions on Image Processing 26.6 (2017): 2825-2838.](https://ieeexplore.ieee.org/abstract/document/7890384/)  

