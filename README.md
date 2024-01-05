<h1 align="center"><b>LookSAM Optimizer</b></h1>
<h3 align="center"><b>Towards Efficient and Scalable Sharpness-Aware Minimization</b></h3>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 

LookSAM is an accelerated [SAM](https://arxiv.org/pdf/2010.01412.pdf) algorithm. Instead of computing the inner gradient
ascent every step, LookSAM computer it periodically and reuses the direction that promotes to flat regions.

This is unofficial repository for [Towards Efficient and Scalable Sharpness-Aware Minimization](https://arxiv.org/pdf/2203.02714.pdf). 
Currently it is only proposed an algorithm without layer-wise adaptive rates (but it will be soon...).

[Unofficial SAM repo](https://github.com/davda54/sam/blob/main/README.md) is my inspiration :)

## Usage
```python
from looksam import LookSAM

model = YourModel()
criterion = YourCriterion()
base_optimizer = YourBaseOptimizer

optimizer = LookSAM(
    k=10,
    alpha=0.7,
    model=model,
    base_optimizer=base_optimizer,
    rho=0.1,
    **kwargs
)

...

for train_index, (samples, targets) in enumerate(loader):
    ...

    loss = criterion(model(samples), targets)
    loss.backward()
    optimizer.step(t=train_index, samples=samples, targets=targets, zero_grad=True)

    ...

```
