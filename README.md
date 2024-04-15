<h1 align="center"><b>LookSAM Optimizer</b></h1>
<h3 align="center"><b>Towards Efficient and Scalable Sharpness-Aware Minimization</b></h3>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 

LookSAM is an accelerated [SAM](https://arxiv.org/pdf/2010.01412.pdf) algorithm. Instead of computing the inner gradient
ascent every step, LookSAM computer it periodically and reuses the direction that promotes to flat regions.

This is unofficial repository for [Towards Efficient and Scalable Sharpness-Aware Minimization](https://arxiv.org/pdf/2203.02714.pdf). 
Currently it is only proposed an algorithm without layer-wise adaptive rates (but it will be soon...).

In rewritten `step` method you are able to fed several arguments:
1. `t` is a train_index to define index of current batch;
2. `samples` are input data;
3. `targets` are input ground-truth data;
4. `zero_sam_grad` is a boolean value to zero gradients under SAM condition (first step) (see discussion [here](https://github.com/rollovd/LookSAM/issues/3) ;
5. `zero_grad` is a boolean value for zero gradient after second step;

[Unofficial SAM repo](https://github.com/davda54/sam/blob/main/README.md) is my inspiration :)

## Usage

```python
from looksam import LookSAM


model = YourModel()
criterion = YourCriterion()
base_optimizer = YourBaseOptimizer
loader = YourLoader()

optimizer = LookSAM(
    k=10,
    alpha=0.7,
    model=model,
    base_optimizer=base_optimizer,
    rho=0.1,
    **kwargs
)

...

model.train()

for train_index, (samples, targets) in enumerate(loader):
    ...

    loss = criterion(model(samples), targets)
    loss.backward()
    optimizer.step(
        t=train_index, 
        samples=samples, 
        targets=targets, 
        zero_sam_grad=True, 
        zero_grad=True
    )
    ...

```
