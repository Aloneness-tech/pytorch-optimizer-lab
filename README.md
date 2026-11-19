# pytorch-optimizer-lab

Self-study log. Goal: measure how much of a neural-network update is really
there — the effective rank of the update matrix ΔW — and how it changes
across optimizers and training conditions.

Status: week 1. Two small experiments so far.

## What this repo contains

| file | what it does |
|---|---|
| `01_ols_1d.py` | least squares `y = a*x + b` solved by SGD and Adam; learning-rate sweep |
| `02.py` | correlated parameters: what happens when `a` and `b` can substitute for each other |

## How to run

```
python 01_ols_1d.py
python 02.py
```

Python 3.13, PyTorch 2.15 (CPU), NumPy. No GPU required.

## What I found so far

### 01 — independent parameters

1. SGD and Adam reach the same solution as the closed-form least-squares
   fit: `a = 1.977`, `b = -1.513`.
2. Step size = learning rate × gradient. Across seven learning rates, the
   first-step displacement divided by the learning rate stayed at `3.752`,
   the gradient at the starting point.
3. The number of steps is not monotone in the learning rate: U-shaped,
   fastest near `lr × steepness ≈ 1`, diverging once `lr × steepness > 2`.
4. At the same learning rate: SGD 26 steps, Adam 85 steps.

### 02 — correlated parameters

5. Correlation flattens one direction: at equal distance from the truth,
   the loss differs by 30-40x between the two directions
   (theory: `(1+rho)/(1-rho) = 39`).
6. `rho = 0.95` is only slow, not degenerate: two different starting points
   meet after 200 steps, so the minimum is unique.
7. `rho = 1` is degenerate: the direction is frozen, the `a-b` offset is
   preserved exactly, and no optimizer can recover it.
8. Both optimizers share one limitation: the loss reaches the noise floor
   while the parameters are still off along the flat direction. Low loss
   does not mean the parameters are determined.
9. The optimizers differ only in speed: at the same learning rate, Adam is
   3-10x closer to the optimum after 100 steps.

## Not here yet

- the update-matrix measurement (ΔW, singular values, effective rank)
- SGD with momentum and Adam implemented by hand

## Scope

This is a learning log, not a finished result. Files I cannot explain line
by line are deliberately not included.
