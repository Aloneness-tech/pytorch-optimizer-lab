# pytorch-optimizer-lab

Self-study log. Goal: measure how much of a neural-network update is really
there — the effective rank of the update matrix ΔW — and how it changes
across optimizers and training conditions.

Status: week 1. Only the basics are here so far.

## What this repo contains

| file | what it does |
|---|---|
| `01_ols_1d.py` | least squares `y = a*x + b` solved by SGD and Adam; learning-rate sweep |

## How to run

python 01_ols_1d.py

Python 3.13, PyTorch 2.15 (CPU), NumPy. No GPU required.

## What I found so far

1. SGD and Adam reach the same solution as the closed-form least-squares
   fit: `a = 1.977`, `b = -1.513`.
2. Step size = learning rate × gradient. Across seven learning rates, the
   first-step displacement divided by the learning rate stayed at `3.752`,
   which is the gradient at the starting point.
3. The number of steps is not monotone in the learning rate. It is
   U-shaped, fastest near `lr × steepness ≈ 1`, and diverges once
   `lr × steepness > 2`.
4. At the same learning rate: SGD 26 steps, Adam 85 steps.

## Not here yet

- the update-matrix measurement (ΔW, singular values, effective rank)
- SGD with momentum and Adam implemented by hand

## Scope

This is a learning log, not a finished result. Files I cannot explain line
by line are deliberately not included.