# pytorch-optimizer-lab

Self-study log. Goal: measure how much of a neural-network update is really
there — the effective rank of the update matrix ΔW — and how it changes
across optimizers and training conditions.

Status: week 1. Three small experiments so far.

## What this repo contains

| file | what it does |
|---|---|
| `01_ols_1d.py` | least squares `y = a*x + b` solved by SGD and Adam; learning-rate sweep |
| `02.py` | correlated parameters: what happens when `a` and `b` can substitute for each other |
| `03.py` | singular values and effective rank: from the data matrix to the update matrix |

## How to run

```
python 01_ols_1d.py
python 02.py
python 03.py
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

### 03 — singular values and effective rank

10. Rank = the number of non-zero singular values. In float32 the "zero"
    is about 1e-8, not exactly 0, so counting non-zero values depends on
    the threshold — which is why the effective rank uses a smooth formula.
11. The effective-rank formula degenerates to the count in the expected
    cases: eight equal singular values give `8.000`; one dominant value
    gives `1.061`; and for `[2, 1]` it gives `1.800`, which matches the
    hand calculation `(2+1)^2/(4+1) = 9/5`.
12. `sigma^2` is the curvature along that direction. Moving `t` along the
    i-th right singular vector increases the loss by `t^2 * sigma_i^2 / n`;
    measured and predicted agree to 1e-7 (float32 precision). This is an
    identity, not a statistical estimate.
13. The effective rank of the design matrix falls from `2.00` at `rho = 0`
    to `1.31` at `rho = 0.95` and `1.00` at `rho = 1` — the quantitative
    version of "the two parameters cannot be separated". It is also about
    ten times more stable than the ratio of the largest to the smallest
    singular value (sd `0.020` versus `5.7`).
14. The rank of `ΔW` is capped by the batch size: with batches of 8, 32 and
    200 on a 64x64 layer, the number of non-zero singular values is exactly
    8, 32 and 64 (effective rank 7.4, 23.6 and 39.6). So the batch size,
    not the layer, sets the ceiling.

## Not here yet

- momentum and Adam implemented by hand
- an experiment whose outcome I do not already know; everything above
  reproduces results that are standard in the literature

## Scope

This is a learning log, not a finished result. Files I cannot explain line
by line are deliberately not included.
