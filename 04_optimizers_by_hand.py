"""第 2 步：手写 SGD / momentum / Adam 的更新公式，和 torch.optim 一位一位对齐。

设计（很重要，否则对不上会以为是公式错）：
    梯度只由 torch 提供一次，然后同一份梯度同时喂给
        左边：torch.optim 的 opt.step()
        右边：我们自己写的三行更新
    两边算出的 w 必须相同。这样比较的就纯粹是"更新规则"，不掺梯度误差。

同一个回归问题，同一个 lr，只换优化器。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(0)

N = 1000
R = 10
LR = 5.3e-3          # 取 1/(最陡方向的陡度)，三个优化器都用这个，方便对比
STEPS = 5
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MOM = 0.9

g = torch.Generator().manual_seed(0)
x1 = torch.randn(N, generator=g)
x2 = R * torch.randn(N, generator=g)
y = 2.0 * x1 + 0.02 * x2 + 0.1 * torch.randn(N, generator=g)


# ---------------------------------------------------------------- 手写更新
# 每个函数签名固定：(w, grad, state, t) -> (新 w, 新 state)
# state 是"优化器自己的记忆"，纯 SGD 没有记忆，其他两个有。

def step_sgd(w, grad, state, t):
    """纯 SGD：w <- w - lr * grad。没有记忆，一步就是一步。"""
    return w - LR * grad, state


def step_momentum(w, grad, state, t):
    """SGD + momentum：多了一个"速度"buf，是梯度的指数加权移动平均。

        buf <- MOM * buf + grad        (注意：PyTorch 这里没有 (1-MOM) 系数)
        w   <- w - LR * buf
    """
    buf = MOM * state["buf"] + grad
    return w - LR * buf, {"buf": buf}


def step_adam(w, grad, state, t):
    """Adam：两个移动平均 + 逐坐标归一化 + 偏差修正。

        m <- B1*m + (1-B1)*grad          梯度的一阶矩（方向）
        v <- B2*v + (1-B2)*grad^2        梯度的二阶矩（大小）
        m_hat = m / (1 - B1^t)           偏差修正
        v_hat = v / (1 - B2^t)           偏差修正
        w <- w - LR * m_hat / (sqrt(v_hat) + EPS)
    """
    m = BETA1 * state["m"] + (1 - BETA1) * grad
    v = BETA2 * state["v"] + (1 - BETA2) * grad ** 2
    m_hat = m / (1 - BETA1 ** t)
    v_hat = v / (1 - BETA2 ** t)
    w = w - LR * m_hat / (np.sqrt(v_hat) + EPS)
    return w, {"m": m, "v": v}


SPECS = {
    "SGD":       (step_sgd,      lambda ps: torch.optim.SGD(ps, lr=LR),
                  lambda: {}),
    "momentum":  (step_momentum, lambda ps: torch.optim.SGD(ps, lr=LR, momentum=MOM),
                  lambda: {"buf": np.zeros(2)}),
    "Adam":      (step_adam,     lambda ps: torch.optim.Adam(ps, lr=LR, betas=(BETA1, BETA2), eps=EPS),
                  lambda: {"m": np.zeros(2), "v": np.zeros(2)}),
}


# ---------------------------------------------------------------- 对拍
for name, (step_fn, make_opt, make_state) in SPECS.items():
    w_t = torch.zeros(2, requires_grad=True)        # torch 一边
    opt = make_opt([w_t])
    w_h = np.zeros(2)                               # 手写一边
    state = make_state()

    print(f"\n=== {name} ===")
    print(f"{'step':>4}  {'torch.w1':>10} {'torch.w2':>10}   {'hand.w1':>10} {'hand.w2':>10}   {'max|diff|':>10}")

    for t in range(1, STEPS + 1):
        loss = ((w_t[0] * x1 + w_t[1] * x2 - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        grad = w_t.grad.detach().numpy().copy()      # 同一份梯度喂给手写版
        w_h, state = step_fn(w_h, grad, state, t)

        diff = np.abs(w_t.detach().numpy() - w_h).max()
        tw = w_t.detach().numpy()
        print(f"{t:>4}  {tw[0]:>10.6f} {tw[1]:>10.6f}   {w_h[0]:>10.6f} {w_h[1]:>10.6f}   {diff:>10.2e}")

    print(f"  5 步后最大偏差 {diff:.2e}  -> {'对上了' if diff < 1e-6 else '没对上，公式有问题'}")

# ---------------------------------------------------------------- Adam 第 1 步的玄机
print("\n=== Adam 第 1 步到底做了什么 ===")
w0 = torch.zeros(2)
grad0 = np.array([-4.22, -3.01])                 # 随便一个梯度
m = (1 - BETA1) * grad0
v = (1 - BETA2) * grad0 ** 2
m_hat, v_hat = m / (1 - BETA1), v / (1 - BETA2)
delta = -LR * m_hat / (np.sqrt(v_hat) + EPS)
print(f"  梯度        {grad0}")
print(f"  第 1 步位移 {delta}")
print(f"  位移/LR     {delta / LR}")
print("  -> 位移/LR 约等于 ±1，与梯度大小无关。这就是 Adam 不怕尺度差的来源。")
