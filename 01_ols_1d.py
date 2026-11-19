"""01 — 最小二乘 + SGD / Adam + 学习率

【这个项目最终要做什么】
    我想弄清楚：神经网络训练时，每一步到底"改了多少"。
    把一步的改动写成一个矩阵 ΔW，量它有几个自由度（erank），
    再看换不同优化器、加不同限制时，这个数怎么变，和收敛快慢有没有关系。
    这是为了申请 UROP（LLM 训练的高效优化）做准备。

    但在那之前，我得先确认最底下三件事：loss 怎么算、梯度怎么来、参数怎么被改。
    这一份就是干这个的。

【这一份解决什么问题】
    用一个最小二乘问题 y = a*x + b，把"训练"拆到没有黑箱：

        1. 最小二乘：要最小化的那个 loss 是什么
        2. 梯度下降：SGD 靠什么方向、多大步子往前走
        3. Adam：同一个问题也能解；并且和 SGD 在同一个学习率下比一比
        4. 学习率：它怎么影响每步走多远、要走多少步

    没有神经网络，没有矩阵，只有两个数 a 和 b。全部能手算。
    收敛的判据统一为：a 和 b 都进到解析解 ±0.01 以内。

【这一份的结论】
    1. SGD 和 Adam 都解到同一个答案，和 np.polyfit 的解析解一致：
       a = 1.977, b = -1.513。

    2. 步长 = 学习率 × 梯度。看输出里「step1/lr」这一列：七个学习率下
       全都是 3.752，而 3.752 正好是起点的梯度大小。

    3. 学习率不是越大越好，收敛步数是个 U 形：
       0.005 → 500 步都到不了
       0.48  → 3 步（最快）
       0.9   → 26 步（来回跳，反而变慢）
       1.05  → 发散
       分界线在「学习率 × 最陡方向的陡度 = 2」附近。

    4. 同一个学习率 lr=0.1 下：SGD 26 步，Adam 85 步。
       SGD 的步长跟着梯度走，梯度大就走得快；Adam 不跟梯度大小走，
       所以在这种简单问题上反而慢。

【我还没搞明白的】
    - 为什么分界线正好是 2。我会用，但推不出来。
    - "陡度"是我自己起的名，正式说法大概是二阶导或曲率，还没查证。

Run:  python 01_ols_1d.py
"""
import numpy as np
import torch

torch.manual_seed(0)

# ---------------------------------------------------------------- 造数据
n = 200
x = torch.randn(n)
y = 2.0 * x - 1.5 + 0.3 * torch.randn(n)   # 真实的 a=2, b=-1.5

# 解析解：能达到的最好结果，后面所有"收敛"都以它为准
slope, intercept = np.polyfit(x.numpy(), y.numpy(), 1)
A_HAT, B_HAT = float(slope), float(intercept)

# 两个方向的陡度（loss 对这个参数的二阶导）
STEEP_A = 2 * float((x ** 2).mean())    # = 2*mean(x^2)
STEEP_B = 2.0                           # = 2*mean(1)，截距方向固定是 2
STEEP_MAX = max(STEEP_A, STEEP_B)       # 稳不稳由最陡的那个决定


def run(opt_name, lr, steps=500):
    """跑一遍。返回 (第1步位移, 收敛步数, 是否发散, 最终a, 最终b)。"""
    a_ = torch.zeros(1, requires_grad=True)
    b_ = torch.zeros(1, requires_grad=True)
    if opt_name == "SGD":
        opt_ = torch.optim.SGD([a_, b_], lr=lr)
    else:
        opt_ = torch.optim.Adam([a_, b_], lr=lr)

    first, hit, diverged = None, None, False
    for t in range(1, steps + 1):
        loss = (((a_ * x + b_) - y) ** 2).mean()
        opt_.zero_grad()
        loss.backward()
        opt_.step()

        if t == 1:
            first = a_.item()                    # 从 0 出发，走到的值就是位移
        if not torch.isfinite(loss).item() or abs(a_.item()) > 1e3:
            diverged = True
            break
        if hit is None and abs(a_.item() - A_HAT) < 1e-2 and abs(b_.item() - B_HAT) < 1e-2:
            hit = t
    return first, hit, diverged, a_.item(), b_.item()


# ---------------------------------------------------------------- 第 1 部分：Adam 跑一遍
print("=" * 74)
print("第 1 部分：Adam lr=0.1，200 步，一步一步在做什么")
print("=" * 74)

a = torch.zeros(1, requires_grad=True)     # 两个标量，初始猜 0
b = torch.zeros(1, requires_grad=True)     # requires_grad=True 表示"请帮我算梯度"

opt = torch.optim.Adam([a, b], lr=0.1)
for step in range(201):
    loss = (((a * x + b) - y) ** 2).mean()
    opt.zero_grad()      # 清空上一步的梯度
    loss.backward()      # 算 a.grad、b.grad
    opt.step()           # 按优化器规则改 a、b

    if step % 50 == 0:
        print(f"step {step:3d}   loss {loss.item():7.4f}   a {a.item():6.3f}   b {b.item():6.3f}")

print()
print(f"梯度下降的答案:   a = {a.item():.3f},  b = {b.item():.3f}")
print(f"最小二乘的解析解: a = {A_HAT:.3f},  b = {B_HAT:.3f}")

# ---------------------------------------------------------------- 第 2 部分：学习率的影响
print()
print("=" * 74)
print("第 2 部分：学习率的影响（纯 SGD）")
print("=" * 74)
print(f"这条数据上：a 方向陡度 {STEEP_A:.3f}，b 方向陡度 {STEEP_B:.3f}")
print(f"判断稳不稳要看更陡的 {STEEP_MAX:.3f}\n")

header = (f"{'lr':>7} {'lr*steep':>9} {'step1':>9} {'step1/lr':>9} "
          f"{'steps':>7} {'a_end':>9} {'b_end':>9}  verdict")
print(header)
print("-" * len(header))

for lr in [0.005, 0.05, 0.2, 0.48, 0.7, 0.9, 1.05]:
    first, hit, diverged, a_end, b_end = run("SGD", lr)
    ratio = lr * STEEP_MAX

    if diverged:
        verdict, n_str = "发散", "div"
    elif hit is None:
        verdict, n_str = "500 步内没到", ">500"
    elif ratio < 1:
        verdict, n_str = "单调收敛", str(hit)
    else:
        verdict, n_str = "震荡但收敛", str(hit)

    a_str = f"{a_end:.3f}" if abs(a_end) < 1e3 else f"{a_end:.2e}"
    b_str = f"{b_end:.3f}" if abs(b_end) < 1e3 else f"{b_end:.2e}"
    print(f"{lr:>7.3f} {ratio:>9.2f} {first:>9.5f} {first / lr:>9.3f} "
          f"{n_str:>7} {a_str:>9} {b_str:>9}  {verdict}")

print()

# ---------------------------------------------------------------- 第 3 部分：同 lr 下 SGD vs Adam
print()
print("=" * 74)
print("第 3 部分：同一个学习率 lr=0.1，SGD 和 Adam 各要多少步")
print("=" * 74)

header3 = f"{'opt':>6} {'steps':>7} {'a_end':>9} {'b_end':>9}"
print(header3)
print("-" * len(header3))
for name in ["SGD", "Adam"]:
    _, hit, diverged, a_end, b_end = run(name, 0.1, steps=400)
    n_str = "div" if diverged else ("n/a" if hit is None else str(hit))
    print(f"{name:>6} {n_str:>7} {a_end:>9.3f} {b_end:>9.3f}")

print()
