"""02 — 参数相关

【这个项目最终要做什么】
    量神经网络每一步的更新矩阵 ΔW 有几个自由度（erank），
    再看不同优化器、不同限制下这个数怎么变、和收敛快慢有没有关系。
    这一份是第二步：01 里两个参数互不干扰，这里让它们互相牵扯。

【这一份解决什么问题】
    01 里 x 只有一个，a 和 b 各走各的。这一份给两个相关的 x：

        x2 = rho * x1 + 噪声        rho 越大，两个变量越像

    然后看四件事：
        1. 参数偏离真值时，不同方向的 loss 涨得一样快吗？
        2. rho=0.95 时，从不同起点训练，多久才会走到同一个地方？
        3. rho=1（完全共线）时，还会走到同一个地方吗？
        4. 同样的起点和学习率，换成 Adam 会不会更快？

【这一份的结论】
    1. 相关会让一个方向变平。同样偏离真值 1.0，沿 (1,-1) 的 loss 约 0.1，
       沿 (1,+1) 约 4，差 30~40 倍（理论值 (1+rho)/(1-rho) = 39）。
    2. rho=0.95 只是慢，不是无解：200 步后两个起点重合，最小值唯一。
    3. rho=1 才是不可识别：5000 步也不动，a-b 保留起始值，换任何优化器都一样。
    4. 两个优化器都逃不掉同一个精度问题：loss 已经到噪声底线，
       但平方向上的参数误差几乎不进 loss，所以"loss 收敛"不等于"参数收敛"。
    5. 差别只在速度：同样 lr、同样 100 步，Adam 离最优点比 SGD 近 3~10 倍。
    
【我还没搞明白的】
    - 平方向每步的收缩比例能不能直接用 rho 算出来，我没推
    - "很慢"和"冻结"是同一个旋钮，但我还没把它们写成同一个式子

Run:  python 02.py
"""
import torch

torch.manual_seed(0)
n = 200
RHO = 0.95


def build(rho):
    x1 = torch.randn(n)
    x2 = rho * x1 + (1 - rho ** 2) ** 0.5 * torch.randn(n)
    return x1, x2


def make_y(x1, x2):
    """真值 a=2, b=-1，加一点噪声。"""
    return 2.0 * x1 - 1.0 * x2 + 0.1 * torch.randn(n)


def train(a0, b0, x1, x2, y, steps, lr=0.4):
    """从 (a0, b0) 出发训练，返回最终的 a、b、loss。"""
    a = torch.tensor([float(a0)], requires_grad=True)
    b = torch.tensor([float(b0)], requires_grad=True)
    opt = torch.optim.SGD([a, b], lr=lr)
    for _ in range(steps):
        loss = ((a * x1 + b * x2 - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return a.item(), b.item(), loss.item()


# ---------------------------------------------------------------- 第 1 部分：沿两个方向走
print("=" * 74)
print("第 1 部分：参数偏离真值，两个方向的 loss 涨得一样快吗")
print("=" * 74)

for rho in (0.0, RHO):
    x1, x2 = build(rho)
    y = make_y(x1, x2)
    print(f"\nrho = {rho}")
    for t in (0.0, 0.1, 0.5, 1.0):
        a, b = 2.0 + t, -1.0 - t                       # 沿 (1,-1)
        loss_minus = ((a * x1 + b * x2 - y) ** 2).mean().item()
        a, b = 2.0 + t, -1.0 + t                       # 沿 (1,+1)
        loss_plus = ((a * x1 + b * x2 - y) ** 2).mean().item()
        print(f"  走 {t:4.1f}   (1,-1)方向 {loss_minus:8.4f}   (1,+1)方向 {loss_plus:8.4f}")
    print("  （噪声底线约 0.01）")

# ---------------------------------------------------------------- 第 2 部分：rho=0.95，换起点换步数
print()
print("=" * 74)
print("第 2 部分：rho=0.95，两个起点，训练多久才会走到同一个地方")
print("=" * 74)

x1, x2 = build(RHO)
y = make_y(x1, x2)

header = f"{'start':>10} {'steps':>7} {'a':>8} {'b':>8} {'a+0.95b':>9} {'loss':>9}"
print(header)
print("-" * len(header))

for a0, b0 in [(0, 0), (5, -5)]:
    for s in (60, 200, 5000):
        a, b, l = train(a0, b0, x1, x2, y, s)
        print(f"{f'({a0}, {b0})':>10} {s:>7} {a:>8.3f} {b:>8.3f} {a + RHO * b:>9.3f} {l:>9.5f}")

print(f"\n  真值 a=2, b=-1，所以 a+0.95b = {2 + RHO * (-1):.3f}；噪声底线约 0.01")

# ---------------------------------------------------------------- 第 3 部分：rho=1，完全共线
print()
print("=" * 74)
print("第 3 部分：rho=1（完全共线），两个起点还会重合吗")
print("=" * 74)

x1, x2 = build(1.0)          # x2 就是 x1
y = make_y(x1, x2)

header = f"{'start':>10} {'steps':>7} {'a':>8} {'b':>8} {'a+b':>8} {'a-b':>8} {'loss':>9}"
print(header)
print("-" * len(header))

for a0, b0 in [(0, 0), (5, -5)]:
    for s in (200, 5000):
        a, b, l = train(a0, b0, x1, x2, y, s)
        print(f"{f'({a0}, {b0})':>10} {s:>7} {a:>8.3f} {b:>8.3f} {a + b:>8.3f} {a - b:>8.3f} {l:>9.5f}")

print("\n  真值 a=2, b=-1，所以 a+b = 1.000")

# ---------------------------------------------------------------- 第 4 部分：SGD vs Adam
print()
print("=" * 74)
print("第 4 部分：同样的起点和学习率，SGD 和 Adam 谁离最优点更近")
print("=" * 74)

x1, x2 = build(RHO)
y = make_y(x1, x2)

# 最小二乘解：这个问题能达到的最好点（不是真值 2 和 -1，被噪声推开了）
X = torch.stack([x1, x2], dim=1)
w_star = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
print(f"最小二乘解：a = {w_star[0]:.4f}, b = {w_star[1]:.4f}   （真值 2 和 -1）\n")


def dist_after(opt_name, lr, steps):
    """从 (0,0) 出发训练 steps 步，返回离最小二乘解的距离。"""
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    if opt_name == "SGD":
        opt = torch.optim.SGD([a, b], lr=lr)
    else:
        opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        loss = ((a * x1 + b * x2 - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ((a.item() - w_star[0].item()) ** 2 + (b.item() - w_star[1].item()) ** 2) ** 0.5


header = f"{'opt':>5} {'lr':>6} {'50':>9} {'100':>9} {'200':>9} {'500':>9}"
print(header)
print("-" * len(header))
for lr in (0.02, 0.05, 0.1, 0.2):
    for name in ("SGD", "Adam"):
        d = [dist_after(name, lr, s) for s in (50, 100, 200, 500)]
        print(f"{name:>5} {lr:>6.2f} " + " ".join(f"{x:>9.4f}" for x in d))

print("\n  表里的数是「离最小二乘解的距离」，越小越好。列是训练步数。")

