"""03 — 从数据到更新：引入奇异值，把 64 个方向压成一个数

【这个项目最终要做什么】
    量神经网络每一步的更新矩阵 ΔW 有几个自由度（erank），
    再看不同优化器、不同限制下这个数怎么变、和收敛快慢有没有关系。
    这一份是第三步：把 02 的手工测量换成工具，把对象从数据换成更新量。

【这一份解决什么问题】
    02 里我用「沿某个方向走一步、看 loss 变多少」来发现平坦方向，那是手工做法。
    这一份做五件事：

        1. 奇异值是什么，秩和它什么关系
        2. erank 公式凭什么代表「用了几个方向」
        3. 证明 σ² 就是那个方向的陡度（恒等式，不是比数字）
        4. 用 erank 量「这份数据能支撑几个参数」
        5. 换对象：64x64 层走一步，看 ΔW 的奇异值和 erank

【这一份的结论】
    1. 秩 = 非零奇异值的个数。四个 2x2 矩阵里，秩 1 的那个第二个 σ 是 0，
       非零个数 1；其余三个都是 2。

    2. erank 公式在极端情况下会退化成「个数」：
           8 个 σ 全相等     -> 8.000（就是秩）
           一个主导、三个极小 -> 1.061（接近 1）
           [2, 1]            -> 1.800，手算 (2+1)^2/(4+1) = 9/5 = 1.8，对上了
       所以这个公式不是黑箱：σ 越平均就越接近「个数」，越悬殊就越接近 1。

    3. 但数值上的 0 不是真的 0。float32 下 [[1,2],[2,4]] 的第二个 σ 相对值是
       1.3e-8。阈值取 1e-6 还是 1e-5 会改变数出来的个数（见第 5 部分两列），
       所以「数非零个数」不可靠 —— 这正是 erank 用连续公式的理由。

    4. σ² 就是那个方向的陡度，这是恒等式不是近似。
       沿第 i 个右奇异向量移动 t，loss 增量 = t^2 * σi^2 / n。
       实测和公式的差在 1e-7 量级（float32 精度），没有任何抽样误差。

    5. erank(X) = 这份数据能支撑几个参数：
           rho=0    -> 2.00       rho=0.95 -> 1.31       rho=1 -> 1.00
       而且 erank 比 σ 比值稳定得多（标准差 0.020 对 5.7），
       因为它不会被极小的 σ 放大。

    6. ΔW 的秩被 batch 卡住：非零 σ 个数正好等于 batch
       （8 -> 8 个、erank 7.4；32 -> 32 个、erank 23.6；200 -> 64 个、erank 39.6）。
       所以量 erank 必须用全批量，否则测到的是 batch 给出的上限。

【这一份还不能说明什么】
    第 5 部分的 y 是随机生成的，不是真实任务。所以 39.6 只能证明「测量跑得通」，
    不能说「真实的训练更新就是 39.6 个自由度」。要等换成真实任务再说。

【我还没搞明白的】
    - 为什么秩一定等于非零奇异值的个数，我只验证过，没推过
    - 全批量时非零 σ 有 64 个，但 erank 只有 39.6；这中间的差是任务结构还是噪声，
      我分不清

Run:  python 03.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
N = 200
RHO = 0.95
TOL_LOOSE = 1e-5        # 判断"非零奇异值"的宽松阈值
TOL_TIGHT = 1e-6        # 严格阈值


def build(rho, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x1 = torch.randn(N, generator=g)
    x2 = rho * x1 + (1 - rho ** 2) ** 0.5 * torch.randn(N, generator=g)
    return x1, x2


def erank(M):
    """effective rank = (Σσ)^2 / Σσ^2，同时把奇异值也返回。"""
    s = torch.linalg.svdvals(M)
    return float(s.sum() ** 2 / (s ** 2).sum()), s


# ---------------------------------------------------------------- 第 1 部分：奇异值是什么
print("=" * 74)
print("第 1 部分：奇异值是什么（四个小矩阵）")
print("=" * 74)

cases = [
    ("diag(3,1)", torch.diag(torch.tensor([3.0, 1.0]))),
    ("[[1,2],[2,4]]", torch.tensor([[1.0, 2.0], [2.0, 4.0]])),
    ("eye(2)", torch.eye(2)),
    ("rotate 45", torch.tensor([[0.7071, -0.7071], [0.7071, 0.7071]])),
]
print(f"{'matrix':>14}  {'sigma':>28}  {'min_rel':>9}  {'nonzero':>7}")
print("-" * 66)
for name, M in cases:
    s = torch.linalg.svdvals(M)
    vals = "[" + ", ".join(f"{v:7.4f}" for v in s.tolist()) + "]"
    rel_min = (s / s[0]).min().item()
    n_nonzero = int((s > TOL_LOOSE * s[0]).sum())
    print(f"{name:>14}  {vals:>28}  {rel_min:>9.1e}  {n_nonzero:>7}")

print("\n  列名：sigma = 奇异值，min_rel = 最小 / 最大，nonzero = 非零个数（阈值 1e-5）。")
print("  秩 = 非零 sigma 的个数。注意 [[1,2],[2,4]] 那行：所谓的零其实是 1e-8 量级。")

# ---------------------------------------------------------------- 第 2 部分：erank 公式
print()
print("=" * 74)
print("第 2 部分：erank 公式凭什么代表「用了几个方向」")
print("=" * 74)


def erank_of(s):
    return float(s.sum() ** 2 / (s ** 2).sum())


s8 = torch.ones(8)
s_one = torch.tensor([1.0, 0.01, 0.01, 0.01])
s_two = torch.tensor([2.0, 1.0])

header = f"{'case':>22}  {'sigmas':>26}  {'erank':>7}"
print(header)
print("-" * len(header))
print(f"{'8 个全相等':>22}  {'1,1,1,1,1,1,1,1':>26}  {erank_of(s8):>7.3f}")
print(f"{'一个主导 + 三个极小':>22}  {'1, .01, .01, .01':>26}  {erank_of(s_one):>7.3f}")
print(f"{'手算验证 [2,1]':>22}  {'2, 1':>26}  {erank_of(s_two):>7.3f}")

print("\n  [2,1] 手算：(2+1)^2 / (4+1) = 9/5 = 1.8，程序给 1.800。")
print("  σ 越平均越接近「个数」，越悬殊越接近 1。所以这个公式是「连续版的秩」。")

# ---------------------------------------------------------------- 第 3 部分：σ² 是陡度
print()
print("=" * 74)
print("第 3 部分：σ² 是不是那个方向的陡度（恒等式验证，不比数字）")
print("=" * 74)

x1, x2 = build(RHO, 0)
g_y = torch.Generator().manual_seed(0)
y = 2.0 * x1 - 1.0 * x2 + 0.1 * torch.randn(N, generator=g_y)
# X 是「设计矩阵」：200 条样本，每行两个变量 [x1, x2]
X = torch.stack([x1, x2], dim=1)
print(f"  X 的形状: {tuple(X.shape)}   （200 条样本 x 2 个变量）")

w_star = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
loss_star = ((X @ w_star - y) ** 2).mean().item()

U, s, Vh = torch.linalg.svd(X)          # X = U diag(s) Vh
print(f"  X 的奇异值: {s[0]:.4f}, {s[1]:.4f}\n")

print(f"{'direction':>10} {'t':>5} {'实测 loss 增量':>16} {'预测 t^2*σ^2/n':>16} {'差':>9}")
print("-" * 62)
for i in (0, 1):
    v = Vh[i]                            # 第 i 个右奇异向量：走这个方向
    for t in (0.1, 0.5):
        w = w_star + t * v
        actual = ((X @ w - y) ** 2).mean().item() - loss_star
        predicted = t ** 2 * s[i].item() ** 2 / N
        print(f"{i:>10} {t:>5.1f} {actual:>16.10f} {predicted:>16.10f} {abs(actual - predicted):>9.1e}")

print("\n  两列差 1e-7 量级 = float32 精度。这不是「对得上」，是恒等式。")

# ---------------------------------------------------------------- 第 4 部分：erank(X)
print()
print("=" * 74)
print("第 4 部分：erank(X) = 这份数据能支撑几个参数")
print("=" * 74)

header = f"{'rho':>6} {'erank':>8} {'sd':>7} {'ratio2':>9} {'ratio2_sd':>10}"
print(header)
print("-" * len(header))
for rho in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
    es, rs = [], []
    for seed in range(100):
        a1, a2 = build(rho, seed)
        e, sv = erank(torch.stack([a1, a2], dim=1))
        es.append(e)
        rs.append((sv[0].item() / sv[1].item()) ** 2 if sv[1] > TOL_TIGHT * sv[0] else float("nan"))
    m = sum(es) / len(es)
    sd = (sum((v - m) ** 2 for v in es) / (len(es) - 1)) ** 0.5
    r = [v for v in rs if v == v]          # 去掉 NaN（rho=1 时最小 sigma 是 0）
    if r:
        rm = sum(r) / len(r)
        rsd = (sum((v - rm) ** 2 for v in r) / (len(r) - 1)) ** 0.5
        r_str, rsd_str = f"{rm:.1f}", f"{rsd:.1f}"
    else:
        r_str, rsd_str = "--", "--"
    print(f"{rho:>6.2f} {m:>8.3f} {sd:>7.3f} {r_str:>9} {rsd_str:>10}")

print("\n  100 次重复。ratio2 = (最大 sigma / 最小 sigma)^2，就是陡度比。")
print("  比一比 sd 那两列：erank 稳得多，因为它不会被极小的 sigma 放大。")
print("  erank 从 2.00 掉到 1.00 —— 就是 02 里「参数分不开」的定量说法。")

# ---------------------------------------------------------------- 第 5 部分：ΔW 和 erank
print()
print("=" * 74)
print("第 5 部分：64x64 的层走一步，ΔW 的奇异值和 erank")
print("=" * 74)


def one_step(batch_size, seed=0):
    """建一个 64x64 的层，走一步，返回更新量 ΔW。"""
    torch.manual_seed(seed)
    layer = nn.Linear(64, 64)
    x = torch.randn(batch_size, 64)
    y = torch.randn(batch_size, 64)

    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss = ((layer(x) - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()

    W_before = layer.weight.detach().clone()      # 必须先 clone，否则会被 step 改掉
    opt.step()
    return layer.weight.detach() - W_before


header = f"{'batch':>6} {'erank':>7} {'nz@1e-5':>8} {'nz@1e-6':>8} {'sigma1':>9} {'sigma_min':>11}"
print(header)
print("-" * len(header))

for b in (8, 32, 200):
    dW = one_step(b)
    e, sv = erank(dW)
    n_loose = int((sv > TOL_LOOSE * sv[0]).sum())
    n_tight = int((sv > TOL_TIGHT * sv[0]).sum())
    print(f"{b:>6} {e:>7.1f} {n_loose:>8} {n_tight:>8} {sv[0]:>9.5f} {sv.min():>11.2e}")

print("\n  满秩是 64。nz 两列是同一个矩阵、不同阈值数出来的非零个数。")
print("  batch=32 那行 32 vs 34：多出来的 2 个是浮点噪声，不是真方向。")
print("  nz 也不该超过 batch —— 超了就说明阈值放太松。")
