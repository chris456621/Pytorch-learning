"""演化計算：GA / NSGA-II / 用 EC 調神經網路超參數（對應 02_topics/09_EC演化計算.md）

用法：
    python 11_ec_ga_nsga2.py --demo ga        # GA 解 Rastrigin，跟隨機搜尋比較
    python 11_ec_ga_nsga2.py --demo nsga2     # NSGA-II 解 ZDT1（跟你的 NSGA-II.py 對照）
    python 11_ec_ga_nsga2.py --demo hpo       # ★ 用 GA 搜尋 CNN 超參數
    python 11_ec_ga_nsga2.py --demo pareto    # ★★ 多目標：準確率 vs 參數量

===========================================================================
EC 的名詞對照（跟深度學習是完全不同的一套詞彙）
---------------------------------------------------------------------------
  個體 individual   一組候選解（這裡是一個長度 n_dim 的向量）
  族群 population   一堆個體（n_pop 個）
  適應度 fitness    這個解有多好（本檔一律「越小越好」，所以準確率要取負號）
  世代 generation   一輪「選擇 -> 交配 -> 突變 -> 淘汰」
  選擇 selection    挑出比較好的當父母（偏好好解 = exploitation）
  交配 crossover    兩個父代混出子代（組合彼此的優點）
  突變 mutation     隨機小幅擾動（跳出局部極值 = exploration）
  精英保留 elitism  確保最好的解不會在下一代消失
  支配 dominance    多目標時：A 每個目標都不比 B 差、且至少一個更好
  Pareto front      沒有任何解能同時在所有目標上贏過它們的那一群解
===========================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))
from common import set_seed, get_device, count_params, save_json

OUT = Path(__file__).resolve().parent / "outputs"


# ============================================================
# 1. GA 框架（實數編碼 + SBX + 多項式突變）
# ============================================================
class GA:
    def __init__(self, n_pop, n_dim, lo, hi, fitness_fn,
                 p_cross=0.9, p_mut=None, eta_c=20, eta_m=20, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_pop, self.n_dim = n_pop, n_dim
        self.lo = np.full(n_dim, lo, float) if np.isscalar(lo) else np.asarray(lo, float)
        self.hi = np.full(n_dim, hi, float) if np.isscalar(hi) else np.asarray(hi, float)
        self.f = fitness_fn
        self.p_cross = p_cross
        self.p_mut = (1.0 / n_dim) if p_mut is None else p_mut
        self.eta_c, self.eta_m = eta_c, eta_m
        # 初始族群：在合法範圍內隨機灑 n_pop 個點，每個點有 n_dim 個決策變數
        self.pop = self.rng.uniform(self.lo, self.hi, (n_pop, n_dim))
        self.fit = np.array([self.f(x) for x in self.pop])

    def tournament(self, k=2):
        # 二元錦標賽選擇：隨機抓 k 個來比，誰的 fitness 好誰當父母。
        # 這樣既偏好好解（exploitation），又不會讓最強的完全壟斷（保留多樣性）。
        idx = self.rng.integers(0, self.n_pop, (self.n_pop, k))
        best = idx[np.arange(self.n_pop), self.fit[idx].argmin(axis=1)]
        return self.pop[best].copy()

    def sbx(self, p1, p2):
        u = self.rng.random(self.n_dim)
        # SBX（模擬二元交配）：讓子代分布在兩個父代附近。
        # eta_c 越大 -> 子代越靠近父代（偏保守）；越小 -> 探索範圍越大。
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta_c + 1)),
                        (1 / (2 * (1 - u) + 1e-12)) ** (1 / (self.eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, self.lo, self.hi), np.clip(c2, self.lo, self.hi)

    def mutate(self, x):
        # 多項式突變：隨機挑幾個維度做小幅擾動。
        # p_mut 預設 1/n_dim -> 平均每個個體變動一個維度。
        m = self.rng.random(self.n_dim) < self.p_mut
        u = self.rng.random(self.n_dim)
        delta = np.where(u < 0.5,
                         (2 * u) ** (1 / (self.eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        x = np.where(m, x + delta * (self.hi - self.lo), x)
        return np.clip(x, self.lo, self.hi)

    def step(self):
        parents = self.tournament()
        kids = []
        for i in range(0, self.n_pop, 2):
            p1, p2 = parents[i], parents[(i + 1) % self.n_pop]
            c1, c2 = self.sbx(p1, p2) if self.rng.random() < self.p_cross else (p1.copy(), p2.copy())
            kids += [self.mutate(c1), self.mutate(c2)]
        kids = np.array(kids[:self.n_pop])
        kid_fit = np.array([self.f(x) for x in kids])

        # ★ 精英保留：父代+子代合併取最好的 n_pop 個
        allp = np.vstack([self.pop, kids])
        allf = np.concatenate([self.fit, kid_fit])
        # ★ 精英保留：父代與子代放在一起排序，取最好的 n_pop 個。
        #   沒有這一步的話，最好的解可能在下一代就消失了。
        keep = allf.argsort()[:self.n_pop]
        self.pop, self.fit = allp[keep], allf[keep]
        return self.fit[0], self.pop[0]

    @property
    def diversity(self):
        return float(self.pop.std(axis=0).mean())


# ============================================================
# 2. NSGA-II 的兩個核心機制（向量化）
# ============================================================
def fast_non_dominated_sort(F):
    """F: (N, M) 目標值（最小化）。回傳每個個體的 rank（0 最好）。"""
    F = torch.as_tensor(F, dtype=torch.float64)
    N = F.shape[0]
    # 用廣播一次算出「誰支配誰」，不用雙層 for 迴圈（快很多）。
    #   F[:, None, :] 形狀 (N,1,M)；F[None, :, :] 形狀 (1,N,M)
    #   相比較後得到 (N,N,M)，再 .all(-1) 沿目標維度收斂成 (N,N)
    # 支配的定義：每個目標都不比對方差(le)，且至少一個嚴格較好(lt)。
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt                                  # dom[i,j]: i 支配 j
    n = dom.sum(0).clone()
    rank = torch.full((N,), -1, dtype=torch.long)
    cur, r = (n == 0), 0
    while cur.any():
        rank[cur] = r
        n = n - dom[cur].sum(0)
        n[cur] = -1
        cur, r = (n == 0), r + 1
    return rank.numpy()


def crowding_distance(F):
    """同一 front 內的擁擠距離。F: (n, M)"""
    F = np.asarray(F, float)
    n, M = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for m in range(M):
        order = F[:, m].argsort()
        f = F[order, m]
        # ★ 邊界解的擁擠距離設成無限大 -> 永遠會被保留，
        #   這樣 Pareto front 的兩端才不會塌掉。
        d[order[0]] = d[order[-1]] = np.inf        # ★ 邊界解永遠保留
        span = max(f[-1] - f[0], 1e-12)
        d[order[1:-1]] += (f[2:] - f[:-2]) / span
    return d


class NSGAII:
    def __init__(self, n_pop, n_dim, lo, hi, objectives_fn, seed=0, **kw):
        self.ga = GA(n_pop, n_dim, lo, hi, lambda x: 0.0, seed=seed, **kw)
        self.obj = objectives_fn
        self.pop = self.ga.pop
        self.F = np.array([self.obj(x) for x in self.pop])

    def _select(self, pop, F, n):
        rank = fast_non_dominated_sort(F)
        keep = []
        r = 0
        while len(keep) < n:
            idx = np.where(rank == r)[0]
            if len(keep) + len(idx) <= n:
                keep.extend(idx.tolist())
            else:
                cd = crowding_distance(F[idx])
                order = idx[np.argsort(-cd)]        # ★ 擁擠距離大的優先（維持多樣性）
                keep.extend(order[: n - len(keep)].tolist())
            r += 1
        keep = np.array(keep)
        return pop[keep], F[keep], rank[keep]

    def step(self):
        rng = self.ga.rng
        n = len(self.pop)
        rank = fast_non_dominated_sort(self.F)
        # 二元錦標賽：rank 小的贏
        idx = rng.integers(0, n, (n, 2))
        win = idx[np.arange(n), rank[idx].argmin(axis=1)]
        parents = self.pop[win]

        kids = []
        for i in range(0, n, 2):
            p1, p2 = parents[i], parents[(i + 1) % n]
            c1, c2 = self.ga.sbx(p1, p2) if rng.random() < self.ga.p_cross else (p1.copy(), p2.copy())
            kids += [self.ga.mutate(c1), self.ga.mutate(c2)]
        kids = np.array(kids[:n])
        Fk = np.array([self.obj(x) for x in kids])

        self.pop, self.F, r = self._select(np.vstack([self.pop, kids]),
                                           np.vstack([self.F, Fk]), n)
        return r


# ============================================================
# 3. Demo：GA 解 Rastrigin
# ============================================================
def rastrigin(x):
    A = 10.0
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def demo_ga(args):
    print("=" * 70)
    print("GA vs 隨機搜尋：Rastrigin 函數（大量局部極值，全域最小值 = 0）")
    print("=" * 70)
    n_dim, n_pop, gens = 20, 60, args.generations
    budget = n_pop * (gens + 1)

    ga = GA(n_pop, n_dim, -5.12, 5.12, rastrigin, seed=args.seed)
    hist = []
    for g in range(gens):
        best, _ = ga.step()
        hist.append(best)
        if (g + 1) % max(1, gens // 10) == 0:
            print(f"  gen {g+1:4d} | best {best:10.4f} | 多樣性 {ga.diversity:.4f}")

    rng = np.random.default_rng(args.seed)
    rand_best = min(rastrigin(x) for x in rng.uniform(-5.12, 5.12, (budget, n_dim)))

    print()
    print(f"  GA        最佳 = {hist[-1]:.4f}   (評估次數 {budget})")
    print(f"  隨機搜尋  最佳 = {rand_best:.4f}   (同樣評估次數)")
    print(f"  ★ GA 好 {rand_best / max(hist[-1], 1e-9):.1f} 倍")
    if ga.diversity < 0.05:
        print("  ★ 多樣性接近 0 -> 過早收斂，可提高突變率或加大族群")
    save_json(OUT / "ec" / "ga_rastrigin.json", {"history": hist, "random": rand_best})


# ============================================================
# 4. Demo：NSGA-II 解 ZDT1
# ============================================================
def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return [f1, f2]


def demo_nsga2(args):
    print("=" * 70)
    print("NSGA-II 解 ZDT1（跟你的 NSGA-II.py 對照）")
    print("=" * 70)
    n_dim, n_pop = 30, 100
    t0 = time.perf_counter()
    algo = NSGAII(n_pop, n_dim, 0.0, 1.0, zdt1, seed=args.seed)
    for g in range(args.generations):
        rank = algo.step()
        if (g + 1) % max(1, args.generations // 10) == 0:
            front = algo.F[rank == 0]
            print(f"  gen {g+1:4d} | front 大小 {len(front):3d} | "
                  f"f1 範圍 [{front[:,0].min():.3f}, {front[:,0].max():.3f}] | "
                  f"f2 最小 {front[:,1].min():.4f}")
    dt = time.perf_counter() - t0

    rank = fast_non_dominated_sort(algo.F)
    front = algo.F[rank == 0]
    # ZDT1 的真實 Pareto front 是 f2 = 1 - sqrt(f1)
    err = np.abs(front[:, 1] - (1 - np.sqrt(front[:, 0]))).mean()
    print()
    print(f"  耗時 {dt:.2f}s   Pareto front 有 {len(front)} 個解")
    print(f"  與真實 front (f2 = 1 - sqrt(f1)) 的平均誤差 = {err:.5f}")
    if err < 0.02:
        print("  ★ 收斂良好（誤差 < 0.02）")
    else:
        print("  ★ 還沒完全收斂。ZDT1 需要 400 代以上，"
              "試試 --generations 400（實測 400 代誤差約 0.03）")
    save_json(OUT / "ec" / "nsga2_zdt1.json",
              {"front": front.tolist(), "error": float(err), "seconds": dt})


# ============================================================
# 5. EC x 深度學習：超參數搜尋
# ============================================================
SPACE = {                       # 名稱: (型別, 下限, 上限)
    "lr":       ("log",   1e-5, 1e-1),
    "wd":       ("log",   1e-6, 1e-1),
    "width":    ("int",   8,    128),
    "n_layers": ("int",   1,    4),
    "dropout":  ("float", 0.0,  0.6),
}


def decode(gene):
    """gene 是 [0,1]^d -> 真實超參數。"""
    cfg = {}
    for g, (name, (kind, lo, hi)) in zip(gene, SPACE.items()):
        g = float(np.clip(g, 0, 1))
        if kind == "log":
            # 學習率這種跨好幾個數量級的參數，要在 log 空間均勻取樣，
            # 否則 [1e-5, 1e-1] 之間隨機取，99% 的樣本都會落在 1e-2 以上。
            cfg[name] = float(10 ** (np.log10(lo) + g * (np.log10(hi) - np.log10(lo))))
        elif kind == "int":
            cfg[name] = int(round(lo + g * (hi - lo)))
        else:
            cfg[name] = float(lo + g * (hi - lo))
    return cfg


def build_mlp(cfg, in_dim=64, n_cls=5):
    layers, prev = [], in_dim
    for _ in range(cfg["n_layers"]):
        layers += [nn.Linear(prev, cfg["width"]), nn.ReLU(), nn.Dropout(cfg["dropout"])]
        prev = cfg["width"]
    layers.append(nn.Linear(prev, n_cls))
    return nn.Sequential(*layers)


def make_toy_task(device, seed=0, n=1500, in_dim=64, n_cls=5):
    """一個可學但不 trivial 的分類任務，讓超參數真的會影響結果。"""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(in_dim, n_cls, generator=g)
    X = torch.randn(n, in_dim, generator=g)
    y = (X @ W + 1.2 * torch.randn(n, n_cls, generator=g)).argmax(1)
    n_tr = int(n * 0.7)
    return (X[:n_tr].to(device), y[:n_tr].to(device),
            X[n_tr:].to(device), y[n_tr:].to(device))


def quick_eval(cfg, data, device, epochs=15, seed=0):
    """★ 便宜的代理評估：只訓練少量 epoch。EC x DL 的關鍵技巧。"""
    Xtr, ytr, Xva, yva = data
    torch.manual_seed(seed)
    model = build_mlp(cfg, Xtr.shape[1], int(ytr.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    crit = nn.CrossEntropyLoss()
    bs = 128
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            loss = crit(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xva).argmax(1) == yva).float().mean().item()
    return acc, count_params(model)[0]


def demo_hpo(args):
    print("=" * 70)
    print("用 GA 搜尋神經網路超參數")
    print("=" * 70)
    set_seed(args.seed)
    device = get_device()
    data = make_toy_task(device, args.seed)
    # n_eval = [0]：用「只有一個元素的 list」當計數器，而不是 n_eval = 0。
    # 原因是 Python 的 closure（下面 fitness 內部函式）只能「讀取」外層變數，
    # 不能直接「重新賦值」外層的 int（會被當成建立一個新的區域變數，報錯）。
    # list 是可變物件，改它裡面的元素（n_eval[0] += 1）不算重新賦值，
    # 所以這是 Python 老手常用的「用容器包住」技巧，繞過這個限制。
    n_eval = [0]

    def fitness(gene):
        # 這個 fitness 函式定義在 demo_hpo 裡面，可以直接讀寫外層的 n_eval、data、device
        # （這就是 closure：內部函式「捕捉」了外部函式的變數）
        n_eval[0] += 1
        acc, _ = quick_eval(decode(gene), data, device, args.proxy_epochs, args.seed)
        return -acc                                    # ★ 最小化 -> 取負

    n_pop = args.pop
    ga = GA(n_pop, len(SPACE), 0.0, 1.0, fitness, seed=args.seed)
    t0 = time.perf_counter()
    for g in range(args.generations):
        best, gene = ga.step()
        if (g + 1) % max(1, args.generations // 5) == 0:
            print(f"  gen {g+1:3d} | best acc {-best:.4f} | 多樣性 {ga.diversity:.4f} | "
                  f"{decode(gene)}")
    dt = time.perf_counter() - t0

    best_cfg = decode(ga.pop[0])
    print(f"\n  GA 最佳: acc={-ga.fit[0]:.4f}  {best_cfg}")
    print(f"  評估次數 {n_eval[0]}，耗時 {dt:.1f}s")

    rng = np.random.default_rng(args.seed + 1)
    rs_best, rs_cfg = -1.0, None
    for _ in range(n_eval[0]):
        c = decode(rng.random(len(SPACE)))
        a, _ = quick_eval(c, data, device, args.proxy_epochs, args.seed)
        if a > rs_best:
            rs_best, rs_cfg = a, c
    print(f"  隨機搜尋（同樣次數）: acc={rs_best:.4f}  {rs_cfg}")
    print(f"  ★ GA 勝出 {-ga.fit[0] - rs_best:+.4f}")
    print("  （評估預算小的時候 GA 未必贏；EC 的優勢要在預算大、空間複雜時才明顯）")


# ============================================================
# 6. ★★ 多目標：準確率 vs 參數量 -> Pareto front
# ============================================================
def demo_pareto(args):
    print("=" * 70)
    print("NSGA-II 多目標：準確率 vs 參數量（★ 很好的研究題目雛形）")
    print("=" * 70)
    set_seed(args.seed)
    device = get_device()
    data = make_toy_task(device, args.seed)

    def objectives(gene):
        cfg = decode(gene)
        acc, n_params = quick_eval(cfg, data, device, args.proxy_epochs, args.seed)
        return [-acc, n_params / 1e3]                  # ★ 兩個都轉成「最小化」

    algo = NSGAII(args.pop, len(SPACE), 0.0, 1.0, objectives, seed=args.seed)
    for g in range(args.generations):
        rank = algo.step()
        if (g + 1) % max(1, args.generations // 5) == 0:
            f = algo.F[rank == 0]
            print(f"  gen {g+1:3d} | front {len(f):3d} 個 | "
                  f"acc 範圍 [{-f[:,0].max():.3f}, {-f[:,0].min():.3f}] | "
                  f"參數量 {f[:,1].min():.1f}K ~ {f[:,1].max():.1f}K")

    rank = fast_non_dominated_sort(algo.F)
    front = algo.F[rank == 0]
    order = front[:, 1].argsort()
    print("\n  Pareto front（依參數量排序）：")
    print(f"  {'參數量(K)':>10} {'準確率':>10}   對應超參數")
    for i in order:
        cfg = decode(algo.pop[rank == 0][i])
        print(f"  {front[i,1]:>10.2f} {-front[i,0]:>10.4f}   "
              f"width={cfg['width']:>3} layers={cfg['n_layers']} "
              f"lr={cfg['lr']:.1e} drop={cfg['dropout']:.2f}")
    print("\n  ★ 這張表回答的是「只能用 XX 參數時最好能到幾趴」，")
    print("    比單一個「我的準確率 93.2%」有價值得多。")
    print("    把它畫成散佈圖就是論文裡的 Pareto front 圖。")
    save_json(OUT / "ec" / "pareto.json",
              {"front": front.tolist(),
               "configs": [decode(g) for g in algo.pop[rank == 0]]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=["ga", "nsga2", "hpo", "pareto"], default="ga")   # 選要跑哪個示範
    ap.add_argument("--generations", type=int, default=100,
                    help="ga 建議 100+；nsga2 建議 400+；hpo/pareto 建議 20~40")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--proxy-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # dispatch table：把「字串 -> 函式」的對應關係放進字典，
    # 用 args.demo 這個字串當 key 查出對應的函式，最後 (args) 呼叫它。
    # 等價於寫一長串 if/elif，但更精簡、加新的 demo 只要加一行字典項目。
    {"ga": demo_ga, "nsga2": demo_nsga2, "hpo": demo_hpo, "pareto": demo_pareto}[args.demo](args)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()