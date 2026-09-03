"""ANN / MLP 完整流程 + 系統性調參實驗（對應 02_topics/01_ANN.md）

用法：
    python 03_ann_mnist.py                      # 訓練 baseline
    python 03_ann_mnist.py --epochs 20
    python 03_ann_mnist.py --experiments        # ★ 跑 W7 的六組對照實驗
    python 03_ann_mnist.py --overfit            # ★ 32 筆過擬合測試（除錯用）
    python 03_ann_mnist.py --synthetic          # 沒網路時用合成資料

===========================================================================
本檔用到的語法
---------------------------------------------------------------------------
  @dataclass                 設定檔類別，自動產生建構子      -> 解碼器 §11
  sys.path.append(...)       讓 Python 找得到同資料夾的 common.py
  from common import a, b    只匯入需要的名字（不用寫 common.a）
  nn.Sequential(*layers)     ★ 星號把 list 拆成一個個參數     -> 解碼器 §7
  mk = lambda d, sh: ...     把重複的長句子包成短名字         -> 解碼器 §2
  dict(hidden=(256,), ...)   用關鍵字建字典，等於 {"hidden": (256,), ...}
  **over                     把字典拆成關鍵字參數傳進去       -> 解碼器 §7
===========================================================================
"""

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

# 把「這個檔案所在的資料夾」加進 Python 的搜尋路徑，
# 這樣不論你在哪個目錄執行，都找得到隔壁的 common.py
sys.path.append(str(Path(__file__).resolve().parent))
from common import (set_seed, get_device, speed_setup, sanity_check, plot_history,
                    print_model_summary, train_one_epoch, evaluate, evaluate_full,
                    overfit_test, EarlyStopping, save_checkpoint, save_json,
                    plot_confusion, grad_stats)

OUT = Path(__file__).resolve().parent / "outputs"


# ============================================================
# @dataclass 會自動幫你產生 __init__ 和好看的 __repr__。
# 用法：cfg = Cfg(lr=3e-4)  <- 沒指定的欄位就用下面的預設值
# 好處：跑實驗時只要改這裡，不用翻遍整個檔案找散落的常數
@dataclass
class Cfg:
    hidden: tuple = (512, 256)
    dropout: float = 0.2
    use_bn: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # sgd | adam | adamw
    batch_size: int = 128
    epochs: int = 15
    label_smoothing: float = 0.0
    seed: int = 42
    normalize: bool = True
    patience: int = 8
    amp: bool = True


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2, use_bn=True):
        super().__init__()
        layers, prev = [], in_dim        # 一行同時建立兩個變數
        for h in hidden:                 # hidden 是 (512, 256) 這種 tuple
            # bias=not use_bn：後面接 BatchNorm 時 bias 會被 BN 的平移項抵消，
            # 留著只是浪費參數，所以有 BN 就不要 bias
            layers.append(nn.Linear(prev, h, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))    # ★ 最後一層不加激活（回傳 logits）
        # ★ 星號必須加！nn.Sequential 要的是「一層、一層、一層」，
        #   不是「一個裝著層的 list」。少了 * 會報錯。
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # flatten(1)：從第 1 維開始攤平，保留第 0 維（batch）
        # (B,1,28,28) -> (B, 1*28*28) = (B, 784)
        return self.net(x.flatten(1))


# ============================================================
def get_data(cfg, synthetic=False, root="./data"):
    from torch.utils.data import DataLoader, TensorDataset, random_split

    if not synthetic:
        try:
            from torchvision import datasets, transforms as T
            tfm = [T.ToTensor()]
            if cfg.normalize:
                tfm.append(T.Normalize((0.1307,), (0.3081,)))
            tfm = T.Compose(tfm)
            full = datasets.MNIST(root, train=True, download=True, transform=tfm)
            test = datasets.MNIST(root, train=False, download=True, transform=tfm)
            g = torch.Generator().manual_seed(cfg.seed)
            train, val = random_split(full, [54000, 6000], generator=g)
            print("[data] MNIST 載入成功")
        except Exception as e:
            print(f"[data] MNIST 載入失敗（{type(e).__name__}），改用合成資料")
            synthetic = True

    if synthetic:
        g = torch.Generator().manual_seed(cfg.seed)
        n, c = 6000, 10
        # 每類一個隨機中心 + 雜訊。★ 中心刻意靠得很近，讓任務「不是一眼滿分」，
        # 否則 --experiments 的各組對照會全部 100%，看不出差異。
        centers = torch.randn(c, 784, generator=g) * 0.20
        y = torch.randint(0, c, (n,), generator=g)
        X = (centers[y] + torch.randn(n, 784, generator=g)).view(n, 1, 28, 28)
        ds = TensorDataset(X, y)
        train, val, test = random_split(ds, [4000, 1000, 1000], generator=g)
        print("[data] 使用合成資料（10 類高斯團，刻意設計成有難度）")

    # 把重複的 DataLoader 建構包成一個小函式，避免抄三遍。
    # sh 同時當 shuffle 和 drop_last：訓練集兩個都 True，驗證/測試都 False。
    #   shuffle=True   打亂順序，避免模型學到資料的排列
    #   drop_last=True 丟掉最後不滿一批的資料（BatchNorm 在 batch 太小時不穩）
    mk = lambda d, sh: DataLoader(d, cfg.batch_size, shuffle=sh, num_workers=0,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=sh)
    return mk(train, True), mk(val, False), mk(test, False), train


def build_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)


# ============================================================
def run(cfg, synthetic=False, tag="baseline", verbose=True, save=True):
    set_seed(cfg.seed)
    speed_setup()
    device = get_device(verbose=verbose)

    tr, va, te, train_ds = get_data(cfg, synthetic)
    if verbose:
        sanity_check(tr, "train")

    model = MLP(784, cfg.hidden, 10, cfg.dropout, cfg.use_bn).to(device)
    if verbose:
        print_model_summary(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    stopper = EarlyStopping(patience=cfg.patience, mode="max")
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best = 0.0
    out_dir = OUT / f"ann_{tag}"

    for ep in range(1, cfg.epochs + 1):
        trl, tra = train_one_epoch(model, tr, criterion, optimizer, device,
                                   amp_dtype=amp_dtype, clip=1.0)
        val, vaa = evaluate(model, va, criterion, device, amp_dtype=amp_dtype)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # zip(hist, [...]) 把「字典的 key」跟「這一輪的數值」配成對：
        #   ("train_loss", trl), ("train_acc", tra), ...
        # 這樣不用寫五行 hist["train_loss"].append(trl)
        for k, v in zip(hist, [trl, tra, val, vaa, lr_now]):
            hist[k].append(v)
        if verbose:
            print(f"  ep {ep:3d}/{cfg.epochs} | train {trl:.4f}/{tra:.2%} | "
                  f"val {val:.4f}/{vaa:.2%} | lr {lr_now:.2e}")

        # stopper(vaa) 回傳 True 代表「這是目前最好的」-> 存檔
        if stopper(vaa):
            best = vaa
            if save:
                save_checkpoint(out_dir / "best.pt", ep, model, optimizer,
                                scheduler, best_metric=best, config=cfg)
        if stopper.should_stop:
            if verbose:
                print(f"  early stop @ep{ep}")
            break

    if save:
        plot_history(hist, out_dir / "curves.png")
        save_json(out_dir / "config.json", asdict(cfg))
        save_json(out_dir / "history.json", hist)

    m = evaluate_full(model, te, device, 10)
    if verbose:
        print(f"  [test] acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
        if save:
            plot_confusion(m["confusion"], save_path=out_dir / "confusion.png")
    return {"best_val_acc": best, "test_acc": m["acc"], "test_f1": m["macro_f1"],
            "history": hist}


# ============================================================
EXPERIMENTS = {
    # 名稱            → 覆寫的欄位                              → 你要觀察什麼
    "E0_baseline":  dict(hidden=(256,), dropout=0.0, use_bn=False),
    "E1_wider":     dict(hidden=(1024,), dropout=0.0, use_bn=False),
    "E2_deeper":    dict(hidden=(256,) * 8, dropout=0.0, use_bn=False),
    "E3_deeper_bn": dict(hidden=(256,) * 8, dropout=0.0, use_bn=True),
    "E4_dropout":   dict(hidden=(256,), dropout=0.5, use_bn=False),
    "E5_sgd":       dict(hidden=(256,), dropout=0.0, use_bn=False,
                         optimizer="sgd", lr=0.1),
    "E6_no_norm":   dict(hidden=(256,), dropout=0.0, use_bn=False, normalize=False),
    "E7_high_lr":   dict(hidden=(256,), dropout=0.0, use_bn=False, lr=1.0),
}

NOTES = """
你應該觀察到的現象（跟你的結果對照）：
  E2_deeper    比 E0 差            -> 梯度消失（8 層無 BN）
  E3_deeper_bn 明顯救回來          -> BatchNorm 讓深層網路可訓練
  E4_dropout   train 較差 val 較好 -> 正則化在作用
  E6_no_norm   明顯變差            -> 輸入正規化不是可選項
  E7_high_lr   發散或震盪          -> lr 是最重要的超參數

★ 用 --synthetic 時，E2 / E3 / E7 的對比看得出來，但 E4 / E6 看不出來
  （合成資料本來就已經是標準常態，而且沒有過擬合的空間）。
  E4 / E6 要用真實 MNIST 跑才有意義 —— 把 --synthetic 拿掉即可（會自動下載）。
"""




def run_experiments(args):
    rows = []
    # .items() 同時拿字典的 key 和 value
    for name, over in EXPERIMENTS.items():
        # **over 把字典拆開當關鍵字參數：
        #   over = {"hidden": (256,), "dropout": 0.0}
        #   Cfg(epochs=5, seed=42, **over) 等於 Cfg(epochs=5, seed=42, hidden=(256,), dropout=0.0)
        cfg = Cfg(epochs=args.epochs, seed=args.seed, **over)
        print(f"\n{'=' * 66}\n{name}: {over}\n{'=' * 66}")
        r = run(cfg, args.synthetic, tag=name, verbose=args.verbose, save=False)
        rows.append((name, r["best_val_acc"], r["test_acc"], r["test_f1"]))
        print(f"  -> best_val={r['best_val_acc']:.4f}  test={r['test_acc']:.4f}")

    print(f"\n{'=' * 66}\n實驗總表\n{'=' * 66}")
    print(f"{'實驗':<16}{'val_acc':>10}{'test_acc':>10}{'macro_f1':>10}")
    for n, v, t, f in rows:
        print(f"{n:<16}{v:>10.4f}{t:>10.4f}{f:>10.4f}")
    print(NOTES)


def run_overfit(args):
    """★ 最重要的除錯測試：能不能把 32 筆資料背起來？"""
    cfg = Cfg(epochs=1, seed=args.seed)
    set_seed(cfg.seed)
    device = get_device()
    _, _, _, train_ds = get_data(cfg, args.synthetic)
    model = MLP(784, cfg.hidden, 10, dropout=0.0, use_bn=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print("\n[過擬合測試] loss 應該降到接近 0；做不到就代表訓練流程有 bug")
    overfit_test(model, train_ds, nn.CrossEntropyLoss(), opt, device,
                 n_samples=32, steps=200, verbose_every=40)
    print("\n[梯度檢查]")
    grad_stats(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true")       # 開關：吃合成資料
    ap.add_argument("--experiments", action="store_true")     # 開關：跑 W7 的六組對照實驗
    ap.add_argument("--overfit", action="store_true")         # 開關：32 筆過擬合測試
    # dest="verbose" + action="store_false"：
    #   命令列名字是 --quiet，但存進 args 的變數名是 verbose（不是 quiet）
    #   而且邏輯「反過來」：沒寫 --quiet 時 verbose=True（預設印很多東西），
    #                       寫了 --quiet 時 verbose=False（安靜模式）
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args()

    if args.overfit:
        run_overfit(args)
    elif args.experiments:
        run_experiments(args)
    else:
        cfg = Cfg(epochs=args.epochs, seed=args.seed)
        r = run(cfg, args.synthetic, verbose=args.verbose)
        print(f"\n最佳驗證準確率 {r['best_val_acc']:.4f}  測試準確率 {r['test_acc']:.4f}")


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()