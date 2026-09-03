"""共用工具箱：所有範例都會 import 這個檔案。

===========================================================================
本檔用到的 Python 語法（看不懂就去查 01_syntax/00_看不懂時先讀這裡.md）
---------------------------------------------------------------------------
  from __future__ import annotations   讓型別提示可以寫得更自由（照抄即可）
  def f(x: int) -> str:                型別提示，執行時「不檢查」，只給 IDE 看
  @contextmanager / yield              自訂 with 區塊              -> 解碼器 §5 §12
  @torch.no_grad()                     裝飾器                      -> 解碼器 §4
  @property                            讓方法用起來像屬性          -> 解碼器 §4
  f"{x:.4f}"  f"{x=}"                  字串格式化                  -> 解碼器 §1
  lambda x: ...                        匿名函式                    -> 解碼器 §2
  sum(p.numel() for p in ...)          generator 推導式            -> 解碼器 §3
  a, b = b, a  /  B,C,H,W = x.shape    解包                        -> 解碼器 §7
  x if cond else y                     三元運算式                  -> 解碼器 §8
  Path(a) / "b"                        路徑串接（不是除法）        -> 解碼器 §15
  *args / **kwargs                     可變參數                    -> 解碼器 §6
===========================================================================


用法：
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent))
    from common import set_seed, get_device, train_one_epoch, evaluate, ...

注意：檔名開頭的 "00_" 不是合法的 Python 識別字，所以本檔同時被複製為
common.py（執行 `python 00_common.py` 會自動產生）。
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# 1. 環境與可重現性
# ============================================================
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """固定所有隨機來源。deterministic=True 完全可重現但較慢。

    語法說明：
      seed: int = 42   -> "int" 是型別提示（不強制），"= 42" 才是預設值
      -> None          -> 宣告這個函式沒有回傳值
    使用方式：
      每個實驗開頭呼叫一次 set_seed(42)，之後所有隨機都可重現。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)   # os.environ 是環境變數的字典
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def get_device(verbose: bool = True) -> torch.device:
    """回傳可用的裝置。使用方式：device = get_device()，之後 x.to(device)。"""
    # 三元運算式：條件成立取前者，否則取後者（跟 C 的 ?: 順序不同）
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if dev.type == "cuda":
            p = torch.cuda.get_device_properties(0)
            # f-string：{} 內是運算式，: 之後是格式（.1f = 小數 1 位）
            # 相鄰的兩個字串會自動接起來，所以這是「一個」字串分兩行寫
            print(f"[device] {p.name}  {p.total_memory / 1e9:.1f} GB  "
                  f"bf16={torch.cuda.is_bf16_supported()}")
        else:
            print("[device] CPU")
    return dev


def speed_setup() -> None:
    """放在訓練腳本開頭的效能樣板。"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ============================================================
# 2. 檢查與除錯
# ============================================================
def inspect(name: str, t: torch.Tensor) -> None:
    """一行看完 tensor 的所有健康指標。

    使用方式：inspect("logits", logits)
    輸出範例：logits  shape=(32, 10) dtype=torch.float32 dev=cuda:0 ...
    什麼時候用：訓練出現 nan 或 shape 錯誤時，從輸入往輸出一路印。
    """
    tf = t.float()          # 統一轉 float 才能算 min/max/mean（bool、int 不能算 mean）
    # {name:<16} = 靠左對齊寬度 16；tuple(t.shape) 讓輸出是 (2,3) 而不是 torch.Size([2,3])
    print(f"{name:<16} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
          f"grad={t.requires_grad} "
          f"min={tf.min().item():.4g} max={tf.max().item():.4g} "
          f"mean={tf.mean().item():.4g} "
          f"nan={torch.isnan(tf).sum().item()} inf={torch.isinf(tf).sum().item()}")


def count_params(model: nn.Module) -> tuple[int, int]:
    """回傳 (總參數量, 可訓練參數量)。

    語法說明：
      sum(p.numel() for p in model.parameters())
        └ 這是 generator 推導式（小括號版的推導式），可以讀成：
          「對 model 的每個參數 p，取出 p.numel()（元素個數），全部加起來」
        └ p.numel() = number of elements，例如 (64,3,3,3) 的卷積核就是 1728
      if p.requires_grad  -> 過濾條件放最後，只算「要訓練的」
      return total, trainable  -> 回傳兩個值時 Python 自動打包成 tuple
    使用方式：
      total, trainable = count_params(model)      # 解包成兩個變數
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module) -> None:
    total, trainable = count_params(model)
    print(f"[model] total={total:,}  trainable={trainable:,}  "
          f"fp32_size={total * 4 / 1e6:.2f} MB")


def grad_stats(model: nn.Module, top_k: int = 0) -> float:
    """印出每層梯度範數，回傳全域梯度範數。

    使用方式：loss.backward() 之後呼叫 grad_stats(model)
    怎麼看結果：
      grad=None      這層完全沒接到梯度（被 detach 或沒參與 forward）
      |g| ~ 1e-8     梯度消失 -> 換 ReLU/加 BatchNorm/加殘差
      |g| ~ 1e+4     梯度爆炸 -> 加 clip_grad_norm_ 或降 lr
      |g| 正常但 loss 不降 -> lr 太小或模型容量不足
    """
    total_sq, rows = 0.0, []
    for name, p in model.named_parameters():
        if p.grad is None:
            rows.append((name, None))
            continue
        g = p.grad.detach()
        total_sq += g.pow(2).sum().item()
        rows.append((name, g.norm().item()))
    if top_k:
        rows = sorted([r for r in rows if r[1] is not None],
                      key=lambda r: -r[1])[:top_k]
    for name, gn in rows:
        print(f"  {name:<45} {'grad=None  <-- 沒接到梯度' if gn is None else f'|g|={gn:.3e}'}")
    total = total_sq ** 0.5
    print(f"  {'TOTAL':<45} |g|={total:.4e}")
    return total


def debug_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """逐層印出輸出的 shape 與統計，找出哪一層開始壞掉。

    使用方式：debug_forward(model, x[:2])      # 拿兩筆資料跑一次就好
    怎麼看：正常的網路每層輸出的 std 應該維持在 0.1 ~ 10 之間。
           一路縮小到 1e-5 就是初始化或架構有問題。
    ⚠️ 只對「一層接一層」的模型有效（Sequential 那種）。
       有分支的模型（ResNet）要改用 forward hook。
    """
    h = x
    # named_children() 產生 (名字, 子模組) 的 pair，所以用兩個變數接
    for name, layer in model.named_children():
        h = layer(h)
        hf = h.float()
        print(f"  {name:<22} {str(tuple(h.shape)):<24} "
              f"mean={hf.mean():.4f} std={hf.std():.4f} "
              f"nan={torch.isnan(hf).any().item()}")
    return h


def sanity_check(loader: DataLoader, name: str = "loader") -> None:
    """訓練前必跑：檢查資料管線。

    使用方式：建好 DataLoader 之後、開始訓練之前呼叫一次。
    要檢查什麼：
      1. x 的數值範圍合理嗎？（Normalize 後大致在 -3 ~ 3）
      2. y 的 dtype 是 int64 嗎？值域是 0 ~ C-1 嗎（不是 1~C）？
      3. batch 數與樣本數符合預期嗎？
    """
    # iter(loader) 取得迭代器，next(...) 拿第一個 batch（不會跑完整個 epoch）
    x, y = next(iter(loader))
    print(f"[{name}] batches={len(loader)}  samples={len(loader.dataset)}")
    print(f"  x: shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.float().min():.3f} max={x.float().max():.3f} "
          f"mean={x.float().mean():.3f}")
    print(f"  y: shape={tuple(y.shape)} dtype={y.dtype} "
          f"unique={y.unique().tolist()[:12]}")
    assert not torch.isnan(x.float()).any(), "輸入含 NaN！"


@contextmanager
def timer(name: str = "block"):
    """計時器。使用方式：

        with timer("forward"):
            out = model(x)
        # 離開區塊時自動印出 [forward] 0.123s

    語法說明：@contextmanager + yield 是「自訂 with 區塊」的標準寫法。
      yield 之前的程式碼 = 進入 with 時執行
      yield             = with 區塊裡的程式碼在這裡執行
      finally 裡的      = 離開 with 時執行（就算中途出錯也會執行）
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[{name}] {time.perf_counter() - t0:.3f}s")


@contextmanager
def eval_mode(model: nn.Module):
    """暫時切到 eval 模式，離開後自動恢復原狀。

        with eval_mode(model):
            acc = evaluate(...)     # 這裡 model 是 eval
        # 離開後自動變回原本的 train/eval 狀態

    為什麼需要：手寫 model.eval() ... model.train() 很容易忘記寫回來，
                尤其中途拋出例外時。用 with 就保證會恢復。
    """
    was_training = model.training      # model.training 是 True/False 的屬性
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


# ============================================================
# 3. 訓練輔助
# ============================================================
class AverageMeter:
    """用樣本數加權的平均，避免最後一個小 batch 造成偏差。

    為什麼需要：如果直接 sum(每個batch的loss) / batch數，
    最後一個 batch 只有 7 筆卻跟 128 筆的 batch 同等權重 -> 平均是錯的。

    使用方式：
        m = AverageMeter()
        for x, y in loader:
            loss = ...
            m.update(loss.item(), y.size(0))     # 值, 這個 batch 有幾筆
        print(m.avg)

    語法說明：@property 讓 m.avg 可以不加括號使用（像屬性一樣）。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class EarlyStopping:
    """驗證指標連續 patience 次沒進步就停。

    使用方式：
        stopper = EarlyStopping(patience=10, mode="max")   # 監控準確率用 max
        for ep in range(EPOCHS):
            ...
            if stopper(val_acc):        # 回傳 True = 這次是新的最佳 -> 存檔
                save_checkpoint(...)
            if stopper.should_stop:     # 連續沒進步太多次
                break

    語法說明：
      __call__ 讓「物件」可以像函式一樣被呼叫 -> stopper(val_acc)
      mode="min" 用於 loss（越小越好）；mode="max" 用於 accuracy（越大越好）
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = "min") -> None:
        assert mode in ("min", "max")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        # 定義 __call__ 之後，stopper(x) 就等於 stopper.__call__(x)
        if self.best is None:                     # 第一次呼叫，還沒有基準
            self.best = metric
            return True
        improved = (metric < self.best - self.min_delta) if self.mode == "min" \
            else (metric > self.best + self.min_delta)
        if improved:
            self.best, self.counter = metric, 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scheduler=None, amp_dtype=None, clip=None,
                    scheduler_per_batch=False):
    """標準訓練迴圈。回傳 (平均 loss, 準確率)。

    amp_dtype: None 表示不用混合精度；torch.bfloat16 建議在 Ada/Ampere 上使用。
    """
    # ① 切成訓練模式：Dropout 開始隨機丟棄、BatchNorm 用當前 batch 統計量
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()   # 一行建立兩個物件（多重賦值）
    use_amp = amp_dtype is not None and device.type == "cuda"

    for x, y in loader:                  # 每次拿出一個 batch：x=(B,...) y=(B,)
        # ② 把資料搬到 GPU。★ tensor 的 .to() 不會就地修改，一定要重新賦值！
        #    non_blocking=True 搭配 DataLoader 的 pin_memory=True 可以非同步傳輸
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ③ 清空上一步的梯度。★ PyTorch 的 .grad 是「累加」不是「覆寫」，
        #    忘記這行 -> 梯度越滾越大 -> loss 爆炸
        #    set_to_none=True 是把 .grad 設成 None（比填 0 快，也省記憶體）
        optimizer.zero_grad(set_to_none=True)

        # ④ 混合精度區塊：區塊內的運算自動用 bfloat16 算（快 1.5~3 倍、省記憶體）
        #    enabled=False 時這個 with 等於什麼都沒做，所以可以無條件包起來
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            logits = model(x)            # ⑤ forward。★ 用 model(x) 不要用 model.forward(x)
            loss = criterion(logits, y)  # ⑥ 算 loss。logits 是原始分數，不要先 softmax

        # ⑦ 反向傳播：從 loss 沿計算圖往回走，把梯度累加到每個參數的 .grad
        loss.backward()
        if clip:      # 梯度裁剪：把整體梯度範數限制在 clip 以內，防止爆炸
            nn.utils.clip_grad_norm_(model.parameters(), clip)   # ★ 放在 backward 後、step 前
        optimizer.step()                 # ⑧ 用 .grad 更新參數

        # ⑨ batch 級的學習率排程（OneCycle / warmup 用），epoch 級的在外面呼叫
        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        # ⑩ 累積統計。★ 一定要 .item() 把 tensor 變成 Python 數字，
        #    否則會一直抓著整張計算圖不放 -> 記憶體一路漲到 OOM
        bs = y.size(0)                   # 這個 batch 有幾筆（最後一批可能不滿）
        loss_m.update(loss.item(), bs)
        # 拆解： logits.argmax(1) 取每列最大值的索引 = 預測類別 (B,)
        #        == y            逐元素比較，得到 bool tensor (B,)
        #        .float().mean() bool 轉 0/1 再平均 = 這批的準確率
        #        .item()         轉成 Python float
        acc_m.update((logits.argmax(1) == y).float().mean().item(), bs)

    return loss_m.avg, acc_m.avg         # 回傳兩個值 -> 呼叫端寫 loss, acc = train_one_epoch(...)


# ★ @torch.no_grad() 讓整個函式都不建計算圖：省一半記憶體、快很多。
#   它跟 model.eval() 是「兩件不同的事」，兩個都要寫：
#     model.eval()  -> 改變層的行為（Dropout 停止、BatchNorm 換用 running stats）
#     no_grad()     -> 關閉梯度追蹤（省資源）
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype=None):
    """驗證/測試。回傳 (平均 loss, 準確率)。"""
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = amp_dtype is not None and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        bs = y.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update((logits.argmax(1) == y).float().mean().item(), bs)
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes):
    """回傳 acc / macro-F1 / 混淆矩陣。類別不平衡時一定要看這個。"""
    model.eval()
    # 混淆矩陣 cm[i][j] = 「真實是 i、被預測成 j」的筆數
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        # 技巧：把 (真實, 預測) 這個二維座標壓成一維編號，就能用 bincount 一次數完
        #       例如 3 類時，(真實=1, 預測=2) -> 1*3+2 = 5
        idx = y.long() * num_classes + pred
        # bincount 數每個編號出現幾次；minlength 保證長度夠；再 view 回二維
        cm += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)

    tp = cm.diag().float()                    # 對角線 = 每類猜對的筆數
    # cm.sum(0) 沿第 0 維加總 = 每一「欄」的和 = 被預測成該類的總數
    precision = tp / cm.sum(0).clamp(min=1)   # clamp(min=1) 防止除以 0
    # cm.sum(1) 沿第 1 維加總 = 每一「列」的和 = 該類實際的總數
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
        "macro_f1": f1.mean().item(),
        "per_class_f1": [round(v, 4) for v in f1.tolist()],
        "confusion": cm,
    }


def overfit_test(model, dataset, criterion, optimizer, device,
                 n_samples=32, steps=300, verbose_every=50):
    """★ 最重要的除錯測試：能不能過擬合一小撮資料？

    loss 應降到接近 0。做不到代表訓練流程有 bug，調參沒有意義。
    """
    from torch.utils.data import Subset
    sub = Subset(dataset, range(min(n_samples, len(dataset))))
    loader = DataLoader(sub, batch_size=n_samples, shuffle=False)
    loss = acc = 0.0
    for step in range(steps):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        if step % verbose_every == 0 or step == steps - 1:
            print(f"  [overfit] step {step:4d}  loss={loss:.6f}  acc={acc:.2%}")
    ok = loss < 0.05
    print("  [overfit] " + ("通過，訓練流程正常 OK"
                            if ok else "失敗！檢查 zero_grad / backward / step / 標籤"))
    return ok


# ============================================================
# 4. Checkpoint 與實驗紀錄
# ============================================================
def save_checkpoint(path, epoch, model, optimizer=None, scheduler=None,
                    best_metric=None, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "epoch": epoch,
        "model": model.state_dict(),
        "best_metric": best_metric,
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler"] = scheduler.state_dict()
    if config is not None:
        ck["config"] = asdict(config) if is_dataclass(config) else config
    torch.save(ck, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and "scheduler" in ck:
        scheduler.load_state_dict(ck["scheduler"])
    return ck.get("epoch", 0), ck.get("best_metric")


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding="utf-8")


class Tee:
    """同時輸出到終端機和檔案。用法：sys.stdout = Tee(path)"""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ============================================================
# 5. 視覺化
# ============================================================
def _get_plt(save_path=None):
    """取得 matplotlib。沒安裝時回傳 None 並提示，而不是讓整個訓練崩掉。"""
    try:
        import matplotlib
    except ImportError:
        print("[warn] 沒有安裝 matplotlib，跳過繪圖。安裝指令：pip install matplotlib")
        return None
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_history(history, save_path=None):
    plt = _get_plt(save_path)
    if plt is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(ep, history["val_acc"], color="green")
    best = max(history["val_acc"]); bi = history["val_acc"].index(best) + 1
    axes[1].scatter([bi], [best], color="red", zorder=5)
    axes[1].annotate(f"best {best:.4f} @ep{bi}", (bi, best),
                     textcoords="offset points", xytext=(5, -12))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Val Accuracy"); axes[1].grid(alpha=.3)

    if "lr" in history:
        axes[2].plot(ep, history["lr"], color="orange")
        axes[2].set_yscale("log")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate"); axes[2].grid(alpha=.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved -> {save_path}")
    return fig


def plot_confusion(cm, class_names=None, normalize=True, save_path=None):
    plt = _get_plt(save_path)
    if plt is None:
        return None

    cm = cm.numpy().astype(float) if torch.is_tensor(cm) else np.asarray(cm, float)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    n = cm.shape[0]
    names = class_names or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(1 + n * 0.6, 1 + n * 0.55))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def show_batch(images, labels=None, class_names=None, n=16,
               mean=None, std=None, save_path=None):
    plt = _get_plt(save_path)
    if plt is None:
        return None

    imgs = images[:n].detach().cpu().float()
    if mean is not None:
        m = torch.tensor(mean).view(1, -1, 1, 1)
        s = torch.tensor(std).view(1, -1, 1, 1)
        imgs = imgs * s + m
    imgs = imgs.clamp(0, 1)

    rows = int(np.ceil(len(imgs) ** 0.5))
    fig, axes = plt.subplots(rows, rows, figsize=(rows * 1.6, rows * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(imgs):
            continue
        img = imgs[i]
        img = img.squeeze(0) if img.shape[0] == 1 else img.permute(1, 2, 0)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if labels is not None:
            nm = class_names[labels[i]] if class_names else str(int(labels[i]))
            ax.set_title(nm, fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ============================================================
# 6. 讓其他檔案可以 `from common import ...`
# ============================================================
def _install_as_common():
    src = Path(__file__).resolve()
    dst = src.parent / "common.py"
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dst)
        print(f"[setup] 已產生 {dst.name}，其他範例可以 from common import ...")


# ★ Python 慣用寫法：「只有直接執行這個檔案時才跑下面的程式碼；
#   被其他檔案 import 時不要跑」。
#   __name__ 這個變數：直接執行（python 00_common.py）時值是 "__main__"；
#                      被 import（from common import xxx）時值是模組名稱 "common"
#   好處：別的檔案可以放心 import 這裡的函式，不會意外觸發下面的自我測試。
if __name__ == "__main__":
    _install_as_common()
    print("=" * 62)
    set_seed(42)
    dev = get_device()
    x = torch.randn(4, 3, 8, 8, device=dev)
    inspect("x", x)

    m = nn.Sequential(nn.Flatten(1), nn.Linear(192, 64), nn.ReLU(),
                      nn.Linear(64, 10)).to(dev)
    print_model_summary(m)
    print("[debug_forward]")
    debug_forward(m, x)

    m(x).sum().backward()
    print("[grad_stats]")
    grad_stats(m)

    with timer("100 forwards"):
        for _ in range(100):
            m(x)
    print("=" * 62)
    print("common.py 自我測試通過")
