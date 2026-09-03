# 01 · Python 進階語法（寫 PyTorch 前一定要會的）

> 你有 C/C++/Java 底子，所以我用**對照**的方式講，直接點出 Python 不一樣的地方。

---

## §1 你來自 C/Java，最容易踩的 5 個坑

### 1.1 變數是「標籤」，不是「盒子」

```python
a = [1, 2, 3]
b = a          # b 和 a 指向同一個 list（像 Java 的 reference）
b.append(4)
print(a)       # [1, 2, 3, 4]  ← a 也變了！

b = a.copy()   # 淺拷貝
import copy
c = copy.deepcopy(a)   # 深拷貝（巢狀結構才需要）
```

**在 PyTorch 裡的對應陷阱：**

```python
w2 = w1                     # 同一個 tensor
w2 = w1.clone()             # 新記憶體，但仍在計算圖上（梯度會流回 w1）
w2 = w1.detach()            # 共享記憶體，但切斷計算圖
w2 = w1.clone().detach()    # ✅ 完全獨立的副本（最常用）
```

### 1.2 可變預設參數（Python 最惡名昭彰的坑）

```python
# ❌ 這個 bug 會讓你 debug 兩小時
def add_layer(x, layers=[]):
    layers.append(x)
    return layers

print(add_layer(1))   # [1]
print(add_layer(2))   # [1, 2]  ← 預設 list 只在定義時建立一次！

# ✅ 正確寫法
def add_layer(x, layers=None):
    if layers is None:
        layers = []
    layers.append(x)
    return layers
```

### 1.3 整數除法

```python
7 / 2      # 3.5   ← Python 的 / 永遠是浮點數（跟 C/Java 不同）
7 // 2     # 3     ← 這才是整數除法
-7 // 2    # -4    ← 向下取整，不是向零取整（跟 C 不同）
-7 % 3     # 2     ← % 結果符號跟除數相同（跟 C 不同）
```

**PyTorch 相關**：算 CNN 輸出尺寸 `(H + 2p - k) // s + 1`，用 `//` 才對。

### 1.4 沒有 switch，但有更好的東西

```python
# match（Python 3.10+）
def get_activation(name):
    match name:
        case "relu":  return nn.ReLU()
        case "gelu":  return nn.GELU()
        case _:       raise ValueError(f"unknown: {name}")

# 實務上 dict 更常見、更好維護
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
act = ACTIVATIONS[name]()      # 注意要加 () 才是實例化
```

### 1.5 縮排就是 block

Tab 和空白不能混用。**統一用 4 個空白**。

---

## §2 序列操作：list / tuple / dict / set

### 2.1 切片 slicing —— NumPy 與 PyTorch 共用同一套

```python
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

a[2:5]      # [2, 3, 4]     含頭不含尾
a[:3]       # [0, 1, 2]
a[7:]       # [7, 8, 9]
a[::2]      # [0, 2, 4, 6, 8]   step=2
a[::-1]     # 反轉
a[-3:]      # [7, 8, 9]     負索引從尾巴數
a[1:8:3]    # [1, 4, 7]     start:stop:step
```

> ★ 這套語法完全適用於 NumPy array 和 PyTorch tensor，而且可以多維：`x[:, 1:3, ::2]`
> （唯一例外：PyTorch tensor **不支援** `[::-1]` 負 step，要用 `torch.flip(x, dims=[0])`）

### 2.2 推導式 comprehension —— 一定要練熟

```python
squares = [i**2 for i in range(10)]
evens   = [i for i in range(20) if i % 2 == 0]
matrix  = [[i*j for j in range(3)] for i in range(3)]

# dict comprehension（PyTorch 裡超常用）
name2idx = {name: i for i, name in enumerate(class_names)}
enc_only = {k: v for k, v in sd.items() if k.startswith("encoder.")}

# generator expression（不佔記憶體，用小括號）
total = sum(p.numel() for p in model.parameters())   # ★ 算參數量的標準寫法
```

### 2.3 解包 unpacking

```python
a, b = b, a                      # 交換，不用暫存變數

first, *rest = [1, 2, 3, 4]      # first=1, rest=[2,3,4]
*init, last  = [1, 2, 3, 4]      # init=[1,2,3], last=4

# PyTorch 常見
B, C, H, W = x.shape             # ★ 每次拿到 4D tensor 都這樣做
B, *spatial = x.shape

# 函式參數展開
cfg = {"in_channels": 3, "out_channels": 64, "kernel_size": 3}
nn.Conv2d(**cfg)                 # 字典展開成關鍵字參數
```

### 2.4 zip / enumerate / sorted

```python
for i, (x, y) in enumerate(dataloader):     # ★ 訓練迴圈標準寫法
    ...

for name, param in model.named_parameters():
    print(f"{name:40s} {tuple(param.shape)}")

# zip(*) 是「轉置」
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
nums, chars = zip(*pairs)        # (1,2,3), ('a','b','c')

sorted(results, key=lambda r: r["val_acc"], reverse=True)[:5]   # 取前 5 好
```

---

## §3 f-string（訓練 log 全靠它）

```python
loss, acc, lr = 0.123456, 0.9873, 0.0001

f"{loss:.4f}"            # '0.1235'        小數 4 位
f"{acc:.2%}"             # '98.73%'        百分比
f"{lr:.2e}"              # '1.00e-04'      科學記號
f"{123456789:,}"         # '123,456,789'   千分位（印參數量超好用）
f"{'name':<20}"          # 左對齊寬 20；> 右對齊；^ 置中
f"{42:05d}"              # '00042'         補零（存 checkpoint 檔名用）
```

除錯神器（Python 3.8+）：

```python
x = 3.14
print(f"{x=}")                # 'x=3.14'  ← 自動印出變數名
print(f"{tensor.shape=}")     # 'tensor.shape=torch.Size([32, 3, 32, 32])'
```

建議固定使用的訓練 log 格式：

```python
print(f"Epoch {ep:3d}/{EPOCHS} | "
      f"train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
      f"val_acc {va_acc:.2%} | lr {lr:.2e} | {dt:.1f}s")
```

---

## §4 函式進階

### 4.1 強制關鍵字參數

```python
def make_model(in_dim, out_dim, *, hidden=128, dropout=0.1):
    ...
make_model(10, 2, hidden=256)      # ✅
make_model(10, 2, 256)             # ❌ TypeError
```

`*` 之後的參數必須用 `name=value` 呼叫。PyTorch 原始碼大量使用，避免呼叫方寫錯順序。

### 4.2 Decorator 裝飾器 ★ PyTorch 到處都是

```python
import functools, time

def timeit(func):
    @functools.wraps(func)             # 保留原函式的 __name__ 與 docstring
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"[{func.__name__}] {time.perf_counter()-t0:.3f}s")
        return result
    return wrapper

@timeit
def evaluate(model, loader): ...
# 等價於 evaluate = timeit(evaluate)
```

**你一定會用到的內建裝飾器：**

```python
@torch.no_grad()            # ★ 整個函式不建計算圖（驗證/測試用）
def evaluate(model, loader): ...

@torch.inference_mode()     # 比 no_grad 更快更省（純推論用，回傳值不能再參與訓練）
def predict(model, x): ...

@staticmethod               # 自訂 autograd.Function 時必用
@property                   # 把方法變成屬性
```

```python
class Model(nn.Module):
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

model.num_params      # 不用加 ()
```

### 4.3 Context manager（with）

```python
with torch.no_grad():
    logits = model(x)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
```

自己寫一個：

```python
from contextlib import contextmanager

@contextmanager
def eval_mode(model):
    """暫時切到 eval，離開後恢復原本狀態。"""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)

with eval_mode(model):
    acc = evaluate(model, val_loader)
```

---

## §5 類別（跟 Java 對照）

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()               # ★ 一定要呼叫，否則 nn.Module 不會初始化
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self._internal = 1               # 單底線 = 約定俗成的 private（無強制）

    def forward(self, x):                # 覆寫父類別方法（不用寫 @Override）
        return self.fc2(torch.relu(self.fc1(x)))

    def __repr__(self):                  # 相當於 Java 的 toString()
        return f"MLP(in={self.fc1.in_features})"
```

| Java | Python |
|---|---|
| `this` | `self`（且**必須**明寫成第一個參數） |
| `super()` | `super().__init__()` |
| `toString()` | `__repr__` / `__str__` |
| `equals()` | `__eq__` |
| `private` | 沒有真 private，用 `_name` 約定 |
| `interface` | `abc.ABC` + `@abstractmethod`，或 duck typing |
| `static` | `@staticmethod` / `@classmethod` |

### 特殊方法（dunder）—— 寫 Dataset 必備

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):                   # len(ds)   ← DataLoader 需要
        return len(self.X)

    def __getitem__(self, idx):          # ds[5]     ← DataLoader 需要
        return self.X[idx], self.y[idx]
```

> ★ **關鍵理解**：`model(x)` 能運作是因為 `nn.Module` 定義了 `__call__`，
> 它會先執行 forward hook 再呼叫你的 `forward(x)`。
> **所以永遠寫 `model(x)`，不要寫 `model.forward(x)`**（後者會跳過 hook）。

---

## §6 dataclass（讓 config 變乾淨）★

```python
from dataclasses import dataclass, field, asdict

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    hidden: list[int] = field(default_factory=lambda: [256, 128])  # ★可變預設值要這樣寫
    device: str = "cuda"
    seed: int = 42

cfg = TrainConfig(lr=3e-4, epochs=100)
print(cfg)            # 自動有好看的 __repr__
print(asdict(cfg))    # 轉 dict，可直接存 json 當實驗紀錄
```

**建議**：把你 `Transformer.py` 開頭那堆大寫常數改成 dataclass。
之後跑實驗只要 `TrainConfig(d_model=64)`，不用改原始碼。

---

## §7 Type hints（做研究一定要寫）

```python
from typing import Optional
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> tuple[float, float]:
    """回傳 (平均 loss, 準確率)。"""
    ...

# Python 3.12 新語法，比 typing.List 好看
def f(x: list[int], y: dict[str, Tensor]) -> Tensor | None: ...
```

> ⚠️ Type hints 執行期**不會**檢查型別，只給 IDE 和人看。但 VS Code 的自動補全會強很多。

---

## §8 檔案與路徑

```python
from pathlib import Path

root = Path(__file__).resolve().parent      # 這個 .py 所在資料夾
data = root / "data" / "2330.csv"           # 用 / 串接，跨平台
data.exists(); data.suffix; data.stem       # True; '.csv'; '2330'
list(root.glob("*.csv"))
list(root.rglob("*.pth"))                   # 遞迴找

ckpt_dir = root / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True) # ★ 存 checkpoint 前一定要這行
```

```python
import json
from dataclasses import asdict
with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
```

---

## §9 例外處理

```python
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
except FileNotFoundError:
    print("找不到 checkpoint，從頭訓練")
    ckpt = None
except Exception as e:
    print(f"載入失敗：{type(e).__name__}: {e}")
    raise                       # 重新拋出，不要吞掉錯誤

# 主動拋錯做參數檢查（好習慣）
if d_model % nhead != 0:
    raise ValueError(f"d_model({d_model}) 必須能被 nhead({nhead}) 整除")

# assert 適合寫 shape 檢查（★ 強烈建議在 forward 裡加）
assert x.dim() == 4, f"expected 4D input, got {tuple(x.shape)}"
```

---

## §10 動手練習

1. 寫一個 decorator `@count_calls`，記錄函式被呼叫幾次。
2. 用一行 comprehension 算出模型的「可訓練參數量」。
3. 寫一個 `@contextmanager` 叫 `timer("name")`。
4. 把 `Transformer.py` 開頭的超參數改寫成 `@dataclass`。
5. 用 `zip(*)` 把 `[(loss1, acc1), (loss2, acc2), ...]` 拆成 `losses, accs`。

<details>
<summary>參考答案</summary>

```python
# 1
def count_calls(func):
    @functools.wraps(func)
    def wrapper(*a, **kw):
        wrapper.calls += 1
        return func(*a, **kw)
    wrapper.calls = 0
    return wrapper

# 2
n = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 3
@contextmanager
def timer(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{name}] {time.perf_counter()-t0:.3f}s")

# 5
losses, accs = zip(*history)
```
</details>

---

## ✅ 自我檢核

- [ ] 說出 `b = a` 和 `b = a.copy()` 對 list 的差別，並對應到 tensor 的 `clone`/`detach`
- [ ] 解釋為何 `def f(x, lst=[])` 是 bug
- [ ] 不查資料寫出「算模型參數量」的一行程式
- [ ] 解釋為何要寫 `model(x)` 而不是 `model.forward(x)`
- [ ] 寫出一個 `@contextmanager`
