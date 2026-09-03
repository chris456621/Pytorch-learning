# 09 · Dataset 與 DataLoader

> 「模型架構決定上限，資料管線決定你能不能跑到那個上限。」
> 而且**資料管線的 bug 最難發現**，因為它不會報錯，只會讓模型悄悄變爛。

---

## §1 兩種 Dataset

### Map-style（99% 的情況用這個）

```python
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, X, y, transform=None):
        # X: (N, W, F) numpy 或 tensor；y: (N,)
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)                    # ★ DataLoader 靠這個決定要抓幾筆

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y                           # ★ 回傳「單一樣本」，不是 batch
```

> ★ **`__getitem__` 回傳的是單一樣本**（不含 batch 維度）。
> 疊成 batch 是 DataLoader 的工作。這是新手最常搞混的地方。

### Iterable-style（資料太大裝不進記憶體、串流資料時用）

```python
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        with open(self.path) as f:
            for line in f:
                yield parse(line)
```

> ⚠️ IterableDataset 不能用 `shuffle=True`（沒有索引可打亂），
> 且多 worker 時要自己處理分片，否則每個 worker 會讀到同樣的資料。

### 現成的 Dataset

```python
from torch.utils.data import TensorDataset
ds = TensorDataset(X_tensor, y_tensor)          # ★ 資料已經是 tensor 時最快

from torchvision import datasets, transforms
train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
train = datasets.ImageFolder(root="./data/train", transform=tfm)  # 依資料夾分類
```

---

## §2 DataLoader 的每個參數

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,             # ★ 訓練 True、驗證/測試 False
    num_workers=4,            # ★ Windows 上建議 0~4，且要有 main guard
    pin_memory=True,          # ★ 有 GPU 時開，配合 .to(device, non_blocking=True)
    drop_last=True,           # ★ 訓練時建議 True（避免最後一個小 batch 讓 BN 不穩）
    persistent_workers=True,  # num_workers>0 時開，避免每個 epoch 重開行程
    prefetch_factor=2,
    collate_fn=None,          # 自訂如何把樣本疊成 batch
    generator=g,              # 傳入 Generator 讓 shuffle 可重現
)

for x, y in loader:           # x: (64, ...), y: (64,)
    ...
len(loader)                   # ★ 有幾個 batch（不是幾筆資料！）
len(loader.dataset)           # 幾筆資料
```

### Windows 的 num_workers 陷阱 ★

```python
# ❌ 直接寫在模組層級 → 子行程重新 import 造成無限遞迴
loader = DataLoader(ds, num_workers=4)
for x, y in loader: ...

# ✅ 必須包在 main guard
def main():
    loader = DataLoader(ds, num_workers=4, persistent_workers=True)
    for x, y in loader: ...

if __name__ == "__main__":
    main()
```

> 你的 8GB 筆電：`num_workers=4` 通常是甜蜜點。
> 資料已經全部在記憶體（例如 TensorDataset）時，`num_workers=0` 反而最快。

---

## §3 自訂 collate_fn（處理不等長序列）★

預設的 collate 是 `torch.stack`，要求所有樣本 shape 相同。
不等長時（NLP、可變長度時序）就要自訂：

```python
from torch.nn.utils.rnn import pad_sequence

def collate_pad(batch):
    """batch 是 list of (seq, label)，seq 長度不一。"""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)   # (B, Lmax, F)
    # ★ 同時產生 padding mask 給 Transformer 用
    mask = torch.arange(padded.size(1))[None, :] >= lengths[:, None]  # (B, Lmax) True=padding
    return padded, torch.stack(labels), mask

loader = DataLoader(ds, batch_size=32, collate_fn=collate_pad)
```

---

## §4 資料切分（★ 錯了整個實驗都白做）

### 一般資料：隨機切

```python
from torch.utils.data import random_split

g = torch.Generator().manual_seed(42)
n = len(ds)
n_train = int(0.7 * n); n_val = int(0.15 * n)
train_ds, val_ds, test_ds = random_split(
    ds, [n_train, n_val, n - n_train - n_val], generator=g
)
```

### 時序資料：一定要按時間切 ★

```python
from torch.utils.data import Subset

n = len(ds)
i1, i2 = int(0.7 * n), int(0.85 * n)
train_ds = Subset(ds, range(0, i1))
val_ds   = Subset(ds, range(i1, i2))
test_ds  = Subset(ds, range(i2, n))
```

> 🔥 **絕對不要對時序資料用 `random_split`**。
> 那等於「用未來的資料預測過去」，準確率會虛高 10~30%，而且完全不可信。
> 這是金融/時序論文被退稿的頭號原因。

### 類別不平衡：分層抽樣或加權採樣

```python
from torch.utils.data import WeightedRandomSampler

counts = np.bincount(labels)                       # 每類幾筆
class_w = 1.0 / counts
sample_w = class_w[labels]                         # 每個樣本的權重
sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)

loader = DataLoader(ds, batch_size=64, sampler=sampler)   # ★ 用 sampler 時不能設 shuffle
```

> 另一個選擇是在 loss 裡加權：`nn.CrossEntropyLoss(weight=torch.tensor(class_w))`。
> 兩種都試試看哪個好。

---

## §5 transforms（資料增強）★

**資料增強是提升泛化能力最便宜的方法**，效果常常勝過換模型。

```python
from torchvision import transforms as T

train_tfm = T.Compose([
    T.RandomCrop(32, padding=4),                    # ★ CIFAR 標配
    T.RandomHorizontalFlip(p=0.5),                  # ★ CIFAR 標配
    T.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.1),
    T.RandomRotation(15),
    T.ToTensor(),                                   # PIL → (C,H,W) tensor 且 /255
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],      # CIFAR-10 的統計量
                std=[0.2470, 0.2435, 0.2616]),
    T.RandomErasing(p=0.25),                        # ★ 要放在 Normalize 之後
])

val_tfm = T.Compose([                               # ★ 驗證集「不做」隨機增強！
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])
```

> 🔥 **驗證集/測試集絕對不能做隨機增強**，否則每次評估結果都不一樣，數字沒有意義。
> （測試時增強 TTA 是另一回事，那是刻意多次推論再平均。）

### 常用增強的效果排序（CIFAR-10 經驗）

| 增強 | 大概提升 | 備註 |
|---|---|---|
| RandomCrop + Flip | +3~5% | ★ 必加，幾乎零成本 |
| Normalize | +2~4% | ★ 必加 |
| Cutout / RandomErasing | +1~2% | |
| Mixup / CutMix | +1~3% | 需要改 loss（見下） |
| AutoAugment / RandAugment | +1~2% | `T.RandAugment()` |

### Mixup（需要改 loss，但很有效）

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# 訓練時
x, ya, yb, lam = mixup(x, y)
logits = model(x)
loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
```

### 時序資料的增強（給你的股票專案）

```python
def jitter(x, sigma=0.03):          # 加高斯雜訊
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=0.1):          # 整體縮放
    return x * (1 + torch.randn(x.size(0), 1, x.size(2)) * sigma)

def window_warp(x):                 # 時間軸伸縮
    ...

def masking(x, p=0.1):              # 隨機遮蔽時間步（★ SSL 常用）
    mask = torch.rand(x.shape[:2], device=x.device) < p
    return x.masked_fill(mask[..., None], 0.0)
```

> ★ 這些正是 **W17-W18 Self-supervised** 主題會用到的增強。
> 對比學習的核心就是「同一筆資料做兩種不同增強，讓模型認出它們是同一個」。

---

## §6 正規化統計量怎麼算 ★

```python
def compute_mean_std(dataset, batch_size=256):
    """對整個訓練集算 per-channel 的 mean / std。★ 只能用訓練集！"""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    n_pixels = 0
    s = torch.zeros(3)
    s2 = torch.zeros(3)
    for x, _ in loader:                          # x: (B, 3, H, W)
        b = x.size(0) * x.size(2) * x.size(3)
        s  += x.sum(dim=[0, 2, 3])
        s2 += (x ** 2).sum(dim=[0, 2, 3])
        n_pixels += b
    mean = s / n_pixels
    std = (s2 / n_pixels - mean ** 2).sqrt()
    return mean, std
```

**常用資料集的統計量（直接抄）：**

```python
MNIST      = ([0.1307], [0.3081])
CIFAR10    = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
CIFAR100   = ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
ImageNet   = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ★ 用預訓練模型時必用這組
```

---

## §7 檢查你的資料管線（★ 訓練前必做）

```python
def sanity_check(loader, class_names=None):
    x, y = next(iter(loader))
    print(f"x: shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f}")
    print(f"y: shape={tuple(y.shape)} dtype={y.dtype} "
          f"unique={y.unique().tolist()}")
    assert not torch.isnan(x).any(), "輸入有 NaN！"
    print(f"batches={len(loader)}  samples={len(loader.dataset)}")
    # 類別分布
    import collections
    cnt = collections.Counter()
    for _, yy in loader:
        cnt.update(yy.tolist())
    print("類別分布:", dict(sorted(cnt.items())))
```

**檢查清單：**

- [ ] `x` 的數值範圍合理嗎？（Normalize 後應該大致在 -3 ~ 3）
- [ ] `y` 的 dtype 是 `int64` 嗎？值域是 `0 ~ C-1` 嗎？（不是 1~C！）
- [ ] batch size 對嗎？`drop_last` 有沒有意外丟掉太多資料？
- [ ] 隨便挑幾張圖出來**用眼睛看**（`show_batch`），標籤對得起來嗎？
- [ ] 訓練/驗證的 transform 有分開嗎？
- [ ] 時序資料是按時間切的嗎？

> 🔥 **「用眼睛看幾筆資料」是最被低估的除錯手段。**
> 我看過太多人 debug 三天，最後發現是標籤錯位一格。

---

## §8 效能優化

```python
# 1. 用 pin_memory + non_blocking
loader = DataLoader(ds, pin_memory=True, num_workers=4)
x = x.to(device, non_blocking=True)

# 2. 小資料集直接全部放 GPU，完全跳過 DataLoader
X = X.to(device); Y = Y.to(device)
perm = torch.randperm(len(X), device=device)
for i in range(0, len(X), BS):
    idx = perm[i:i+BS]
    xb, yb = X[idx], Y[idx]          # ★ 零 CPU-GPU 傳輸，超快

# 3. 測量瓶頸在哪
import time
t0 = time.perf_counter()
for x, y in loader: pass
print(f"純讀資料一個 epoch: {time.perf_counter()-t0:.1f}s")
# 如果這個時間接近整個 epoch 的時間 → 瓶頸在資料，加 num_workers 或改快取
```

---

## §9 動手練習

1. 把你 `utils.py` 的 `FileStockDataset` 改成本章的寫法（加 type hint、加 sanity check）。
2. 寫一個 `collate_fn` 處理不等長序列，並產生 padding mask。
3. 用 `WeightedRandomSampler` 處理你的 8 類漲跌幅不平衡問題。
4. 對 CIFAR-10 分別用「有/無資料增強」訓練，比較驗證準確率。
5. 實作三種時序增強（jitter / scaling / masking），畫圖看看長什麼樣。
6. 測量你的資料管線速度，判斷瓶頸是 GPU 還是資料讀取。

---

## ✅ 自我檢核

- [ ] 說出 `__getitem__` 回傳的是單樣本還是 batch
- [ ] 說出 `len(loader)` 和 `len(loader.dataset)` 的差別
- [ ] 說出為何驗證集不能做隨機增強
- [ ] 說出為何時序資料不能用 `random_split`
- [ ] 寫出自訂 `collate_fn` 處理不等長序列
- [ ] 說出 Windows 上 `num_workers>0` 為何要 main guard
- [ ] 列出訓練前的資料 sanity check 清單
