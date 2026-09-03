# 02 · NumPy 完整教學

> **為什麼要先學 NumPy？** PyTorch 的 tensor API 有 80% 跟 NumPy 一模一樣。
> 學會 NumPy 等於免費拿到 80% 的 PyTorch 語法。
> 而且資料前處理階段幾乎都在 NumPy / pandas 裡完成。

**【新版注意】你的環境是 NumPy 2.4.4（NumPy 2.x）。** 以下舊語法已被移除：

| 已移除（NumPy 1.x 的舊寫法） | 現在要寫 |
|---|---|
| `np.float_`, `np.complex_` | `np.float64`, `np.complex128` |
| `np.int0`, `np.uint0` | `np.intp`, `np.uintp` |
| `np.NaN`, `np.Inf`, `np.NAN` | `np.nan`, `np.inf` |
| `np.object`, `np.bool`, `np.str` | `object`, `bool`, `str` |
| `np.alltrue`, `np.sometrue` | `np.all`, `np.any` |
| `np.round_` | `np.round` |
| `arr.ptp()` | `np.ptp(arr)` |
| `np.in1d` | `np.isin` |

另外 NumPy 2.x 的 `repr` 會顯示成 `np.float64(3.0)` 而不是 `3.0`，這是正常的。

---

## §1 建立陣列

```python
import numpy as np

# 從 Python 物件
a = np.array([1, 2, 3])                    # shape (3,)   dtype int64
b = np.array([[1, 2], [3, 4]])             # shape (2, 2)
c = np.array([1, 2, 3], dtype=np.float32)  # ★ 深度學習幾乎都用 float32

# 常數陣列
np.zeros((2, 3))              # 全 0
np.ones((2, 3))               # 全 1
np.full((2, 3), 7.0)          # 全填 7
np.empty((2, 3))              # 不初始化（內容是垃圾值，最快）
np.eye(3)                     # 單位矩陣
np.zeros_like(a)              # 跟 a 同 shape 同 dtype 的全 0

# 數列
np.arange(0, 10, 2)           # [0 2 4 6 8]     像 range，含頭不含尾
np.linspace(0, 1, 5)          # [0. 0.25 0.5 0.75 1.]  ★含頭含尾，指定「幾個點」
np.logspace(-4, -1, 4)        # [1e-4 1e-3 1e-2 1e-1]  ★調 learning rate 掃描超好用

# 隨機（★ 現代寫法用 Generator，不要用 np.random.seed）
rng = np.random.default_rng(42)
rng.random((2, 3))            # [0,1) 均勻
rng.normal(0, 1, (2, 3))      # 常態分布
rng.integers(0, 10, (2, 3))   # 整數 [0,10)
rng.permutation(10)           # 0~9 隨機排列
rng.choice(100, size=10, replace=False)   # 不重複抽 10 個
```

> ⚠️ 舊教學的 `np.random.rand()` / `np.random.seed()` 仍可用但已不建議。
> `default_rng` 的優點：每個 rng 物件獨立，不會被別的函式庫偷偷改掉全域狀態。

---

## §2 屬性與 dtype

```python
a = np.zeros((2, 3, 4), dtype=np.float32)

a.shape       # (2, 3, 4)      ★ 最常看的
a.ndim        # 3              維度數
a.size        # 24             總元素數
a.dtype       # float32
a.itemsize    # 4              每個元素幾 bytes
a.nbytes      # 96             總共幾 bytes
a.T           # 轉置（只對 2D 直觀；高維請用 transpose）
```

### dtype 對照表（深度學習只需記這幾個）

| dtype | bytes | 用途 |
|---|---|---|
| `float32` | 4 | ★ 深度學習預設，權重/特徵 |
| `float64` | 8 | NumPy 預設，但 GPU 上很慢，**要記得轉成 float32** |
| `float16` | 2 | 混合精度訓練 |
| `int64` | 8 | ★ 分類標籤（PyTorch 的 CrossEntropyLoss 要求 int64） |
| `bool` | 1 | mask |
| `uint8` | 1 | 原始圖片像素 0~255 |

```python
a.astype(np.float32)          # 轉型（會複製）
a.astype(np.float32, copy=False)  # 已是該型別就不複製

# ★ 最常見的錯誤來源
img = np.array(pil_image)     # uint8, 0~255
img = img.astype(np.float32) / 255.0    # ✅ 一定要轉 float 並正規化
```

---

## §3 索引與切片

```python
a = np.arange(24).reshape(2, 3, 4)

a[0]              # 第 0 個「切片」，shape (3, 4)
a[0, 1, 2]        # 單一元素（比 a[0][1][2] 快）
a[:, 1, :]        # shape (2, 4)      ← 中間那維取第 1 個，該維消失
a[:, 1:2, :]      # shape (2, 1, 4)   ← 用切片就保留該維 ★重要區別
a[..., 0]         # 省略號：等於 a[:, :, 0]，shape (2, 3)
a[..., None]      # 在最後加一維，shape (2, 3, 4, 1)  ← 等同 np.newaxis
```

### 3.1 View vs Copy ★ 一定要搞懂

```python
a = np.arange(10)
b = a[2:5]          # ★ 基本切片 → view（共享記憶體！）
b[0] = 999
print(a)            # [0 1 999 3 4 5 6 7 8 9]  ← a 被改了

c = a[[2, 3, 4]]    # ★ fancy indexing → copy（獨立記憶體）
c[0] = -1
print(a[2])         # 999   ← a 不受影響

# 檢查是不是 view
b.base is a         # True  → b 是 a 的 view
c.base is None      # True  → c 是獨立的

# 想要獨立副本就明確 copy
b = a[2:5].copy()
```

> ★ **PyTorch 完全相同**：`x[2:5]` 是 view，`x[[2,3,4]]` 是 copy。
> 這解釋了為何有時候改一個 tensor 會神秘地影響另一個。

### 3.2 布林遮罩（boolean mask）

```python
a = np.array([1, -2, 3, -4, 5])

mask = a > 0                  # [True False True False True]
a[mask]                       # [1 3 5]        ← 取出來，回傳 copy
a[a > 0] = 0                  # 就地修改：把正數都設為 0
np.where(a > 0, a, 0)         # ★ 三元運算：正數保留，負數變 0（= ReLU）

# 多條件（★ 一定要用 & | ~ 且要加括號，不能用 and/or/not）
a[(a > 0) & (a < 5)]          # ✅
# a[a > 0 and a < 5]          # ❌ ValueError: truth value of array is ambiguous

# 統計
(a > 0).sum()                 # 有幾個正數（True 當 1）
(a > 0).any()                 # 有沒有任何一個正數
(a > 0).all()                 # 是不是全部都正
np.isnan(a).any()             # ★ 檢查 NaN，debug 必備
```

### 3.3 fancy indexing（整數陣列索引）

```python
a = np.arange(10) * 10        # [0 10 20 ... 90]
idx = np.array([3, 1, 7])
a[idx]                        # [30 10 70]     ← 可以重複、可以亂序

# 二維：對每一列取不同的欄
scores = np.array([[1, 5, 3],
                   [9, 2, 7]])
rows = np.arange(2)           # [0 1]
cols = np.array([1, 0])       # 第 0 列取第 1 欄、第 1 列取第 0 欄
scores[rows, cols]            # [5 9]

# ★ 這正是「取出每個樣本的正確類別分數」的做法（cross entropy 手刻時會用到）
logits = np.random.randn(4, 10)
labels = np.array([3, 7, 1, 0])
correct_logits = logits[np.arange(4), labels]     # shape (4,)
```

### 3.4 argmax / argsort / 排序

```python
a = np.array([3, 1, 4, 1, 5])

a.argmax()                    # 4     最大值的索引
a.argmin()                    # 1
np.argsort(a)                 # [1 3 0 2 4]  排序後的索引
a[np.argsort(a)]              # [1 1 3 4 5]  等同 np.sort(a)
np.argsort(a)[::-1]           # 由大到小的索引

# ★ 分類任務：從 logits 拿預測類別
logits = np.random.randn(32, 10)
preds = logits.argmax(axis=1)          # shape (32,)
top5  = np.argsort(logits, axis=1)[:, -5:][:, ::-1]   # top-5 預測

# 準確率
acc = (preds == labels).mean()
```

---

## §4 形狀操作 ★★★

```python
a = np.arange(12)

a.reshape(3, 4)               # (3, 4)
a.reshape(3, -1)              # -1 = 自動推算 → (3, 4)  ★超常用
a.reshape(-1)                 # 攤平成 1D
a.ravel()                     # 攤平（盡量回傳 view，快）
a.flatten()                   # 攤平（一定 copy）

x = np.zeros((2, 3, 4))
x.transpose(1, 0, 2)          # (3, 2, 4)   指定新的維度順序
x.swapaxes(0, 1)              # (3, 2, 4)   只交換兩個軸
np.moveaxis(x, 0, -1)         # (3, 4, 2)   把第 0 軸移到最後
```

### 4.1 增減維度（新手最常卡的地方）

```python
a = np.arange(5)              # (5,)

a[np.newaxis, :]              # (1, 5)     np.newaxis 就是 None
a[None, :]                    # (1, 5)     同上，寫法更短
a[:, None]                    # (5, 1)     ★ 變成「直的」
a.reshape(1, -1)              # (1, 5)
np.expand_dims(a, axis=0)     # (1, 5)

b = np.zeros((1, 5, 1))
b.squeeze()                   # (5,)       移除所有長度為 1 的維度
b.squeeze(axis=0)             # (5, 1)     只移除指定的
```

> ★ **實務場景**：模型只吃 batch 輸入，但你只有一張圖 →
> `x = img[None, ...]` 把 `(3,32,32)` 變成 `(1,3,32,32)`。

### 4.2 合併與分割

```python
a = np.zeros((2, 3)); b = np.ones((2, 3))

np.concatenate([a, b], axis=0)   # (4, 3)  ★ 在既有的軸上接起來
np.concatenate([a, b], axis=1)   # (2, 6)

np.stack([a, b], axis=0)         # (2, 2, 3)  ★ 新增一個軸再堆疊
np.stack([a, b], axis=-1)        # (2, 3, 2)

np.vstack([a, b])                # (4, 3)  = concatenate axis=0
np.hstack([a, b])                # (2, 6)  = concatenate axis=1

np.split(np.arange(9), 3)        # 切成 3 等份
np.array_split(np.arange(10), 3) # 不能整除也 OK
```

> ★ **`concatenate` vs `stack` 記法**：
> concatenate 不增加維度數；stack 一定會多一維。
> PyTorch 是 `torch.cat` 和 `torch.stack`，行為完全一樣。

---

## §5 Broadcasting 廣播 ★★★（本章最重要）

**規則（從右邊往左邊對齊）**：
1. 維度數不同時，在左邊補 1
2. 每個維度要嘛相等，要嘛其中一個是 1，否則報錯
3. 長度為 1 的維度會被「複製」到對應長度（實際上不真的複製記憶體）

```
    A      (3, 1)
    B      (1, 4)
    ------------
    結果    (3, 4)     ✅

    A      (256, 256, 3)
    B                (3,)     → 補成 (1, 1, 3)
    -----------------------
    結果    (256, 256, 3)     ✅

    A      (3, 4)
    B      (4, 3)
    ------------
    ❌ ValueError: operands could not be broadcast together
```

```python
a = np.arange(3).reshape(3, 1)     # [[0], [1], [2]]
b = np.arange(4).reshape(1, 4)     # [[0, 1, 2, 3]]
a + b                              # (3, 4) 的加法表

# ★ 實戰 1：對每個特徵做標準化（每欄減自己的平均）
X = rng.normal(size=(1000, 20))        # (N, D)
mu    = X.mean(axis=0)                 # (D,)
sigma = X.std(axis=0)                  # (D,)
X_norm = (X - mu) / (sigma + 1e-8)     # (1000,20) - (20,) → 自動廣播 ✅

# ★ 實戰 2：算所有點對點的歐氏距離（完全不用 for 迴圈）
A = rng.normal(size=(100, 3))
B = rng.normal(size=(50, 3))
diff = A[:, None, :] - B[None, :, :]   # (100,1,3) - (1,50,3) → (100,50,3)
dist = np.sqrt((diff ** 2).sum(axis=-1))   # (100, 50)

# ★ 實戰 3：one-hot 編碼（一行）
labels = np.array([2, 0, 1, 2])
onehot = (labels[:, None] == np.arange(3)[None, :]).astype(np.float32)
# (4,1) == (1,3) → (4,3)
```

**常見廣播錯誤：**

```python
y_true = np.zeros(100)         # (100,)
y_pred = np.zeros((100, 1))    # (100, 1)
loss = (y_true - y_pred) ** 2  # ★ 靜默廣播成 (100, 100)！loss 會完全錯但不報錯
# ✅ 修正：y_pred = y_pred.squeeze()  或  y_pred.reshape(-1)
```

> 🔥 **這是深度學習裡最陰險的 bug**：它不會報錯，只會讓 loss 莫名其妙不下降。
> 養成習慣：**算 loss 前印一次兩邊的 shape**。

---

## §6 軸（axis）的正確理解 ★

**口訣：`axis=k` 代表「把第 k 個維度消掉」。**

```python
a = np.arange(24).reshape(2, 3, 4)

a.sum()                # 純量，全部加總
a.sum(axis=0)          # (3, 4)      ← 第 0 維消失
a.sum(axis=1)          # (2, 4)      ← 第 1 維消失
a.sum(axis=-1)         # (2, 3)      ← 最後一維消失
a.sum(axis=(0, 1))     # (4,)        ← 消掉兩維
a.sum(axis=0, keepdims=True)   # (1, 3, 4)  ★ 保留維度，方便後續廣播
```

```python
# ★ keepdims 的經典用途：softmax
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)     # 減最大值防止 exp 溢位
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)  # 沒有 keepdims 這裡會 shape 錯
```

**常用歸約函式**（都支援 `axis` 和 `keepdims`）：

```python
a.sum()   a.mean()   a.std()   a.var()
a.min()   a.max()    a.argmin()  a.argmax()
a.prod()  a.cumsum()  a.cumprod()
np.median(a)   np.percentile(a, [25, 50, 75])
np.nanmean(a)  np.nansum(a)     # 忽略 NaN ★ 處理真實資料很需要
```

---

## §7 數學運算與線性代數

```python
a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

a + b, a - b, a * b, a / b     # ★ 全部都是「逐元素」運算
a ** 2, np.sqrt(a), np.exp(a), np.log(a + 1e-8)
np.abs(a), np.sign(a), np.clip(a, 0, 2)
np.maximum(a, 0)               # ← 這就是 ReLU
np.round(a), np.floor(a), np.ceil(a)
```

### 矩陣乘法 ★ 別跟逐元素乘搞混

```python
A = np.ones((3, 4)); B = np.ones((4, 5))

A * A          # ❌ 逐元素乘（shape 要一樣）
A @ B          # ✅ 矩陣乘法 → (3, 5)     ★ 現代寫法
np.matmul(A, B)  # 同上
np.dot(A, B)     # 2D 時同上，高維行為不同，建議一律用 @

# batch 矩陣乘法（廣播）
X = np.ones((32, 3, 4))
W = np.ones((4, 5))
X @ W          # (32, 3, 5)   ← W 自動廣播到每個 batch

X = np.ones((32, 3, 4))
Y = np.ones((32, 4, 5))
X @ Y          # (32, 3, 5)   ← 逐 batch 相乘
```

### einsum ★★（學會這個，attention 一行搞定）

```python
# einsum 的規則：出現在輸入但不在輸出的下標 → 對它求和
np.einsum('ij,jk->ik', A, B)          # 矩陣乘法
np.einsum('ij->ji', A)                # 轉置
np.einsum('ii->i', M)                 # 取對角線
np.einsum('ij->', A)                  # 全部加總
np.einsum('ij->j', A)                 # 沿 axis=0 加總
np.einsum('bij,bjk->bik', X, Y)       # batch 矩陣乘法
np.einsum('bi,bi->b', a, b)           # batch 內積

# ★ Attention 分數（Q @ K^T）
Q = rng.normal(size=(2, 8, 10, 64))   # (B, heads, L, d)
K = rng.normal(size=(2, 8, 10, 64))
scores = np.einsum('bhqd,bhkd->bhqk', Q, K)   # (2, 8, 10, 10)
# 比 Q @ K.transpose(0,1,3,2) 好讀太多
```

### 線性代數

```python
np.linalg.norm(a)              # L2 範數
np.linalg.norm(a, ord=1)       # L1
np.linalg.inv(M)               # 反矩陣
np.linalg.solve(A, b)          # 解 Ax=b（比 inv(A)@b 準且快）
np.linalg.eig(M)               # 特徵值分解
np.linalg.svd(M)               # SVD（低秩近似、壓縮會用到）
np.trace(M)                    # 跡
np.linalg.matrix_rank(M)
```

---

## §8 向量化：把 for 迴圈消滅掉

**原則：看到 for 迴圈在做數值運算，就想辦法用廣播消掉。** 速度差距通常 50～500 倍。

```python
import time
N = 1_000_000
x = rng.normal(size=N)

# ❌ 慢
t0 = time.perf_counter()
out = np.empty(N)
for i in range(N):
    out[i] = x[i] ** 2 + 3 * x[i]
print(f"loop: {time.perf_counter()-t0:.3f}s")     # ~0.5s

# ✅ 快
t0 = time.perf_counter()
out = x ** 2 + 3 * x
print(f"vec:  {time.perf_counter()-t0:.4f}s")     # ~0.005s
```

### 常見的向量化技巧

```python
# 1. 移動平均（滑動視窗）—— 用 cumsum
def moving_average(x, w):
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w

# 2. 滑動視窗切樣本（做時序模型的必備操作）★
from numpy.lib.stride_tricks import sliding_window_view
prices = np.arange(100.0)
windows = sliding_window_view(prices, window_shape=20)   # (81, 20) 零複製！
# 這正是你 Transformer.py 裡 WINDOW_SIZE=60 在做的事

# 3. 分組統計 —— np.bincount
labels = rng.integers(0, 5, 1000)
counts = np.bincount(labels, minlength=5)          # 每類幾筆
values = rng.normal(size=1000)
sums   = np.bincount(labels, weights=values, minlength=5)
means  = sums / counts                             # 每類平均

# 4. 條件賦值
np.where(x > 0, x, 0.01 * x)                       # LeakyReLU
np.select([x < -1, x < 1], [-1, x], default=1)     # 多條件

# 5. 排除迴圈算混淆矩陣
def confusion_matrix(y_true, y_pred, n_cls):
    return np.bincount(y_true * n_cls + y_pred,
                       minlength=n_cls**2).reshape(n_cls, n_cls)
```

---

## §9 與 PyTorch 的橋接 ★

```python
import torch

# NumPy → PyTorch
a = np.arange(6, dtype=np.float32).reshape(2, 3)
t1 = torch.from_numpy(a)       # ★ 共享記憶體！改 a 會影響 t1
t2 = torch.tensor(a)           # ★ 複製，獨立
t3 = torch.as_tensor(a)        # 盡量共享（已是 tensor 就不動）

a[0, 0] = 999
print(t1[0, 0])                # 999   ← 共享
print(t2[0, 0])                # 0     ← 獨立

# PyTorch → NumPy
b = t1.numpy()                 # 共享記憶體（tensor 必須在 CPU 上）
b = t1.cpu().numpy()           # GPU tensor 要先搬回 CPU
b = t1.detach().cpu().numpy()  # ★ 有梯度的 tensor 要先 detach（最常用的組合）
```

> 🔥 **背下這一行**：`x.detach().cpu().numpy()`
> —— 訓練中想把任何 tensor 拿出來畫圖/存檔，都是這一行。

**dtype 對應：**

| NumPy | PyTorch |
|---|---|
| `np.float32` | `torch.float32` / `torch.float` |
| `np.float64` | `torch.float64` / `torch.double` |
| `np.int64` | `torch.int64` / `torch.long` |
| `np.uint8` | `torch.uint8` |
| `bool` | `torch.bool` |

> ⚠️ NumPy 預設 `float64`，PyTorch 預設 `float32`。
> 從 NumPy 轉過來忘記轉型會得到 double tensor，跟模型的 float32 權重相乘會直接報錯：
> `RuntimeError: expected scalar type Float but found Double`
> **解法**：`torch.from_numpy(a.astype(np.float32))`

---

## §10 數值穩定性（做深度學習一定要知道）

```python
# ❌ 會溢位
x = np.array([1000., 1001., 1002.])
np.exp(x) / np.exp(x).sum()          # nan（exp(1000) = inf）

# ✅ 減掉最大值（數學上完全等價）
x_shift = x - x.max()
np.exp(x_shift) / np.exp(x_shift).sum()

# ❌ log(0) = -inf
np.log(p)                            # p 可能有 0

# ✅ 加 epsilon
np.log(p + 1e-12)
np.log(np.clip(p, 1e-12, 1.0))

# ❌ 除以 0
a / b

# ✅
a / (b + 1e-8)
np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
```

**檢查數值健康的三行（debug 必備）**：

```python
def check(name, x):
    print(f"{name:20s} shape={x.shape} dtype={x.dtype} "
          f"min={np.nanmin(x):.4g} max={np.nanmax(x):.4g} mean={np.nanmean(x):.4g} "
          f"nan={np.isnan(x).sum()} inf={np.isinf(x).sum()}")
```

---

## §11 存取與效能

```python
np.save("x.npy", a)                    # 單一陣列
a = np.load("x.npy")
np.savez("data.npz", X=X, y=y)         # 多個陣列
d = np.load("data.npz"); X, y = d["X"], d["y"]
np.savez_compressed("data.npz", X=X)   # 壓縮版

np.savetxt("a.csv", a, delimiter=",")  # 文字檔（慢，只適合小資料）
np.loadtxt("a.csv", delimiter=",")
```

**效能小抄：**

```python
# 1. 記憶體連續性影響速度
a.flags['C_CONTIGUOUS']        # 是否 row-major 連續
b = np.ascontiguousarray(a.T)  # 轉置後變不連續，需要時重排

# 2. 就地運算省記憶體
a += 1              # ✅ 就地
a = a + 1           # ❌ 產生新陣列
np.add(a, 1, out=a) # ✅ 明確就地

# 3. 預先配置好陣列，不要在迴圈裡 append
out = np.empty((n, d))          # ✅
for i in range(n): out[i] = f(i)

results = []                    # ❌ list append 再 np.array 會多一次複製
```

---

## §12 動手練習（用你的 `2330.csv`）

1. 讀入 `2330.csv` 的 `Volume` 欄，做 z-score 標準化，**不用任何迴圈**。
2. 用 `sliding_window_view` 把價格序列切成 `(N, 60)` 的訓練樣本。
3. 手刻 `softmax`，並驗證跟 `scipy.special.softmax` 結果一致。
4. 手刻 KNN：給 `X_train (1000, 20)`、`X_test (200, 20)`，用廣播算出 `(200, 1000)` 距離矩陣，取 k=5 投票，**全程禁用迴圈**。
5. 手刻 cross-entropy loss：給 `logits (N, C)` 和 `labels (N,)`，回傳純量 loss。

<details>
<summary>參考答案（第 4、5 題）</summary>

```python
# 4
d2 = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)   # (200, 1000)
idx = np.argsort(d2, axis=1)[:, :5]                              # (200, 5)
votes = y_train[idx]                                             # (200, 5)
pred = np.array([np.bincount(v).argmax() for v in votes])

# 5
def cross_entropy(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(logits).sum(axis=1))          # (N,)
    correct = logits[np.arange(len(labels)), labels]   # (N,)
    return (logZ - correct).mean()
```
</details>

---

## ✅ 自我檢核

- [ ] 說出 `a[1:3]` 和 `a[[1,2]]` 哪個是 view、哪個是 copy
- [ ] 不查資料判斷 `(3,1,5) + (4,1)` 的結果 shape（答案：`(3,4,5)`）
- [ ] 解釋 `axis=1` 在 `sum` 裡是「消掉哪一維」
- [ ] 說出 `keepdims=True` 什麼時候一定要加
- [ ] 寫出 `x.detach().cpu().numpy()` 並解釋每一步為何需要
- [ ] 用 `einsum` 寫出 batch attention 分數
- [ ] 解釋 `(100,) - (100,1)` 為何是危險的靜默 bug
