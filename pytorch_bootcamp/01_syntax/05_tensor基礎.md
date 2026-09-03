# 05 · PyTorch Tensor 基礎

> Tensor = NumPy ndarray + **GPU 支援** + **自動微分**。
> 讀完 `02_numpy.md` 後，這章有 80% 是複習，重點在**多出來的那 20%**。

---

## §1 建立 tensor

```python
import torch

# 從 Python / NumPy
torch.tensor([1, 2, 3])                    # int64
torch.tensor([1., 2., 3.])                 # float32 ★ PyTorch 預設 float32（NumPy 是 float64）
torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cuda")

# 常數
torch.zeros(2, 3)                          # ★ 直接傳數字，不用像 NumPy 包成 tuple
torch.ones(2, 3)
torch.full((2, 3), 7.0)
torch.empty(2, 3)                          # 未初始化（最快）
torch.eye(3)
torch.arange(0, 10, 2)                     # [0,2,4,6,8]
torch.linspace(0, 1, 5)

# _like 系列（★ 超常用，自動繼承 dtype 和 device）
torch.zeros_like(x)
torch.ones_like(x)
torch.randn_like(x)
torch.full_like(x, 0.5)

# 隨機
torch.rand(2, 3)                           # [0,1) 均勻
torch.randn(2, 3)                          # 標準常態 ★ 初始化、GAN 的 noise 都用它
torch.randint(0, 10, (2, 3))
torch.randperm(10)                         # 0~9 隨機排列
torch.multinomial(probs, num_samples=1)    # ★ 依機率抽樣（RL 選動作、文字生成用）
torch.bernoulli(torch.full((3,), 0.7))
```

### 為什麼 `_like` 這麼重要

```python
# ❌ 這樣寫在 GPU 上會爆
noise = torch.randn(x.shape)               # 建在 CPU！
y = x + noise                              # RuntimeError: 兩個 tensor 不在同一裝置

# ✅
noise = torch.randn_like(x)                # 自動跟 x 同 device、同 dtype
```

---

## §2 dtype 與型別轉換

| dtype | 別名 | 用途 |
|---|---|---|
| `torch.float32` | `torch.float` | ★ 預設，權重與特徵 |
| `torch.float64` | `torch.double` | 很少用，GPU 上極慢 |
| `torch.float16` | `torch.half` | AMP 混合精度 |
| `torch.bfloat16` | — | ★ AMP 首選（動態範圍大，較不會 overflow） |
| `torch.int64` | `torch.long` | ★ 分類標籤、索引 |
| `torch.uint8` | — | 原始圖片像素 |
| `torch.bool` | — | mask |

```python
x.float()      # → float32
x.long()       # → int64  ★ 標籤轉型最常用
x.bool()
x.to(torch.float32)                          # 通用寫法
x.to(dtype=torch.float32, device="cuda")     # ★ 一次搞定
x.type_as(y)                                 # 轉成跟 y 一樣的型別
```

**三個最常見的 dtype 錯誤：**

```python
# 1. 從 NumPy 帶進 float64
x = torch.from_numpy(np_array)                       # float64！
# RuntimeError: expected scalar type Float but found Double
x = torch.from_numpy(np_array.astype(np.float32))    # ✅

# 2. 分類標籤不是 long
loss = F.cross_entropy(logits, labels.float())       # ❌
loss = F.cross_entropy(logits, labels.long())        # ✅

# 3. 圖片沒有正規化
img = torch.from_numpy(np_img)                       # uint8, 0~255
img = img.float() / 255.0                            # ✅
```

---

## §3 device 管理 ★

```python
# 標準寫法（放在每個檔案最上面）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device)                # ★ 回傳新 tensor，不會就地修改
model = model.to(device)        # ★ Module 是就地修改（也會回傳 self）

x.device                        # 印出所在裝置
x.is_cuda                       # True / False
x.cpu()                         # 搬回 CPU
```

> 🔥 **最常見的錯誤 #1**
> ```python
> model.to(device)     # ✅ Module 就地搬移，這樣寫可以
> x.to(device)         # ❌ Tensor 不會就地搬移！必須 x = x.to(device)
> ```

訓練迴圈的標準搬移寫法：

```python
for x, y in loader:
    x = x.to(device, non_blocking=True)     # non_blocking 搭配 pin_memory 可加速
    y = y.to(device, non_blocking=True)
```

---

## §4 基本運算（跟 NumPy 幾乎相同）

```python
a + b, a - b, a * b, a / b        # 逐元素
a @ b                             # 矩陣乘法
a ** 2, a.sqrt(), a.exp(), a.log()
a.abs(), a.sign(), a.clamp(0, 1)  # ★ clamp 對應 np.clip
torch.maximum(a, torch.zeros_like(a))     # = ReLU

# 就地運算（結尾底線 _）★ PyTorch 特有的命名慣例
a.add_(1)          # 就地加
a.zero_()          # 就地清零
a.clamp_(0, 1)
a.normal_(0, 0.02) # 就地填入常態亂數（權重初始化常用）
```

> ⚠️ **就地運算會破壞 autograd**：
> 若某 tensor 的值在 backward 之前被就地改掉，PyTorch 會報
> `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`。
> **在 forward 裡盡量不要用 `_` 結尾的運算**（`nn.ReLU(inplace=True)` 是少數安全的例外）。

### 歸約 reduction

```python
x.sum(), x.mean(), x.std(), x.var()
x.sum(dim=0)                      # ★ PyTorch 用 dim，NumPy 用 axis（意思一樣）
x.sum(dim=(0, 1), keepdim=True)   # ★ 注意是 keepdim（NumPy 是 keepdims，多一個 s）
torch.linalg.norm(x, ord=2)

# ★ max/min 回傳 (values, indices)，跟 NumPy 不同！
values, indices = x.max(dim=1)
x.argmax(dim=1)                   # 只要索引
x.amax(dim=1)                     # 只要值

# topk（top-5 accuracy 用）
values, indices = x.topk(5, dim=1)
```

---

## §5 比較、遮罩與選取

```python
x > 0                             # bool tensor
(x > 0).sum()                     # 有幾個正數
(x > 0).float().mean()            # 正數比例
torch.where(x > 0, x, 0.01 * x)   # LeakyReLU

x[x > 0]                          # 取出（回傳 1D）
x[x > 0] = 0                      # 就地賦值

# masked_fill ★ Transformer 的 attention mask 就靠它
scores = scores.masked_fill(mask, float("-inf"))

# gather ★ 從每列取出指定索引的值（RL 的 Q-learning 必用）
q = torch.randn(4, 6)                          # (batch, n_actions)
actions = torch.tensor([[2], [0], [5], [1]])   # (batch, 1)
q_taken = q.gather(1, actions)                 # (4, 1) 取出各自動作的 Q 值

# one-hot
import torch.nn.functional as F
onehot = F.one_hot(labels, num_classes=10).float()
```

---

## §6 tensor 轉成其他型別

```python
loss.item()                       # ★ 單一元素 tensor → Python 數字
t.tolist()                        # 轉巢狀 list
t.detach().cpu().numpy()          # ★ 轉 NumPy 的萬用寫法（背下來）

torch.from_numpy(arr)             # NumPy → tensor（共享記憶體）
torch.as_tensor(arr)              # 盡量共享
torch.tensor(arr)                 # 一定複製
```

> 🔥 **記憶體洩漏頭號原因**
> ```python
> total_loss += loss              # ❌ loss 帶著整張計算圖，累積到 OOM
> total_loss += loss.item()       # ✅
> ```

---

## §7 隨機性與可重現

```python
torch.manual_seed(42)             # CPU + 所有 GPU
torch.cuda.manual_seed_all(42)

# 用獨立的 generator（推薦，不污染全域狀態）
g = torch.Generator().manual_seed(42)
torch.randn(3, generator=g)
DataLoader(ds, shuffle=True, generator=g)   # 讓資料順序也可重現

# 完全確定性（會變慢）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

---

## §8 檢查 tensor 健康狀況（debug 必備）

```python
def inspect(name, t):
    print(f"{name:<16} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
          f"grad={t.requires_grad} "
          f"min={t.min().item():.4g} max={t.max().item():.4g} "
          f"mean={t.float().mean().item():.4g} "
          f"nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()}")
```

**訓練出現 nan 時的排查順序：**

1. `torch.isnan(x).any()` —— 輸入資料本身有 nan？
2. 每一層輸出印一次 —— 從哪一層開始變 nan？
3. 檢查 `log()` / `sqrt()` / 除法的分母
4. 檢查 loss 公式（`log(0)`、`0/0`）
5. 降 learning rate 十倍再試
6. 加梯度裁剪 `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

```python
# 讓 PyTorch 在產生 nan 的那一刻就報錯並指出位置（debug 時打開，平常關掉，很慢）
torch.autograd.set_detect_anomaly(True)
```

---

## §9 動手練習

1. 建立 `(4, 3, 32, 32)` 的隨機 tensor，印出所有屬性。
2. 把它搬到 GPU，再轉成 NumPy（想想中間需要幾個步驟）。
3. 用 `gather` 從 `(8, 10)` 的 logits 取出每筆的正確類別分數。
4. 用 `masked_fill` 把上三角區域填成 `-inf`（causal mask 的雛形）。
5. 把 `inspect()` 放進 `03_code/00_common.py`。
6. 故意讓一個 tensor 變 nan（`torch.log(torch.tensor(0.))`），用 `torch.isnan` 抓到它。

---

## ✅ 自我檢核

- [ ] 說出 `x.to(device)` 和 `model.to(device)` 的行為差別
- [ ] 說出 PyTorch 和 NumPy 的預設 dtype 各是什麼，為何會踩坑
- [ ] 寫出 `x.detach().cpu().numpy()` 並解釋每一步為何需要
- [ ] 說出 `loss.item()` 為何不能寫成 `loss`
- [ ] 說出 `_` 結尾的就地運算有什麼風險
- [ ] 說出 `x.max(dim=1)` 回傳什麼（提示：兩個東西）
