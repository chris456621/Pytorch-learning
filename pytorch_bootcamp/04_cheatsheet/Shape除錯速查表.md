# Shape 除錯速查表

> **你 80% 的 bug 是 shape 錯。** 訓練壞掉時先翻這一頁。

---

## 0. 三十秒定位法

```python
# 1. 印出兩邊
print(f"{pred.shape=}  {target.shape=}")
assert pred.shape == target.shape

# 2. 逐層看
from torchinfo import summary
summary(model, input_size=(1, 3, 32, 32))

# 3. 手動逐層
h = x
for name, layer in model.named_children():
    h = layer(h)
    print(f"{name:20s} {tuple(h.shape)}")
```

---

## 1. 錯誤訊息 → 原因 → 修法

| 錯誤訊息片段 | 原因 | 修法 |
|---|---|---|
| `Expected all tensors to be on the same device` | 忘記 `x = x.to(device)`；或 `__init__` 裡建了 tensor 但沒 `register_buffer` | 見 §2 |
| `expected scalar type Float but found Double` | NumPy 預設 float64 | `arr.astype(np.float32)` |
| `mat1 and mat2 shapes cannot be multiplied (32x400 and 512x10)` | `nn.Linear` 的 `in_features` 算錯 | 用 `torchinfo` 或印攤平後 shape |
| `view size is not compatible with input tensor's size and stride` | `permute`/`transpose` 後直接 `view` | 改 `reshape` 或先 `contiguous()` |
| `Expected 4-dimensional input ... got 3-dimensional` | 忘了 batch 維度 | `x[None]` |
| `Expected input batch_size (32) to match target batch_size (64)` | x 與 y 沒對齊 | 檢查 `__getitem__` / `collate_fn` |
| `Target size (torch.Size([32])) must be the same as input size (torch.Size([32, 1]))` | 迴歸 shape 沒對齊 | `pred.squeeze(-1)` |
| `The size of tensor a (32) must match ... at dimension 1` | 廣播失敗 | 印兩邊 shape |
| `index out of range in self` | Embedding 索引 ≥ `num_embeddings`，或標籤從 1 開始 | 檢查 `labels.min()/max()` |
| `Dimension out of range (expected to be in range of [-2, 1])` | `dim` 超過張量維度 | 印 `x.dim()` |
| `Expected more than 1 value per channel when training` | BatchNorm 遇到 batch=1 | `drop_last=True` 或改 GroupNorm |
| `one of the variables needed for gradient computation has been modified` | 就地運算破壞 autograd | `x = x + y` 取代 `x += y` |
| `Trying to backward through the graph a second time` | 圖被重用 | `hidden.detach()` 或合併 loss |
| `element 0 of tensors does not require grad` | loss 沒接到任何可訓練參數 | 檢查 `no_grad` / `detach` / 凍結 |
| `grad can be implicitly created only for scalar outputs` | 對非純量 backward | `.mean()` 或 `.sum()` |

---

## 2. device 錯誤的三個藏身處

```python
# (1) tensor 不會就地搬移
x.to(device)          # ❌ 沒作用
x = x.to(device)      # ✅

# (2) __init__ 裡建的 tensor 沒註冊
self.pe = torch.zeros(L, D)                       # ❌ model.cuda() 不會搬它
self.register_buffer("pe", torch.zeros(L, D))     # ✅

# (3) 憑空建立的 tensor
noise = torch.randn(x.shape)      # ❌ 建在 CPU
noise = torch.randn_like(x)       # ✅
mask  = torch.ones(L, L)                          # ❌
mask  = torch.ones(L, L, device=x.device)         # ✅
```

---

## 3. 標準 shape 轉換（背下來）

```python
# CNN 特徵圖 -> Transformer 序列
x.flatten(2).transpose(1, 2)          # (B,C,H,W) -> (B, H*W, C)

# Transformer 序列 -> CNN 特徵圖
x.transpose(1, 2).reshape(B, C, H, W) # (B,L,C) -> (B,C,H,W)

# 時序資料 -> Conv1d
x.transpose(1, 2)                     # (B,L,F) -> (B,F,L)

# tensor -> matplotlib
img.permute(1, 2, 0)                  # (C,H,W) -> (H,W,C)

# 單張圖 -> 模型
x[None]                               # (C,H,W) -> (1,C,H,W)

# CNN -> 全連接
x.flatten(1)                          # (B,C,H,W) -> (B, C*H*W)
```

## Multi-Head Attention 的完整流（★ 一定要背）

```
x            (B, L, D)
q_proj       (B, L, D)
.view(B,L,h,d)      (B, L, h, d)
.transpose(1,2)     (B, h, L, d)      ← head 變成 batch 維
q @ k^T / sqrt(d)   (B, h, L, L)
softmax @ v         (B, h, L, d)
.transpose(1,2)     (B, L, h, d)
.reshape(B,L,D)     (B, L, D)         ← ★ 這裡要 reshape 不能 view
out_proj            (B, L, D)
```

---

## 4. 卷積尺寸計算

```
H_out = (H_in + 2*padding - dilation*(kernel-1) - 1) // stride + 1
沒有 dilation 時： H_out = (H + 2p - k) // s + 1
```

```python
def conv_out(h, k, s=1, p=0, d=1):
    return (h + 2*p - d*(k-1) - 1) // s + 1
```

| k | s | p | 32×32 → |
|---|---|---|---|
| 3 | 1 | 1 | 32×32 |
| 3 | 2 | 1 | 16×16 |
| 4 | 2 | 1 | 16×16 |
| 5 | 1 | 2 | 32×32 |
| 5 | 2 | 2 | 16×16 |
| 7 | 2 | 3 | 16×16 |
| 1 | 1 | 0 | 32×32（只改 channel） |

**感受野**：`RF += (k-1) * 累積stride`

```python
def receptive_field(layers):        # layers = [(k, s), ...]
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf
```

---

## 5. 靜默 shape bug（不報錯但結果全錯）★★

```python
# ① 迴歸任務的 (B,1) vs (B,)
loss = F.mse_loss(pred, y)          # pred(32,1) y(32,) -> 廣播成 (32,32)！
# 修：pred.squeeze(-1) 或 y.unsqueeze(-1)

# ② squeeze() 不指定 dim，batch=1 時多殺一維
out.squeeze()          # (1,10) -> (10,)  ❌
out.squeeze(-1)        # ✅ 指定 dim

# ③ 標籤與特徵錯位一格
# 檢查：直接印幾筆出來用眼睛看

# ④ CrossEntropyLoss 前多做了 softmax
return self.fc(x)                   # ✅ 回傳 logits

# ⑤ 序列任務 logits 的 reshape
F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1))   # (B,L,C),(B,L)
F.cross_entropy(logits.transpose(1, 2), labels)              # 等價寫法

# ⑥ Python list 裝子模組
self.layers = [nn.Linear(...)]      # ❌ 那些層永遠不會被訓練
self.layers = nn.ModuleList([...])  # ✅
```

**防禦寫法（加在 forward 和算 loss 之前）：**

```python
assert x.dim() == 4, f"expected 4D, got {tuple(x.shape)}"
assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
```

---

## 6. 訓練不動時的 12 步檢查

```
1.  用眼睛看幾筆資料和標籤       ← 最被低估的一步
2.  labels 是 int64 且值域 0~C-1？
3.  輸入正規化後 mean~0 std~1？
4.  ★ 32 筆過擬合測試：loss 能降到 ~0 嗎？
      不能 -> 流程有 bug，別調參
5.  model.train() / model.eval() 都有嗎？
6.  optimizer.zero_grad() 有嗎？
7.  loss.backward() + optimizer.step() 都有嗎？
8.  pred.shape == target.shape？
9.  最後一層有沒有多餘的 softmax / sigmoid？
10. lr 掃 1e-2 ~ 1e-4 試過嗎？
11. grad_stats()：有 None 嗎？norm 是 1e-8 或 1e4 嗎？
12. 參數真的在變嗎？
      before = p.clone(); ...一步...; print((p-before).abs().sum())
```

---

## 7. 常用維度符號

| 符號 | 意義 | 典型位置 |
|---|---|---|
| `B` | batch size | 永遠是第 0 維 |
| `C` | channels | CNN 第 1 維 |
| `H`,`W` | 高、寬 | CNN 第 2、3 維 |
| `L`/`T` | 序列長度 | Transformer 第 1 維 |
| `D` | 特徵維度 | Transformer 第 2 維 |
| `h` | attention head 數 | |
| `d` | 每 head 維度 = `D // h` | |

**寫註解的習慣：**

```python
def forward(self, x):
    # x: (B, C, H, W)
    x = self.conv(x)       # (B, 64, H, W)
    x = self.pool(x)       # (B, 64, H//2, W//2)
    x = x.flatten(1)       # (B, 64*H*W//4)
    return self.fc(x)      # (B, num_classes)
```
