# 06 · Shape 操作大全 ★★★

> **這章是整份教材最高投資報酬率的一章。**
> 你未來 80% 的 bug 是 shape 錯，而其中 90% 是因為搞不清楚
> `view` / `reshape` / `permute` / `transpose` / broadcasting 的差別。

---

## §0 先建立習慣：永遠標註 shape

```python
def forward(self, x):
    # x: (B, C, H, W)
    x = self.conv(x)        # (B, 64, H, W)
    x = self.pool(x)        # (B, 64, H//2, W//2)
    x = x.flatten(1)        # (B, 64*H*W//4)
    x = self.fc(x)          # (B, num_classes)
    return x
```

**常用符號約定**（整份教材與論文通用）：

| 符號 | 意義 |
|---|---|
| `B` | batch size |
| `C` | channels |
| `H`, `W` | height, width |
| `L` / `T` / `S` | sequence length（時間步） |
| `D` / `d_model` | feature dimension |
| `h` / `nhead` | attention head 數 |
| `N` | 樣本總數 |

---

## §1 記憶體佈局：為什麼會有 view 和 reshape 兩個東西

Tensor 在記憶體裡其實是**一條連續的一維陣列** + 一組 **stride（步長）**。

```python
x = torch.arange(12).reshape(3, 4)
x.shape        # (3, 4)
x.stride()     # (4, 1)  ← 沿 dim0 走一格要跳 4 個元素，沿 dim1 走一格跳 1 個
x.is_contiguous()   # True
```

`permute` / `transpose` **不搬動任何資料**，只是改 stride：

```python
y = x.t()               # 轉置
y.shape                 # (4, 3)
y.stride()              # (1, 4)  ← stride 變了，但底層記憶體沒動
y.is_contiguous()       # False  ★ 不連續了！
```

於是：

```python
y.view(12)              # ❌ RuntimeError: view size is not compatible with
                        #    input tensor's size and stride
y.reshape(12)           # ✅ 會自動幫你複製一份連續記憶體
y.contiguous().view(12) # ✅ 手動轉連續再 view（跟上面等價）
```

### 結論表 ★★

| 操作 | 會不會複製資料 | 需要連續嗎 | 什麼時候用 |
|---|---|---|---|
| `view(...)` | 不會（一定是 view） | **需要** | 確定連續時，最快 |
| `reshape(...)` | 可能會 | 不需要 | ★ **不確定時用這個，最安全** |
| `permute(...)` | 不會 | 不需要 | 重排「多個」維度順序 |
| `transpose(a,b)` | 不會 | 不需要 | 只交換「兩個」維度 |
| `contiguous()` | 需要時會 | — | 在 permute 之後、view 之前 |
| `flatten(s,e)` | 可能會 | 不需要 | 攤平連續幾個維度 |
| `squeeze/unsqueeze` | 不會 | 不需要 | 增減長度 1 的維度 |

> 🔥 **黃金法則**：
> **`permute` 之後如果要改形狀，用 `reshape`，不要用 `view`。**
> 或寫成 `.permute(...).contiguous().view(...)`。

---

## §2 改形狀

```python
x = torch.arange(24)

x.view(2, 3, 4)
x.view(2, -1)              # -1 自動推算 → (2, 12)
x.reshape(2, 3, 4)
x.reshape(-1)              # 攤平

x = torch.randn(8, 3, 32, 32)
x.flatten()                # (24576,)      全部攤平
x.flatten(1)               # (8, 3072)     ★ 從 dim1 開始攤平（CNN → FC 的標準寫法）
x.flatten(1, 2)            # (8, 96, 32)   攤平 dim1 到 dim2
x.unflatten(1, (1, 3))     # (8, 1, 3, 32, 32)  反向操作
```

> ★ `nn.Flatten()` 預設就是 `flatten(1)`，保留 batch 維度。

---

## §3 換維度順序

```python
x = torch.randn(8, 3, 32, 32)     # (B, C, H, W)

x.permute(0, 2, 3, 1)             # (8, 32, 32, 3)  → (B, H, W, C)  給 matplotlib
x.transpose(1, 2)                 # (8, 32, 3, 32)  只換 dim1 和 dim2
x.movedim(1, -1)                  # (8, 32, 32, 3)  把 dim1 移到最後
```

**你一定會用到的四個轉換：**

```python
# 1. CNN 特徵圖 → Transformer 序列   (B,C,H,W) → (B, H*W, C)
x = x.flatten(2)                  # (B, C, H*W)
x = x.transpose(1, 2)             # (B, H*W, C)
# 或一行：x.flatten(2).transpose(1, 2)

# 2. Transformer 序列 → CNN 特徵圖   (B, L, C) → (B, C, H, W)
x = x.transpose(1, 2).reshape(B, C, H, W)

# 3. 給 Conv1d 用（時序資料）        (B, L, F) → (B, F, L)
x = x.transpose(1, 2)             # ★ Conv1d 吃 (B, channels, length)
# 這正是你 Transformer.py 裡 CNN 前面要做的事

# 4. 顯示圖片                        (C, H, W) → (H, W, C)
img.permute(1, 2, 0)
```

---

## §4 增減維度

```python
x = torch.randn(5)

x.unsqueeze(0)          # (1, 5)
x.unsqueeze(1)          # (5, 1)
x.unsqueeze(-1)         # (5, 1)
x[None, :]              # (1, 5)   ← 等價寫法，比較短
x[:, None]              # (5, 1)
x[..., None]            # (5, 1)

y = torch.randn(1, 5, 1)
y.squeeze()             # (5,)      移除所有長度 1 的維度
y.squeeze(0)            # (5, 1)    只移除 dim0
y.squeeze(-1)           # (1, 5)
```

> ⚠️ **`squeeze()` 不加參數很危險**：
> batch size 剛好是 1 時，`(1, 10)` 會被 squeeze 成 `(10,)`，
> 後面的 loss 計算就會 shape 錯。**永遠指定 dim**。

**最常見的使用場景：單張圖片要餵給模型**

```python
img = dataset[0][0]          # (3, 32, 32)
logits = model(img)          # ❌ 模型要 4D
logits = model(img[None])    # ✅ (1, 3, 32, 32)
pred = logits.argmax(1).item()
```

---

## §5 合併與分割

```python
a = torch.zeros(2, 3); b = torch.ones(2, 3)

torch.cat([a, b], dim=0)      # (4, 3)     ★ 在既有維度上接（不增加維度數）
torch.cat([a, b], dim=1)      # (2, 6)
torch.stack([a, b], dim=0)    # (2, 2, 3)  ★ 新增一個維度再堆疊
torch.stack([a, b], dim=-1)   # (2, 3, 2)

x = torch.randn(10, 6)
x.chunk(3, dim=1)             # 切成 3 塊，各 (10, 2)
x.split(2, dim=1)             # 每塊大小 2
x.split([1, 2, 3], dim=1)     # 指定各塊大小
torch.unbind(x, dim=0)        # 拆成 10 個 (6,) 的 tuple
```

> ★ **記法**：`cat` 不增加維度數；`stack` 一定多一維。
> DataLoader 的預設 `collate_fn` 就是用 `torch.stack` 把樣本疊成 batch。

**QKV 一次算完再切開（Transformer 的常見寫法）：**

```python
qkv = self.qkv_proj(x)                 # (B, L, 3*D)
q, k, v = qkv.chunk(3, dim=-1)         # 各 (B, L, D)
```

---

## §6 複製與擴張

```python
x = torch.randn(1, 3)

x.expand(4, 3)          # (4, 3)  ★ 不複製記憶體！只是假裝有 4 份（唯讀）
x.repeat(4, 1)          # (4, 3)  ★ 真的複製 4 份（佔記憶體，可寫入）
x.repeat_interleave(2, dim=0)     # 每個元素重複 2 次（不是整塊重複）

# expand 只能對「長度為 1」的維度擴張
torch.randn(2, 3).expand(4, 3)    # ❌ RuntimeError
```

> ★ **優先用 `expand`**（省記憶體）。只有在需要修改結果時才用 `repeat`。
> `expand` 出來的 tensor 不能做就地修改。

```python
# 實戰：把 (B, D) 的條件向量廣播到 (B, L, D)
cond = cond[:, None, :].expand(B, L, D)
```

---

## §7 Broadcasting 廣播 ★★★

規則跟 NumPy **完全一樣**（從右往左對齊，維度要相等或其中一個是 1）：

```python
torch.randn(3, 1) + torch.randn(1, 4)      # (3, 4)
torch.randn(8, 3, 32, 32) + torch.randn(3, 1, 1)   # (8,3,32,32) 對每個 channel 加不同值
torch.randn(8, 10) * torch.randn(10)       # (8, 10)
```

**實戰：正規化**

```python
x = torch.randn(32, 3, 64, 64)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x = (x - mean) / std                       # 廣播到每個 channel
```

### 🔥 靜默廣播災難（最陰險的 bug）

```python
pred = model(x)                # (32, 1)
target = y                     # (32,)
loss = F.mse_loss(pred, target)
# ⚠️ 會廣播成 (32, 32) 再算 loss！不報錯，但 loss 完全是錯的
# 症狀：loss 降不下來、或降到一個奇怪的平台就不動了

# ✅ 修正
loss = F.mse_loss(pred.squeeze(-1), target)
# 或
loss = F.mse_loss(pred, target.unsqueeze(-1))
```

**防禦寫法（強烈建議加在算 loss 之前）：**

```python
assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
```

> PyTorch 其實會對這種情況發出 `UserWarning`，但很容易被大量輸出淹沒。
> 養成寫 assert 的習慣，可以省下無數小時。

---

## §8 einsum ★★（Transformer 必備）

規則：**出現在輸入但不在輸出的下標 → 對它求和**。

```python
torch.einsum("ij,jk->ik", A, B)          # 矩陣乘法
torch.einsum("ij->ji", A)                # 轉置
torch.einsum("ii->i", M)                 # 對角線
torch.einsum("ij->j", A)                 # 沿 dim0 求和
torch.einsum("bij,bjk->bik", X, Y)       # batch 矩陣乘法
torch.einsum("bi,bi->b", a, b)           # batch 內積
torch.einsum("bl,bld->bd", w, v)         # 加權平均（w 是權重，v 是向量）
```

**完整的 attention（用 einsum 寫最清楚）：**

```python
# Q, K, V: (B, h, L, d)
scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / d ** 0.5      # (B, h, Lq, Lk)
attn   = scores.softmax(dim=-1)
out    = torch.einsum("bhqk,bhkd->bhqd", attn, V)              # (B, h, Lq, d)
```

對照不用 einsum 的寫法：

```python
scores = (Q @ K.transpose(-2, -1)) / d ** 0.5
out    = attn @ V
```

> 兩種都要會。`@` 比較快也比較常見；`einsum` 在維度多的時候可讀性壓倒性勝出。

---

## §9 Multi-Head Attention 的完整 shape 流（背下來）★

```python
B, L, D = x.shape          # (batch, seq_len, d_model)
h = nhead                  # head 數
d = D // h                 # 每個 head 的維度

# 1. 線性投影
q = self.q_proj(x)                       # (B, L, D)

# 2. 拆成多頭  ★ 這兩步是關鍵
q = q.view(B, L, h, d)                   # (B, L, h, d)
q = q.transpose(1, 2)                    # (B, h, L, d)   ← 讓 head 變成 batch 維度

# 3. 算 attention
scores = q @ k.transpose(-2, -1) / d**0.5    # (B, h, L, L)
attn = scores.softmax(-1)
out = attn @ v                               # (B, h, L, d)

# 4. 合併多頭  ★ 這裡一定要 contiguous 或用 reshape
out = out.transpose(1, 2)                    # (B, L, h, d)
out = out.contiguous().view(B, L, D)         # (B, L, D)
# 或  out = out.transpose(1, 2).reshape(B, L, D)

# 5. 輸出投影
out = self.out_proj(out)                     # (B, L, D)
```

> 🔥 **步驟 4 的 `contiguous()` 是新手最常漏掉的一行**，
> 少了它會得到 `view size is not compatible with input tensor's size and stride`。

---

## §10 常見 shape 錯誤與修法對照表

| 錯誤訊息 | 原因 | 修法 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (32x400 and 512x10)` | Linear 的 `in_features` 寫錯 | 用 `torchinfo.summary` 看實際攤平後的維度 |
| `view size is not compatible ... stride` | permute/transpose 後直接 view | 改用 `reshape` 或先 `contiguous()` |
| `Expected 4-dimensional input ... but got 3-dimensional` | 忘記 batch 維度 | `x[None]` 或 `x.unsqueeze(0)` |
| `Expected input batch_size (32) to match target batch_size (64)` | 資料與標籤沒對齊 | 檢查 Dataset 的 `__getitem__` |
| `The size of tensor a (32) must match ... at dimension 1` | 廣播失敗 | 印出兩邊 shape 對照 |
| `Target size (torch.Size([32])) must be the same as input size (torch.Size([32, 1]))` | 迴歸任務 shape 沒對齊 | `pred.squeeze(-1)` |
| `index out of range in self` | Embedding 的索引超過 `num_embeddings` | 檢查詞表大小 / 標籤範圍 |
| `IndexError: Dimension out of range` | `dim` 參數超過張量維度 | 印 `x.dim()` |

---

## §11 動手練習（★ 一定要動手，這章光看沒用）

1. 建 `(2,3,4)` tensor，用五種方法把它變成 `(6,4)`，說出哪些是 view 哪些是 copy。
2. 寫出 `(B,C,H,W) → (B, H*W, C)` 和反向的轉換。
3. 手刻 Multi-Head Attention，在每一步後面 `print(x.shape)` 驗證跟 §9 一致。
4. 故意製造靜默廣播 bug：`(32,1)` 對 `(32,)` 算 MSE，印出 loss 的 shape 觀察災難。
5. 用 `einsum` 實作 `(B,L,D) @ (D,D)` 並跟 `@` 比對結果是否相同。
6. 用 `expand` 和 `repeat` 各做一次 `(1,3) → (4,3)`，比較 `data_ptr()` 和記憶體用量。

<details>
<summary>第 1 題參考</summary>

```python
x = torch.arange(24).reshape(2, 3, 4)
x.view(6, 4)          # view（連續）
x.reshape(6, 4)       # view（因為本來就連續）
x.flatten(0, 1)       # view
x.contiguous().view(6, 4)          # view
x.permute(1, 0, 2).reshape(6, 4)   # ★ copy（permute 後不連續）
```
</details>

---

## ✅ 自我檢核

- [ ] 說出 `view` 和 `reshape` 的差別，以及什麼時候 `view` 會失敗
- [ ] 說出 `permute` 之後為何不能直接 `view`
- [ ] 說出 `cat` 和 `stack` 的差別
- [ ] 說出 `expand` 和 `repeat` 的差別
- [ ] 背出 Multi-Head Attention 的完整 shape 流
- [ ] 說出靜默廣播 bug 的症狀與防禦寫法
- [ ] 用 `einsum` 寫出 attention 分數
