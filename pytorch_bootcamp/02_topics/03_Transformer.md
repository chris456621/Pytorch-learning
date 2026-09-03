# 主題 03 · Transformer ★★

> **對應**：*Attention Is All You Need* · The Annotated Transformer · `03_code/05_transformer_scratch.py`
> 你已經寫過 `Transformer.py`，這章補上**數學細節、mask、現代改良**。
> 這是整份教材最需要花時間的主題（4 週）。


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 為什麼需要 Attention

**RNN 的兩個致命問題：**
1. **無法平行化**：`h_t` 必須等 `h_{t-1}` 算完，序列長度 = 計算深度
2. **長距離依賴衰減**：第 1 個 token 的資訊要經過 100 步才到第 100 個 token

**Attention 的解法**：讓每個位置**直接**跟所有位置連線，路徑長度永遠是 1，而且全部可以平行算。

代價是計算量 `O(L²)`。這就是為什麼長序列的 Transformer 很貴。

---

## §2 Scaled Dot-Product Attention ★

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**逐項理解**：
- `Q`（query）：「我想找什麼」
- `K`（key）：「我有什麼特徵」
- `V`（value）：「我實際提供的內容」
- `QKᵀ`：每個 query 跟每個 key 的相似度 → `(L_q, L_k)` 的分數矩陣
- `softmax`：把分數變成權重（每列加起來 = 1）
- `· V`：用權重對 value 做加權平均

### 為什麼要除以 √d_k ★（面試常考）

假設 `q`、`k` 的每個分量獨立、均值 0、變異數 1，
則 `q·k = Σᵢ qᵢkᵢ` 的變異數是 `d_k`，標準差是 `√d_k`。

`d_k = 64` 時，分數的範圍大約是 ±8 甚至更大。
softmax 對這種大數值會變得**極度尖銳**（幾乎變成 one-hot），
而 softmax 在飽和區的**梯度趨近於 0** → 訓練不動。

除以 `√d_k` 讓分數變異數回到 1，softmax 保持在有梯度的區域。

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """q,k,v: (B, h, L, d)   mask: (B, 1, Lq, Lk) 或可廣播，True = 要遮蔽"""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5        # (B, h, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))  # ★ 遮蔽處給 -inf
    attn = scores.softmax(dim=-1)                         # (B, h, Lq, Lk)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    out = attn @ v                                        # (B, h, Lq, d)
    return out, attn
```

> ★ PyTorch 2.x 有內建的高效版本（會自動選 FlashAttention）：
> ```python
> out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
> ```
> **學習時手刻，實務上用內建的**（更快、更省記憶體）。

---

## §3 Multi-Head Attention

**動機**：一組 QKV 只能學一種「關係」。多頭讓模型同時關注不同類型的關係
（語法關係、語意關係、指代關係……）。

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必須能被 nhead 整除"
        self.h = nhead
        self.d = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x):                 # (B, L, D) → (B, h, L, d)
        B, L, D = x.shape
        return x.view(B, L, self.h, self.d).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B, Lq, D = query.shape
        q = self._split(self.q_proj(query))      # (B, h, Lq, d)
        k = self._split(self.k_proj(key))        # (B, h, Lk, d)
        v = self._split(self.v_proj(value))      # (B, h, Lk, d)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask, self.dropout if self.training else 0.0)

        out = out.transpose(1, 2).reshape(B, Lq, D)   # ★ 合併多頭（要 reshape 或 contiguous）
        return self.out_proj(out), attn
```

### 逐行拆解 `_split`（新手 90% 卡在這裡）

```python
def _split(self, x):        # x: (B, L, D)
    B, L, _ = x.shape
    return x.view(B, L, self.h, self.d).transpose(1, 2)
```

**第 1 行** `B, L, _ = x.shape`
- `x.shape` 回傳 `torch.Size([2, 7, 64])`，可以直接解包成三個變數
- `_` 是慣例，代表「這個值我不需要」（第三維就是 D，我們用不到）

**第 2 行前半** `x.view(B, L, self.h, self.d)`
- 把最後一維 `D=64` **切成** `h=8` 份、每份 `d=8`
- `(2, 7, 64)` → `(2, 7, 8, 8)`
- **注意這裡沒有搬動任何資料**，只是重新解讀同一塊記憶體
- 因為 `h * d == D`（`8 * 8 == 64`），所以總元素數不變，`view` 才成立

**第 2 行後半** `.transpose(1, 2)`
- 交換第 1、2 維：`(2, 7, 8, 8)` → `(2, 8, 7, 8)`，也就是 `(B, h, L, d)`
- **為什麼要換？** 因為接下來的 `q @ k.transpose(-2,-1)` 只會對**最後兩維**做矩陣乘法，
  前面的維度全部視為「batch」。把 `h` 移到前面，8 個頭就能一次平行算完。

**合併回去時的對稱操作：**

```python
out = out.transpose(1, 2).reshape(B, Lq, D)
#         └ (B,h,L,d) -> (B,L,h,d)  └ (B,L,h,d) -> (B,L,D)
```

> 🔥 **這裡必須用 `reshape` 不能用 `view`**：`transpose` 之後記憶體不連續，
> `view` 會報 `view size is not compatible with input tensor's size and stride`。
> 想用 `view` 就得先 `.contiguous()`。差異詳見 `01_syntax/15_同性質語法差異比較.md` §1。

**完整 shape 流（背下來）：**

```
x            (B, L, D)
q_proj       (B, L, D)
view         (B, L, h, d)
transpose    (B, h, L, d)      ← head 變成 batch 維度
scores       (B, h, L, L)
attn @ v     (B, h, L, d)
transpose    (B, L, h, d)
reshape      (B, L, D)
out_proj     (B, L, D)
```

> ★ **注意**：多頭**不會**增加參數量或計算量（`h × d = D`）。
> 它只是把同樣的計算「切成幾份分頭做」，是免費的表達力提升。

---

## §4 三種 Attention

| 名稱 | Q 來源 | K, V 來源 | 用途 |
|---|---|---|---|
| **Self-Attention** | 自己 | 自己 | Encoder、Decoder 內部 |
| **Cross-Attention** | Decoder | Encoder 輸出 | ★ 翻譯任務中「看原文」 |
| **Masked Self-Attention** | 自己 | 自己（遮住未來） | ★ Decoder / GPT 自迴歸生成 |

---

## §5 Mask ★（新手最常搞錯的地方）

### 5.1 Padding Mask —— 遮住補零的位置

```python
def make_padding_mask(lengths, max_len):
    """lengths: (B,)  → mask: (B, 1, 1, max_len)，True 代表 padding 要遮掉"""
    idx = torch.arange(max_len, device=lengths.device)
    mask = idx[None, :] >= lengths[:, None]          # (B, L)
    return mask[:, None, None, :]                    # 廣播到 (B, h, Lq, Lk)
```

> **語法先備**：這一節會用到
> `torch.triu`（取上三角）、`masked_fill`（依遮罩填值）、
> `[None, None]`（加兩個維度用來廣播）、`|`（逐元素「或」，不是 `or`）。
> 不熟的話先看 `01_syntax/00_看不懂時先讀這裡.md` §13。

### 5.2 Causal Mask —— 遮住未來（自迴歸必備）

```python
def make_causal_mask(L, device):
    """上三角（不含對角線）為 True = 遮掉未來"""
    return torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device),   # 先做一個 L×L 全 True
        diagonal=1,          # ★ 1 = 不含對角線（0 會連自己也遮掉，模型就瞎了）
    )
    # dtype=torch.bool  -> masked_fill 需要 bool 遮罩，用 int 會報錯
    # device=device     -> ★ 不傳的話 mask 建在 CPU，跟 GPU 上的 scores 相加會噴
    #                        "Expected all tensors to be on the same device"

# 用法
mask = make_causal_mask(L, x.device)[None, None]     # (L,L) -> (1, 1, L, L)
#      加兩個長度 1 的維度，才能跟 scores 的 (B, h, L, L) 做廣播
```

```
L=4 的 causal mask（True = 遮蔽）：
        k0   k1   k2   k3
  q0    F    T    T    T      ← 位置 0 只能看自己
  q1    F    F    T    T
  q2    F    F    F    T
  q3    F    F    F    F      ← 位置 3 可以看全部
```

### 5.3 兩種 mask 合併

```python
mask = causal_mask | padding_mask        # ★ 用 or，任一要遮就遮
```

> 🔥 **PyTorch 內建模組的 mask 慣例混亂，一定要看清楚**：
> - `nn.MultiheadAttention` 的 `attn_mask`：**bool 時 True = 遮蔽**；float 時是「加上去的值」（用 `-inf`）
> - `key_padding_mask`：**True = 該位置是 padding，要遮蔽**
> - `nn.Transformer.generate_square_subsequent_mask(L)`：回傳 **float** mask（`-inf` 上三角）
>
> 搞錯的症狀：模型看得到未來 → 訓練 loss 低到不合理，但推論完全不能用。
> **驗證方法**：把某個未來位置的輸入改掉，檢查當前位置的輸出有沒有變。變了就是漏了。

```python
def verify_causal(model, x):
    """驗證因果性：改動未來的 token 不應影響現在的輸出。"""
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        x2 = x.clone(); x2[:, -1] = torch.randn_like(x2[:, -1])   # 改最後一個
        out2 = model(x2)
    diff = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
    print(f"前面位置的最大差異 = {diff:.2e}  （應該接近 0）")
```

---

## §6 位置編碼 ★

Self-attention 是**完全對稱**的 —— 打亂輸入順序，輸出也只是跟著打亂。
它本身不知道「順序」。所以必須額外注入位置資訊。

### 6.1 正弦位置編碼（原論文）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

```python
import math

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float)[:, None]        # (L, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))              # (d/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[None])                           # ★ buffer 不是 parameter
        
    def forward(self, x):              # x: (B, L, D)
        return x + self.pe[:, :x.size(1)]
```

**為什麼用 sin/cos？** 因為 `PE(pos+k)` 可以表示成 `PE(pos)` 的線性組合
（三角函數的和角公式），所以模型有機會學到**相對位置**關係。
另外它可以外推到訓練時沒見過的長度。

> ★ 你 `Transformer.py` 裡的實作是對的，而且正確使用了 `register_buffer`。

### 6.2 可學習位置編碼（BERT / ViT 用）

```python
self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
nn.init.trunc_normal_(self.pos_embed, std=0.02)
x = x + self.pos_embed[:, :L]
```

簡單有效，但**無法外推**到比訓練時更長的序列。

### 6.3 RoPE 旋轉位置編碼（現代 LLM 標配）

不是「加」上去，而是把 q、k 向量在複數平面上**旋轉**一個跟位置成正比的角度。
內積自然帶有相對位置資訊，且外推能力好。

```python
def build_rope_cache(L, d, device, base=10000.0):
    theta = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
    pos = torch.arange(L, device=device).float()
    freqs = torch.outer(pos, theta)          # (L, d/2)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """x: (B, h, L, d)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)
```

| 方式 | 外推 | 相對位置 | 用在 |
|---|---|---|---|
| 正弦 | ✅ | 間接 | 原始 Transformer |
| 可學習 | ❌ | ❌ | BERT、ViT |
| **RoPE** | ✅ | ✅ | ★ LLaMA、GPT-NeoX、多數現代 LLM |
| ALiBi | ✅ | ✅ | 部分長文本模型 |

---

## §7 完整的 Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, norm_first=True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),                       # ★ 現代用 GELU，原論文是 ReLU
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, mask=None):
        if self.norm_first:                  # ★ Pre-LN（推薦）
            a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.drop(a)
            x = x + self.drop(self.ff(self.norm2(x)))
        else:                                # Post-LN（原論文）
            a, _ = self.attn(x, x, x, mask)
            x = self.norm1(x + self.drop(a))
            x = self.norm2(x + self.drop(self.ff(x)))
        return x
```

### Pre-LN vs Post-LN ★

```
Post-LN（原論文）:  x = LayerNorm(x + Sublayer(x))
Pre-LN（現代）:     x = x + Sublayer(LayerNorm(x))
```

**Pre-LN 的優勢**：殘差路徑上沒有 LayerNorm，梯度可以無阻礙地直達淺層
→ **不需要 warmup 也能訓練**，深層時穩定得多。

**現在的實務**：一律用 Pre-LN（`norm_first=True`）。
`nn.TransformerEncoderLayer` 預設是 `norm_first=False`（Post-LN），**記得手動改**。

### FFN 為什麼要先放大再縮小

`dim_ff` 通常是 `4 × d_model`。
Attention 負責「混合不同位置的資訊」，FFN 負責「對每個位置做非線性變換」。
放大再縮小提供了足夠的表達能力。整個模型約 2/3 的參數在 FFN 裡。

---

## §8 完整 Encoder + 分類頭

```python
class TransformerClassifier(nn.Module):
    def __init__(self, n_feat, d_model=128, nhead=4, num_layers=4,
                 dim_ff=512, n_classes=2, max_len=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pe = SinusoidalPE(d_model, max_len)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))    # ★ 借用 BERT 的 [CLS]
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):        # x: (B, L, n_feat)
        B = x.size(0)
        x = self.input_proj(x)                                  # (B, L, D)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                          # (B, L+1, D)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x[:, 0])                               # ★ 用 [CLS] 的輸出分類
```

**三種池化方式（都可以試）：**

```python
out = x[:, 0]                    # ① [CLS] token
out = x.mean(dim=1)              # ② 平均池化（★ 簡單有效，時序任務常勝過 CLS）
out = x[:, -1]                   # ③ 最後一個時間步（適合預測「下一步」）
```

---

## §9 用 PyTorch 內建模組（實務首選）

```python
layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=4, dim_feedforward=512, dropout=0.1,
    activation="gelu",
    batch_first=True,        # ★★ 一定要設，否則輸入是 (L, B, D)
    norm_first=True,         # ★★ 用 Pre-LN
)
encoder = nn.TransformerEncoder(layer, num_layers=4, norm=nn.LayerNorm(128))

out = encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
```

> ★ **建議流程**：W11-W13 手刻理解原理 → W14 之後實務一律用內建的
>（內建版本用了 fused kernel 和 SDPA，快很多）。

---

## §10 Vision Transformer (ViT)

核心想法：**把圖片切成 patch，當作 token 序列**。

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_c=3, d_model=192):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2
        # ★ 用一個 stride=patch 的卷積就完成「切塊 + 線性投影」
        self.proj = nn.Conv2d(in_c, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):              # (B, 3, 32, 32)
        x = self.proj(x)               # (B, D, 8, 8)
        return x.flatten(2).transpose(1, 2)     # (B, 64, D)
```

> ★ **ViT 的關鍵限制**：它沒有 CNN 的局部性歸納偏置，
> 所以在小資料集（CIFAR-10 從零訓練）上通常**打不過 CNN**。
> 要靠大量資料預訓練或強力資料增強才能發揮。
> 你的 8GB 顯卡跑小型 ViT（d=192, 6 層）沒問題，但別期待它贏過 ResNet。

---

## §11 常見陷阱總整理

| 陷阱 | 症狀 | 解法 |
|---|---|---|
| 忘記 `batch_first=True` | shape 錯或結果亂七八糟 | 明確指定 |
| mask 語意搞反 | 訓練 loss 低到不合理 | 用 §5.3 的驗證函式 |
| 忘記除以 `√d_k` | 訓練很慢或不收斂 | 檢查公式 |
| 沒有 warmup（Post-LN） | 前幾步就 nan | 用 Pre-LN 或加 warmup |
| 沒有位置編碼 | 打亂輸入結果不變 | 加 PE，並實測驗證 |
| 沒有 `contiguous()` | view 報錯 | 用 `reshape` |
| `d_model % nhead != 0` | assert 失敗 | 檢查設定 |
| 序列太長爆記憶體 | OOM | `O(L²)` 是本質，降 L 或用 SDPA |

---

## §12 動手練習

1. 手刻 `scaled_dot_product_attention`，跟 `F.scaled_dot_product_attention` 比對結果。
2. 手刻 `MultiHeadAttention`，每步 `print(x.shape)` 驗證跟 §3 的 shape 流一致。
3. 實作 causal mask，用 §5.3 的方法**驗證因果性**。
4. 比較「有無位置編碼」：把輸入序列隨機打亂，看輸出變不變。
5. 比較 Pre-LN 和 Post-LN 在不加 warmup 時的訓練穩定性。
6. 把你的 `Transformer.py` 改成 Pre-LN + 用內建 `nn.TransformerEncoder`，比較準確率和速度。
7. 實作一個小 ViT 跑 CIFAR-10，跟 ResNet-18 比較（並解釋為何輸）。
8. 用 attention 權重畫熱力圖，看模型在時序上關注哪幾天。

---

## ✅ 自我檢核

- [ ] 解釋為何要除以 `√d_k`（要講到 softmax 梯度飽和）
- [ ] 不看筆記畫出 MHA 的完整 shape 流
- [ ] 寫出 causal mask 的產生程式碼並說明形狀
- [ ] 說出 padding mask 和 causal mask 的差別與合併方式
- [ ] 說出 Pre-LN 相對 Post-LN 的優勢與原因
- [ ] 說出為何 self-attention 需要位置編碼
- [ ] 說出正弦、可學習、RoPE 三種位置編碼的取捨
- [ ] 說出 ViT 在小資料集上輸給 CNN 的原因
