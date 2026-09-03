# 08 · nn.Module 完整教學

> `nn.Module` 是 PyTorch 組織模型的方式。搞懂它，你就能讀懂任何開源實作。

---

## §1 最小骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p_drop: float = 0.2):
        super().__init__()                    # ★ 忘記這行 → 所有註冊機制失效
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):                      # x: (B, in_dim)
        x = self.fc1(x)                        # (B, hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        return self.fc2(x)                     # (B, out_dim)

model = MLP(784, 256, 10)
logits = model(torch.randn(32, 784))           # ★ 用 model(x)，不要 model.forward(x)
```

### `nn.Module` 幫你做了什麼

1. **自動註冊子模組與參數** → `model.parameters()` 能拿到全部
2. **`.to(device)`** 一次搬移所有參數與 buffer
3. **`.train()` / `.eval()`** 一次切換所有子模組的模式
4. **`state_dict()` / `load_state_dict()`** 存讀權重
5. **hook 機制** 讓你能攔截中間結果

---

## §2 三種「屬性」：parameter / buffer / 普通屬性 ★

```python
class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 4))     # ① 參數：會被訓練、會存檔
        self.register_buffer("running_mean", torch.zeros(4))   # ② buffer：不訓練、會存檔、會跟著 to(device)
        self.scale = 2.0                             # ③ 普通屬性：不訓練、不存檔、不搬移
        self.const = torch.ones(4)                   # ⚠️ 危險！tensor 但沒註冊
```

| 類型 | 有梯度 | 在 `state_dict` | 跟著 `.to(device)` | 例子 |
|---|---|---|---|---|
| `nn.Parameter` | ✅ | ✅ | ✅ | 權重、bias |
| buffer | ❌ | ✅ | ✅ | BN 的 running_mean、位置編碼 |
| 普通 tensor 屬性 | ❌ | ❌ | ❌ | ← **會造成 device 錯誤** |

> 🔥 **這是新手最常踩的 device bug**：
> ```python
> self.pe = torch.zeros(max_len, d_model)      # ❌ model.cuda() 後 pe 還在 CPU
> self.register_buffer("pe", torch.zeros(max_len, d_model))   # ✅
> ```
> 你的 `Transformer.py` 裡的 `self.register_buffer('pe', ...)` 就寫對了，很好。

```python
# 查看
list(model.parameters())            # 所有可訓練參數
list(model.named_parameters())      # (名稱, 參數)
list(model.buffers())
list(model.named_buffers())
list(model.children())              # 直接子模組
list(model.named_modules())         # ★ 遞迴列出所有模組（剪枝、量化時常用）
```

---

## §3 容器：Sequential / ModuleList / ModuleDict

```python
# Sequential：依序執行，適合線性堆疊
self.features = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
)
x = self.features(x)                # 直接呼叫

# 用 OrderedDict 給每層命名（debug 時好認）
from collections import OrderedDict
self.features = nn.Sequential(OrderedDict([
    ("conv1", nn.Conv2d(3, 32, 3, padding=1)),
    ("relu1", nn.ReLU()),
]))

# ModuleList：需要自訂 forward 邏輯時用（★ Transformer 堆層就用這個）
self.layers = nn.ModuleList([EncoderLayer(d) for _ in range(n_layers)])
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

# ModuleDict：依 key 選路徑
self.heads = nn.ModuleDict({"cls": nn.Linear(d, 10), "reg": nn.Linear(d, 1)})
out = self.heads["cls"](x)
```

> 🔥 **絕對不能用 Python 原生 list 裝模組**：
> ```python
> self.layers = [nn.Linear(10, 10) for _ in range(3)]      # ❌ 參數不會被註冊！
> #  → model.parameters() 拿不到，optimizer 不會更新，to(device) 也不會搬
> self.layers = nn.ModuleList([...])                       # ✅
> ```
> 這個 bug 的症狀是「模型看起來有在跑，但那幾層永遠不學習」，非常難發現。

---

## §4 常用層速查

### 全連接與正則化

```python
nn.Linear(in_features, out_features, bias=True)      # (…, in) → (…, out)
nn.Dropout(p=0.5)                                    # 訓練時隨機歸零並放大 1/(1-p)
nn.Dropout2d(p=0.2)                                  # 整個 channel 一起 drop
nn.Flatten(start_dim=1)
nn.Identity()                                        # ★ 佔位用，取代某層時很好用
```

### 卷積

```python
nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1)
nn.Conv1d(...)          # ★ 時序資料用，輸入 (B, C, L)
nn.ConvTranspose2d(...) # 上採樣（GAN 生成器、AE 解碼器用）
nn.MaxPool2d(2); nn.AvgPool2d(2)
nn.AdaptiveAvgPool2d((1, 1))    # ★ 不管輸入多大都輸出 1x1，讓模型能吃任意尺寸
```

### 正規化 ★ 選錯會嚴重影響效果

| 層 | 對哪些維度算統計量 | 適用 |
|---|---|---|
| `nn.BatchNorm1d/2d` | 跨 **batch** 內同一 channel | CNN、batch 夠大時 |
| `nn.LayerNorm` | 每個樣本的**特徵**維度 | ★ Transformer、NLP、小 batch |
| `nn.GroupNorm` | 每個樣本的 channel 分組 | batch 很小的 CNN |
| `nn.InstanceNorm2d` | 每個樣本每個 channel | 風格轉換、GAN |

```python
nn.BatchNorm2d(64)              # 輸入 (B, 64, H, W)
nn.LayerNorm(512)               # 輸入 (…, 512)，對最後一維正規化
nn.LayerNorm([512])             # 同上
nn.GroupNorm(8, 64)             # 64 channel 分 8 組
```

> ★ **BatchNorm 的 batch size 至少要 16 以上才穩定**。
> 你的 8GB 顯卡如果被迫用 batch=4，請改用 GroupNorm 或 LayerNorm。

### 激活函數

```python
nn.ReLU(inplace=True)     # 最常用，簡單快
nn.LeakyReLU(0.2)         # ★ GAN 的判別器標配（避免死亡 ReLU）
nn.GELU()                 # ★ Transformer 標配
nn.SiLU()                 # = Swish，現代 CNN 常用
nn.Sigmoid()              # 二分類輸出（但通常用 BCEWithLogitsLoss 就不用它）
nn.Tanh()                 # ★ GAN 生成器輸出（配合資料正規化到 [-1,1]）
nn.Softmax(dim=-1)        # ⚠️ 分類時通常不用！CrossEntropyLoss 內建了
```

### 序列模型

```python
nn.Embedding(num_embeddings, embedding_dim)     # 詞表 → 向量
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)   # ★ batch_first 一定要設
nn.GRU(...)
nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True, norm_first=True)
nn.TransformerEncoder(encoder_layer, num_layers)
```

> ⚠️ `batch_first=False` 是 PyTorch 的歷史預設（輸入是 `(L, B, D)`）。
> **一律明確寫 `batch_first=True`**，否則 shape 會錯得莫名其妙。

---

## §5 train() 和 eval() ★

```python
model.train()      # 訓練模式
model.eval()       # 評估模式
model.training     # 查詢目前狀態
```

**只影響兩種層，但影響巨大：**

| 層 | `train()` | `eval()` |
|---|---|---|
| `Dropout` | 隨機丟棄神經元 | **完全不丟棄** |
| `BatchNorm` | 用**當前 batch** 的統計量，同時更新 running stats | 用**累積的 running stats** |

```python
# ★ 標準流程，一個都不能少
for epoch in range(EPOCHS):
    model.train()                       # ← 訓練前
    for x, y in train_loader:
        ...

    model.eval()                        # ← 驗證前
    with torch.no_grad():               # ← 兩個都要！它們管的是不同的事
        for x, y in val_loader:
            ...
```

> 🔥 **`model.eval()` 和 `torch.no_grad()` 是兩件不同的事，不能互相取代**：
> - `model.eval()` 改變**層的行為**（dropout/BN）
> - `torch.no_grad()` 關閉**梯度追蹤**（省記憶體、加速）
>
> 忘記 `model.eval()` 的症狀：驗證準確率明顯偏低且每次都不一樣。
> 忘記 `model.train()` 的症狀：第二個 epoch 開始 dropout 失效，開始過擬合。

---

## §6 權重初始化

PyTorch 的預設初始化已經不錯，但**特定架構需要特定初始化**。

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)          # ★ apply 會遞迴套用到每個子模組
```

| 初始化 | 搭配 | 場景 |
|---|---|---|
| `kaiming_normal_` / `kaiming_uniform_` | ReLU / LeakyReLU | ★ CNN 預設選擇 |
| `xavier_normal_` / `xavier_uniform_` | tanh / sigmoid | RNN、舊架構 |
| `normal_(0, 0.02)` | — | ★ DCGAN 論文指定 |
| `trunc_normal_(std=0.02)` | — | ★ ViT / BERT 常用 |
| `zeros_` | — | bias、殘差分支的最後一層 |

```python
# DCGAN 的標準初始化（W19 會用到）
def dcgan_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
```

---

## §7 Hook：攔截中間結果

```python
# forward hook：拿中間層輸出（特徵圖視覺化、Grad-CAM、知識蒸餾）
feats = {}
def save_output(name):
    def fn(module, inp, out):
        feats[name] = out.detach()
    return fn

h = model.layer3.register_forward_hook(save_output("layer3"))
_ = model(x)
h.remove()                          # ★ 一定要 remove

# backward hook：拿梯度
def print_grad(module, grad_in, grad_out):
    print(module.__class__.__name__, grad_out[0].norm().item())
h = model.fc.register_full_backward_hook(print_grad)

# tensor 層級的 hook
h = feat.register_hook(lambda g: print("grad norm:", g.norm().item()))
```

**用途**：Grad-CAM、特徵蒸餾、debug 梯度流、模型剖析。

---

## §8 凍結與部分訓練（遷移學習 / SSL 必備）

```python
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 1. 凍結全部
for p in model.parameters():
    p.requires_grad_(False)

# 2. 換掉分類頭（新層預設 requires_grad=True）
model.fc = nn.Linear(model.fc.in_features, 10)

# 3. optimizer 只收要訓練的參數
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# 4. ★ 別忘了：凍結的 BatchNorm 在 train() 下仍會更新 running stats！
#    要完全凍結骨幹，還要把它保持在 eval 模式
model.train()
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
```

**分層學習率（fine-tune 的常見技巧）：**

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},   # 骨幹小一點
    {"params": model.fc.parameters(),       "lr": 1e-3},   # 新頭大一點
], weight_decay=0.01)
```

---

## §9 檢視模型

```python
print(model)                       # 印出結構（不含 shape）

# ★ 強烈推薦：torchinfo
from torchinfo import summary
summary(model, input_size=(1, 3, 32, 32))
# 會列出每層的 output shape、參數量、記憶體用量

# 參數量統計
total = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total {total:,} | trainable {train:,} | "
      f"size {total * 4 / 1e6:.2f} MB (float32)")
```

---

## §10 一個完整的範例模型（可直接拿去改）

```python
class ConvBlock(nn.Module):
    """Conv → BN → 激活，是 CNN 的基本積木。"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),   # ★ 後面接 BN 時 bias 可省
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))     # ★ 殘差連接


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.stem = ConvBlock(3, width)                     # (B,64,32,32)
        self.stage1 = nn.Sequential(ResidualBlock(width), nn.MaxPool2d(2))       # (B,64,16,16)
        self.stage2 = nn.Sequential(ConvBlock(width, width*2), ResidualBlock(width*2), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # (B, 128, 1, 1)  ★ 讓模型能吃任意尺寸
            nn.Flatten(1),                # (B, 128)
            nn.Dropout(0.2),
            nn.Linear(width*2, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x):                 # x: (B, 3, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.head(x)               # (B, num_classes)
```

> 為什麼 `bias=False`？因為後面接 BatchNorm，BN 自己有 `beta` 平移項，
> conv 的 bias 會被完全抵消，留著只是浪費參數。

---

## §11 動手練習

1. 建一個模型，故意用 Python list 裝 `nn.Linear`，印出 `model.parameters()` 的數量，
   再改成 `nn.ModuleList`，比較差異。
2. 用 `register_buffer` 實作位置編碼，驗證 `model.cuda()` 後 buffer 也在 GPU 上。
3. 訓練一個有 dropout 的模型，故意不呼叫 `model.eval()`，觀察驗證準確率的變化。
4. 用 `torchinfo.summary` 分析你的 `CNN.py`，找出參數量最大的那一層。
5. 用 forward hook 抓出 `Transformer.py` 第一層 CNN 的輸出並畫圖。
6. 把 `SmallResNet` 拿去跑 CIFAR-10，看能到幾趴。

---

## ✅ 自我檢核

- [ ] 說出 parameter / buffer / 普通屬性三者的差別
- [ ] 說出為何不能用 Python list 裝子模組，症狀是什麼
- [ ] 說出 `model.eval()` 和 `torch.no_grad()` 各自管什麼，為何兩個都要
- [ ] 說出 BatchNorm 和 LayerNorm 的統計量算法差在哪，各自適用場景
- [ ] 說出 conv 後面接 BN 時為何 `bias=False`
- [ ] 寫出「凍結骨幹只訓練分類頭」的完整程式碼（含 BN 的處理）
- [ ] 用 `model.apply()` 套用自訂初始化
