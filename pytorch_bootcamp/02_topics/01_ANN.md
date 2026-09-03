# 主題 01 · ANN 人工神經網路

> **對應**：李宏毅《機器學習》Lecture 1-5 · `03_code/03_ann_mnist.py`
> **前置**：`01_syntax/07_autograd.md`、`08_nn_Module.md`


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 為什麼需要非線性

一層線性層：`y = Wx + b`。堆兩層呢？

```
y = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = W'x + b'
```

**兩層線性 = 一層線性。** 不管堆幾層都一樣。
所以層與層之間**必須**放非線性激活函數，否則深度完全沒意義。

```python
# ❌ 這個「三層」網路其實等價於一層
nn.Sequential(nn.Linear(784, 256), nn.Linear(256, 128), nn.Linear(128, 10))

# ✅
nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
              nn.Linear(256, 128), nn.ReLU(),
              nn.Linear(128, 10))
```

**通用近似定理**：一個有足夠寬度的單隱藏層網路可以逼近任意連續函數。
但「足夠寬」可能是天文數字 —— **深度比寬度有效率得多**，這就是 deep learning 的 deep。

---

## §2 激活函數的選擇

| 函數 | 公式 | 優點 | 缺點 |
|---|---|---|---|
| Sigmoid | `1/(1+e^-x)` | 輸出 (0,1) | ★ 兩端梯度趨近 0 → 梯度消失 |
| Tanh | `(e^x-e^-x)/(e^x+e^-x)` | 零均值 | 一樣會飽和 |
| **ReLU** | `max(0,x)` | ★ 不飽和、計算快 | 負半邊梯度為 0 → 神經元死亡 |
| LeakyReLU | `max(0.01x, x)` | 解決死亡問題 | 多一個超參數 |
| **GELU** | `x·Φ(x)` | ★ Transformer 標配，平滑 | 稍慢 |
| SiLU/Swish | `x·σ(x)` | 現代 CNN 常用 | |

**梯度消失的數學根源**：sigmoid 的導數最大值是 0.25。
10 層網路連鎖律相乘 → `0.25^10 ≈ 1e-6`，最前面幾層完全學不到東西。
ReLU 在正半邊導數恆為 1，所以不會有這個問題。

```python
# 實驗：親手驗證梯度消失（強烈建議做一次）
for act in [nn.Sigmoid, nn.ReLU]:
    model = nn.Sequential(*[nn.Sequential(nn.Linear(64, 64), act()) for _ in range(10)])
    x = torch.randn(32, 64)
    model(x).sum().backward()
    g0 = model[0][0].weight.grad.norm().item()
    g9 = model[9][0].weight.grad.norm().item()
    print(f"{act.__name__:10s} 第1層 |g|={g0:.3e}  第10層 |g|={g9:.3e}  比值={g0/g9:.2e}")
```

---

## §3 完整的 MLP 實作

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, p_drop=0.2, use_bn=True):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            prev = h
        layers.append(nn.Linear(prev, out_dim))       # ★ 最後一層不加激活、不加 BN
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))                 # (B, ...) → (B, in_dim)

model = MLP(784, [512, 256, 128], 10)
```

### 逐行拆解「用迴圈組出 MLP」

```python
layers = []
prev = in_dim
for h in hidden_dims:                        # hidden_dims 例如 [512, 256, 128]
    layers.append(nn.Linear(prev, h, bias=not use_bn))
    ...
    prev = h                                 # ★ 下一層的輸入 = 這一層的輸出
layers.append(nn.Linear(prev, out_dim))      # ★ 最後一層獨立處理
self.net = nn.Sequential(*layers)
```

**`prev` 這個變數在做什麼？**
接龍。`Linear` 的 `in_features` 必須等於上一層的 `out_features`，
所以每建完一層就把 `prev` 更新成這一層的寬度。

```
in_dim=784, hidden=[512, 256]
  第 1 圈：Linear(784, 512)，prev 變成 512
  第 2 圈：Linear(512, 256)，prev 變成 256
  迴圈外：Linear(256, out_dim)
```

**為什麼最後一層要拉出迴圈外？**
因為它**不能加激活函數、不能加 BN、不能加 Dropout** ——
它要輸出原始的 logits 給 `CrossEntropyLoss`。
加了 softmax 會變成做兩次 softmax，模型幾乎學不動。

**`nn.Sequential(*layers)` 的星號**
`Sequential` 的參數是「一層、一層、一層」，不是「一個裝著層的 list」。
`*` 把 list 拆開成一個個參數送進去。少寫這個星號會報錯。
（語法說明見 `01_syntax/00_看不懂時先讀這裡.md` §7）

**`bias=not use_bn` 這個小技巧**
`use_bn=True` 時 `not use_bn` 就是 `False` → 不要 bias。
理由同 CNN 那章：BN 的平移項會抵消掉 conv/linear 的 bias。

### 層的順序：Linear → BN → 激活 → Dropout

這是最常見的順序。有幾個爭論點：

- **BN 放激活前還是後？** 原論文放前面，實務上兩者差異不大，跟著原論文走即可。
- **Dropout 和 BN 一起用？** 有研究指出兩者會互相干擾。
  現代 CNN 常常**只用 BN 不用 Dropout**；MLP 則兩個都用。實驗看哪個好。

---

## §4 梯度消失 / 爆炸的四種解法

| 解法 | 原理 | 何時用 |
|---|---|---|
| 換激活函數（ReLU 系列） | 導數不飽和 | ★ 一定要 |
| 好的初始化（Kaiming/Xavier） | 讓每層輸出的變異數維持穩定 | ★ 一定要 |
| Normalization（BN/LN） | 強制每層輸入分布穩定 | ★ 深度 > 5 層時 |
| 殘差連接（skip connection） | 梯度有一條「高速公路」直達淺層 | ★ 深度 > 20 層時 |

### 初始化的數學直覺

要讓 `y = Wx` 的變異數維持不變：`Var(y) = n_in · Var(W) · Var(x)`
所以 `Var(W)` 應該是 `1/n_in`。

- **Xavier**：`Var(W) = 2/(n_in + n_out)`，假設激活函數在 0 附近線性（tanh）
- **Kaiming**：`Var(W) = 2/n_in`，考慮 ReLU 砍掉一半 → 多乘 2

```python
nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
```

---

## §5 過擬合與正則化

### 診斷

```
訓練 loss ↓  驗證 loss ↓        → 欠擬合或正常，繼續訓練
訓練 loss ↓  驗證 loss ↑        → ★ 過擬合
訓練 loss 高 驗證 loss 高        → 欠擬合，模型太小或 lr 不對
```

### 處方（依投報率排序）

1. **更多資料** —— 最有效但最貴
2. **資料增強** —— ★ 最划算
3. **Early stopping** —— 免費
4. **Weight decay** —— `AdamW(weight_decay=0.01~0.1)`
5. **Dropout** —— `p=0.1~0.5`
6. **降低模型容量** —— 最後手段（通常寧可用大模型 + 強正則）

### Dropout 的機制

```python
# 訓練時：以機率 p 把神經元歸零，剩下的乘 1/(1-p) 補償
# 測試時：全部保留，不做任何事（因為訓練時已經補償過了）
# 這叫 inverted dropout，PyTorch 用的就是這個
```

> ★ 這解釋了為何忘記 `model.eval()` 時驗證結果會變差且不穩定 ——
> 測試時還在隨機丟棄神經元。

### Weight decay 的直覺

在 loss 上加 `λ‖w‖²`，等於告訴模型「在能解釋資料的前提下，權重越小越好」。
小權重 = 函數更平滑 = 更不容易記住雜訊。

---

## §6 用 ANN 處理你的股票資料

```python
class StockMLP(nn.Module):
    """輸入 (B, W, F) 的時間視窗，攤平後用 MLP 分類。"""
    def __init__(self, window, n_feat, n_classes, hidden=(256, 128)):
        super().__init__()
        self.net = MLP(window * n_feat, list(hidden), n_classes, p_drop=0.3)

    def forward(self, x):          # x: (B, W, F)
        return self.net(x.flatten(1))     # (B, W*F) → (B, n_classes)
```

> ⚠️ **MLP 攤平會丟失時間順序資訊**（模型不知道哪個特徵是「昨天」哪個是「前天」）。
> 這正是為什麼 CNN / Transformer 在時序上更好 —— 它們有結構上的歸納偏置。
> **但一定要先做 MLP baseline**，才知道複雜模型有沒有真的比較好。

---

## §7 系統性調參實驗（W7 的作業）

固定 seed，一次只改一個變因：

```python
experiments = {
    "baseline":     dict(hidden=[256], p_drop=0.0, use_bn=False, lr=1e-3),
    "wider":        dict(hidden=[1024], p_drop=0.0, use_bn=False, lr=1e-3),
    "deeper":       dict(hidden=[256]*8, p_drop=0.0, use_bn=False, lr=1e-3),
    "deeper+bn":    dict(hidden=[256]*8, p_drop=0.0, use_bn=True,  lr=1e-3),
    "dropout":      dict(hidden=[256], p_drop=0.5, use_bn=False, lr=1e-3),
    "high_lr":      dict(hidden=[256], p_drop=0.0, use_bn=False, lr=1e-1),
}
for name, cfg in experiments.items():
    set_seed(42)
    acc = run(cfg)
    print(f"{name:12s} {acc:.4f}")
```

**你應該觀察到的現象：**

- `deeper`（8 層無 BN）反而比 1 層差 → 梯度消失
- `deeper+bn` 明顯改善 → BN 的作用
- `high_lr` 直接發散或震盪
- `dropout` 訓練 loss 較高但驗證 loss 較低 → 正則化在作用

---

## §8 動手練習

1. 跑 `03_code/03_ann_mnist.py`，達到 98% 以上。
2. 完成 §7 的六個實驗，做成表格 + 曲線圖。
3. 驗證 §2 的梯度消失實驗，記錄數字。
4. 手刻兩層 MLP 的反向傳播（純 NumPy），跟 PyTorch 的梯度用 `np.allclose` 比對。
5. 把 MNIST 的輸入**不做正規化**訓練一次，觀察差異。
6. 用 `StockMLP` 跑你的 2330 資料，建立 baseline 準確率。

---

## ✅ 自我檢核

- [ ] 證明「兩層線性 = 一層線性」
- [ ] 說出 sigmoid 造成梯度消失的數學原因（提示：導數最大值）
- [ ] 說出 Kaiming 初始化為何要乘 2
- [ ] 說出 Dropout 在 train 和 eval 的行為差異，以及 inverted dropout 是什麼
- [ ] 從 loss 曲線判斷過擬合/欠擬合並開出處方
- [ ] 說出 MLP 用在時序資料的根本限制
