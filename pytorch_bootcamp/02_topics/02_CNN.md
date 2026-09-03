# 主題 02 · CNN 卷積神經網路

> **對應**：李宏毅 CNN 篇 · CS231n · `03_code/04_cnn_cifar10.py`
> 你已經寫過 `CNN.py`，這章補上**你可能沒學到的部分**：感受野、殘差、遷移學習、視覺化。


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 為什麼是卷積（三個歸納偏置）

MLP 處理 `28×28` 圖片要攤平成 784 維，第一層就要 `784×256 = 200,704` 個參數，
而且**完全丟失空間結構**（左上角的像素和右下角的像素在 MLP 眼中沒有任何關係）。

CNN 用三個假設解決這件事：

| 假設 | 意義 | 實現方式 |
|---|---|---|
| **局部性** | 有意義的圖樣是局部的（眼睛、邊緣） | 小的 kernel（3×3） |
| **平移不變性** | 貓在左上角和右下角都是貓 | 權重共享（同一個 kernel 掃過全圖） |
| **階層性** | 邊緣 → 紋理 → 部件 → 物體 | 堆疊多層 + 池化 |

參數量對比：`Conv2d(1, 32, 3)` 只有 `1×32×3×3 + 32 = 320` 個參數。

---

## §2 卷積的數學 ★ 一定要會手算

### 輸出尺寸公式

```
H_out = floor((H_in + 2·padding - dilation·(kernel-1) - 1) / stride) + 1
```

沒有 dilation 時簡化為：

```
H_out = floor((H_in + 2p - k) / s) + 1
```

**常用組合（背下來）：**

| kernel | stride | padding | 效果 |
|---|---|---|---|
| 3 | 1 | 1 | ★ 尺寸不變（最常用） |
| 5 | 1 | 2 | 尺寸不變 |
| 7 | 2 | 3 | 尺寸減半（ResNet 的 stem） |
| 3 | 2 | 1 | ★ 尺寸減半 |
| 1 | 1 | 0 | ★ 只改 channel 數 |
| 4 | 2 | 1 | ★ 尺寸減半（DCGAN 用） |

```python
# 驗證：Conv2d(3, 64, k=5, s=2, p=2) 對 32×32
# (32 + 2*2 - 5) // 2 + 1 = 31 // 2 + 1 = 15 + 1 = 16   → (B, 64, 16, 16)

# 快速驗算工具
def conv_out(h, k, s=1, p=0, d=1):
    return (h + 2*p - d*(k-1) - 1) // s + 1
```

### 參數量與計算量

```
參數量 = C_in × C_out × k × k  (+ C_out 個 bias)
FLOPs ≈ H_out × W_out × C_in × C_out × k × k
```

> ★ **參數量跟輸入尺寸無關，但計算量跟輸入尺寸成正比。**
> 這就是為什麼「模型很小但跑很慢」是可能的。

### 感受野（Receptive Field）★

一個輸出像素能「看到」原圖多大的區域。

```
RF_out = RF_in + (k - 1) × 累積stride
```

```python
def receptive_field(layers):
    """layers = [(k, s), ...]"""
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf

receptive_field([(3,1), (3,1), (3,1)])            # 7
receptive_field([(3,1), (2,2), (3,1), (2,2)])     # 14
```

> ★ **三層 3×3 的感受野 = 一層 7×7**，但：
> - 參數量：`3 × (3×3×C²) = 27C²` vs `7×7×C² = 49C²` → **少 45%**
> - 非線性：三次 ReLU vs 一次 → **表達能力更強**
>
> 這就是 VGG 論文的核心洞見，也是為何現代網路幾乎只用 3×3。

---

## §3 各種卷積變體

```python
# 標準卷積
nn.Conv2d(64, 128, 3, padding=1)                  # 參數 64*128*9 = 73,728

# 1×1 卷積 ★ 三個用途
nn.Conv2d(256, 64, 1)     # ① 降維（bottleneck，省計算量）
                          # ② 跨 channel 資訊融合
                          # ③ 當作「對每個位置的全連接層」

# Depthwise Separable（MobileNet 的核心）★ 參數少 8~9 倍
nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1, groups=64),   # depthwise: 每個 channel 各自卷積
    nn.Conv2d(64, 128, 1),                        # pointwise: 1×1 混合 channel
)   # 參數 64*9 + 64*128 = 8,768（相比 73,728）

# 空洞卷積（擴大感受野不增加參數，語意分割常用）
nn.Conv2d(64, 64, 3, padding=2, dilation=2)       # 感受野等同 5×5

# 轉置卷積（上採樣，GAN/AE 解碼器用）
nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # 尺寸 ×2

# ★ 上採樣的更好選擇（避免棋盤格 artifact）
nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
              nn.Conv2d(128, 64, 3, padding=1))

# 1D 卷積（★ 時序資料用，你的股票專案）
nn.Conv1d(in_channels=n_feat, out_channels=64, kernel_size=3, padding=1)
# 輸入 (B, C, L)，所以要先 x.transpose(1, 2)
```

---

## §4 池化與下採樣

```python
nn.MaxPool2d(2)                    # 取最大值，保留最強的特徵
nn.AvgPool2d(2)                    # 平均，較平滑
nn.AdaptiveAvgPool2d((1, 1))       # ★ 全域平均池化，輸出固定 1×1
nn.Conv2d(c, c, 3, stride=2, p=1)  # ★ 用 stride 卷積取代池化（現代做法，可學習）
```

> ★ **Global Average Pooling 的價值**：
> 傳統 CNN 最後是 `Flatten → Linear(大量參數)`，
> 用 GAP 可以把參數量從幾百萬降到幾千，同時讓模型能吃任意尺寸的輸入。
> ```python
> nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(512, 10))
> ```

---

## §5 殘差連接 ResNet ★

**問題**：網路加深到某個程度，訓練誤差反而變高（不是過擬合，是**退化問題**）。

**解法**：讓每個 block 學「殘差」`F(x)` 而不是直接學 `H(x)`，輸出是 `H(x) = F(x) + x`。

**為什麼有效（兩個角度）**：
1. **恆等映射變容易**：如果最佳解就是「什麼都不做」，只要把 `F(x)` 的權重壓到 0 即可。
2. **梯度高速公路**：`∂L/∂x = ∂L/∂H · (1 + ∂F/∂x)`，那個 `1` 保證梯度至少能原樣傳回去，
   不會因為連乘而消失。

```python
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        # ★ 當 shape 不一致時，捷徑要用 1×1 conv 對齊
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)       # ★ 相加（不是 concat）
        return F.relu(out)                 # ★ 相加之後才 ReLU
```

### 逐行拆解 `BasicBlock`

```python
self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
#                      └輸入 └輸出 └核 └步幅 └補邊
```
- 位置參數的順序是 `(in_channels, out_channels, kernel_size, stride, padding)`
- `bias=False`：後面接 BatchNorm，BN 自己有平移項 `beta`，
  conv 的 bias 會被完全抵消，留著只是白白多幾百個參數

```python
self.shortcut = nn.Sequential()      # 空的 Sequential = 什麼都不做
if stride != 1 or in_c != out_c:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, bias=False),
        nn.BatchNorm2d(out_c))
```
- **為什麼要判斷？** 殘差是 `out + x`，兩者形狀必須一樣。
  當這個 block 改變了通道數或縮小了尺寸，捷徑那條路也得跟著改
- `1x1 conv` 就是專門用來「只改通道數不動空間尺寸」的工具
- 形狀一樣時用空的 `Sequential`（或 `nn.Identity()`）當佔位符，
  這樣 `forward` 裡可以無條件寫 `self.shortcut(x)`，不用寫 if

```python
out = out + self.shortcut(x)         # ✅
# out += self.shortcut(x)            # ❌
```
- 兩者數學上一樣，但 `+=` 是**就地修改**，會覆寫掉 backward 需要的中間值
- 症狀：`RuntimeError: one of the variables needed for gradient computation
  has been modified by an inplace operation`

```python
return F.relu(out, inplace=True)
```
- ReLU 放在**相加之後**（原論文的 post-activation 設計）
- `inplace=True` 直接改寫輸入省記憶體 —— ReLU 是少數這樣做安全的層
  （因為它的反向傳播只需要知道「輸入是否 > 0」，而輸出剛好保留了這個資訊）

> ⚠️ **兩個常見寫錯的地方**：
> 1. `out += self.shortcut(x)` 用就地運算會壞掉 autograd → 寫成 `out = out + ...`
> 2. 最後的 ReLU 要放在相加**之後**（Post-activation，原論文版本）

---

## §6 經典架構演進（知道脈絡才知道為什麼）

| 年份 | 架構 | 關鍵創新 |
|---|---|---|
| 1998 | LeNet-5 | 卷積 + 池化的基本範式 |
| 2012 | AlexNet | ReLU、Dropout、GPU 訓練、資料增強 |
| 2014 | VGG | ★ 只用 3×3 堆疊，結構統一 |
| 2014 | GoogLeNet | Inception 多尺度、1×1 降維 |
| 2015 | **ResNet** | ★★ 殘差連接，可訓練 100+ 層 |
| 2016 | DenseNet | 每層都連到後面所有層 |
| 2017 | MobileNet | Depthwise separable，行動裝置 |
| 2019 | EfficientNet | 複合縮放（深度/寬度/解析度一起調） |
| 2020 | ViT | 用 Transformer 取代卷積 |
| 2022 | ConvNeXt | 用 ViT 的訓練技巧回頭改良 CNN |

---

## §7 遷移學習（★ 實務上最常用的技術）

```python
from torchvision.models import resnet18, ResNet18_Weights

# ★ 新版 API 用 weights，不要用已棄用的 pretrained=True
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
preprocess = weights.transforms()          # ★ 直接拿到官方的前處理

# 換掉分類頭
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### 三種策略

| 策略 | 做法 | 適用 |
|---|---|---|
| **Linear probing** | 凍結全部，只訓練 `fc` | 資料很少（< 1000 張）、快速驗證 |
| **Fine-tune 全部** | 全部解凍，用小 lr（1e-4~1e-5） | ★ 資料中等以上，效果最好 |
| **分層學習率** | 骨幹 1e-5、頭 1e-3 | ★ 兼顧兩者 |

```python
# 分層學習率
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")],
     "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-3},
], weight_decay=0.01)
```

> ⚠️ **用預訓練模型時，正規化統計量必須用 ImageNet 的那組**：
> `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`
> 用錯會讓效果掉一大截，而且不會報錯。

---

## §8 資料增強對 CNN 的重要性

CIFAR-10 從零訓練的實測參考（ResNet-18）：

| 設定 | 大約準確率 |
|---|---|
| 無增強 | 84% |
| + RandomCrop + Flip | 92% |
| + Cosine LR + 200 epochs | 94% |
| + Cutout / RandAugment | 95%+ |

> ★ **資料增強的效益常常大於換模型。** 先把增強做好再考慮換架構。

---

## §9 CNN 視覺化

### 9.1 看第一層學到什麼

```python
w = model.conv1.weight.detach().cpu()     # (64, 3, 7, 7)
w = (w - w.min()) / (w.max() - w.min())   # 正規化到 [0,1]
grid = torchvision.utils.make_grid(w, nrow=8)
plt.imshow(grid.permute(1, 2, 0))
# 訓練好的第一層應該看得到「邊緣偵測器」和「顏色斑塊」
```

### 9.2 Grad-CAM（看模型在看哪裡）★

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.acts = o.detach()

    def _save_grad(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)                        # (1, C)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1) 全域平均梯度
        cam = (weights * self.acts).sum(1, keepdim=True)      # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam[0, 0].cpu().numpy(), class_idx

# 用法
cam_fn = GradCAM(model, model.layer4[-1])
heatmap, cls = cam_fn(img[None].to(device))
plt.imshow(img.permute(1,2,0)); plt.imshow(heatmap, cmap="jet", alpha=0.5)
```

> ★ Grad-CAM 是**檢查模型有沒有學到正確特徵**的關鍵工具。
> 如果模型分類「馬」是靠圖片角落的浮水印，Grad-CAM 一看就知道。

---

## §10 CNN 用在你的時序資料（1D CNN）

```python
class CNN1D(nn.Module):
    """輸入 (B, W, F) 的時間視窗。"""
    def __init__(self, n_feat, n_classes, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1),     nn.BatchNorm1d(ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch*2, 3, padding=1),   nn.BatchNorm1d(ch*2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),             # (B, ch*2, 1)
        )
        self.fc = nn.Linear(ch*2, n_classes)

    def forward(self, x):              # x: (B, W, F)
        x = x.transpose(1, 2)          # ★ (B, F, W)  Conv1d 要 channel 在中間
        x = self.net(x).flatten(1)     # (B, ch*2)
        return self.fc(x)
```

> ★ **時序上更好的選擇：因果卷積（causal conv）**
> 一般卷積的 `padding=1` 會讓輸出看到「未來」的時間步。
> 做預測任務時要用因果卷積（只往左 pad）：
> ```python
> x = F.pad(x, (k-1, 0))          # 只在左邊補
> x = nn.Conv1d(c, c, k, padding=0)(x)
> ```
> 加上 dilation 就是 **TCN / WaveNet** 的核心結構，很適合你的股票專案。

---

## §11 動手練習

1. 手算 `Conv2d(3, 64, k=5, s=2, p=2)` 對 `32×32` 的輸出尺寸，再用程式驗證。
2. 算出 `[(3,1),(3,1),(2,2),(3,1),(3,1),(2,2)]` 的感受野。
3. 跑 `03_code/04_cnn_cifar10.py`，用資料增強 + cosine 排程衝到 90%+。
4. 實作 `BasicBlock`，比較「有無殘差」在 20 層網路上的訓練曲線。
5. 用預訓練 ResNet-18 做遷移學習，比較 linear probe / full fine-tune / 分層 lr。
6. 實作 Grad-CAM，找出模型分類錯誤的圖片並看它在看哪裡。
7. 把 `CNN1D` 改成因果卷積 + dilation，用在 2330 資料上，跟你原本的 `CNN_finance.py` 比較。

---

## ✅ 自我檢核

- [ ] 不查表寫出卷積輸出尺寸公式並手算
- [ ] 說出 CNN 的三個歸納偏置
- [ ] 說出為何用三層 3×3 而不是一層 7×7（兩個理由）
- [ ] 說出 1×1 卷積的三個用途
- [ ] 說出殘差連接為何能解決退化與梯度消失（寫出梯度公式）
- [ ] 說出 Global Average Pooling 的兩個好處
- [ ] 說出用預訓練模型時最容易忘記的一件事（正規化統計量）
- [ ] 說出一般卷積用在預測任務上的資訊洩漏風險
