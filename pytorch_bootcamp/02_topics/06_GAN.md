# 主題 06 · GAN 生成對抗網路

> **對應**：李宏毅 GAN 篇 · Goodfellow (2014) / DCGAN / WGAN-GP · `03_code/08_gan_dcgan.py`
> **前置**：`01_syntax/07_autograd.md`（★ 沒學好 detach 這章會很痛苦）


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 核心想法：兩個網路互相對抗

```
Generator  G:  z ~ N(0,I)  →  假樣本 G(z)
Discriminator D:  樣本  →  「是真的」的機率

G 的目標：騙過 D
D 的目標：分辨真假
```

**Minimax 目標**：

```
min_G max_D  E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 − D(G(z)))]
```

**理論結果**：最佳解時 `p_G = p_data`，此時 `D(x) = 0.5`（完全分不出來）。

### Non-saturating loss ★（實務上一定用這個）

原始的 G loss 是 `min log(1 − D(G(z)))`。
訓練初期 G 很爛，`D(G(z)) ≈ 0`，此時 `log(1−D(G(z)))` 的**梯度趨近 0** → G 學不動。

**改法**：把 G 的目標改成 `max log D(G(z))`（等價於 `min −log D(G(z))`）。
數學上不完全等價，但梯度在初期大得多。

```python
# 實作上：兩者都用 BCE，只是標籤不同
bce = nn.BCEWithLogitsLoss()
# G 的 loss：把假的標成「真的」
g_loss = bce(D(fake), torch.ones_like(...))     # ★ non-saturating
```

---

## §2 標準訓練迴圈 ★（detach 的位置是關鍵）

```python
import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()

for real, _ in loader:
    real = real.to(device)
    B = real.size(0)
    ones  = torch.ones(B, 1, device=device)
    zeros = torch.zeros(B, 1, device=device)

    # ---------- 1. 訓練 D ----------
    z = torch.randn(B, z_dim, device=device)
    fake = G(z)

    d_real = D(real)
    d_fake = D(fake.detach())            # ★★★ detach！不讓梯度流回 G
    d_loss = bce(d_real, ones) + bce(d_fake, zeros)

    opt_D.zero_grad(set_to_none=True)
    d_loss.backward()
    opt_D.step()

    # ---------- 2. 訓練 G ----------
    d_fake2 = D(fake)                    # ★ 這裡「不能」detach，梯度要流回 G
    g_loss = bce(d_fake2, ones)          # ★ 把假的當成真的（non-saturating）

    opt_G.zero_grad(set_to_none=True)
    g_loss.backward()
    opt_G.step()
```

### 逐行拆解訓練迴圈

```python
z = torch.randn(B, z_dim, device=device)
```
- `randn` = 標準常態亂數（不是 `rand`，`rand` 是 0~1 均勻分布）
- `device=device` 必須加，否則雜訊建在 CPU、G 在 GPU，相乘就報錯

```python
fake = G(z)                     # (B, C, H, W) 一批假圖
ones  = torch.ones(B, 1, device=device)      # 「這是真的」的標籤
zeros = torch.zeros(B, 1, device=device)     # 「這是假的」的標籤
```
- 為什麼是 `(B, 1)` 不是 `(B,)`？因為 D 的輸出是 `(B,1)`，
  `BCEWithLogitsLoss` 要求兩者形狀完全相同，不然會**靜默廣播**成 `(B,B)`

```python
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
#                                      └────────┘
```
- `bce` 是 `nn.BCEWithLogitsLoss()`，**吃 logits 不吃機率**（所以 D 最後不加 sigmoid）
- `D(real)` 要被判成 1，`D(fake)` 要被判成 0 → 兩個 loss 相加
- **`.detach()` 是這整段最重要的東西**：它回傳一個「值一樣但不在計算圖上」的張量，
  於是 `d_loss.backward()` 的梯度走到這裡就停住，不會流回 G

```python
opt_D.zero_grad(set_to_none=True)
d_loss.backward()
opt_D.step()
```
- `opt_D` 建立時只收了 `D.parameters()`，所以 `step()` 只會更新 D
- 但**梯度還是會被算出來並累積到 G 的 `.grad` 上**（如果沒 detach），
  這就是「污染」—— 你可以用 `--no-detach` 親眼看到 G 的梯度範數不是 0

```python
g_loss = bce(D(fake), ones)      # ★ 這裡「不」detach
```
- G 想騙 D，所以把假圖標成「真的」
- 梯度路徑：`g_loss → D 的各層 → fake → G 的各層`。
  中間非 detach 不可的是 D 這一段 —— 沒有它梯度根本到不了 G

### 🔥 三個必須理解的細節

**1. 為什麼訓練 D 時要 `fake.detach()`？**
不 detach 的話，`d_loss.backward()` 會把梯度一路傳回 G 的參數。
雖然 `opt_D.step()` 不會更新 G，但 G 的 `.grad` 被污染了，
下一步訓練 G 時如果忘記 `zero_grad` 就會出錯。而且白白浪費計算。

**2. 為什麼訓練 G 時不能 detach？**
G 的梯度**必須**經過 D 才能算出來：
`∂g_loss/∂θ_G = ∂g_loss/∂D · ∂D/∂fake · ∂fake/∂θ_G`。
detach 掉就斷了，G 完全學不到東西（loss 會報 `does not require grad`）。

**3. 可以重用 `fake` 嗎？**
可以，如上面的寫法（`fake` 只 forward 一次，D 跑兩次）。
但因為 D 剛剛被更新過，第二次的 `D(fake)` 用的是新的 D —— 這是正確且標準的做法。

> ✍️ **驗證你懂了**：把 `fake.detach()` 的 detach 拿掉，
> 印出 `G.parameters()` 的 grad norm，觀察它被污染。

---

## §3 DCGAN 架構準則（論文的實作指南）

```python
class Generator(nn.Module):
    """z (B, nz, 1, 1) → 圖片 (B, 3, 64, 64)"""
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            # 輸入 (B, nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),          # (B, 512, 4, 4)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),          # (B, 256, 8, 8)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),          # (B, 128, 16, 16)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),            # (B, 64, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),                                     # ★ 輸出 [-1, 1]
        )                                                  # (B, 3, 64, 64)
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),                       # ★ D 用 LeakyReLU
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),      # (B, 1, 1, 1)
        )
    def forward(self, x): return self.net(x).view(-1, 1)   # ★ 輸出 logits，不加 sigmoid
```

### DCGAN 的五條準則

1. **不用池化**：D 用 stride conv 下採樣，G 用 transposed conv 上採樣
2. **G 和 D 都用 BatchNorm**（但 G 的輸出層和 D 的輸入層不加）
3. **G 用 ReLU，輸出層用 Tanh**
4. **D 用 LeakyReLU(0.2)**
5. **權重初始化 `N(0, 0.02)`**

```python
def dcgan_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

G.apply(dcgan_init); D.apply(dcgan_init)
```

> ⚠️ **資料正規化必須配合 Tanh**：
> ```python
> T.Normalize([0.5]*3, [0.5]*3)      # 把 [0,1] 映射到 [-1,1]
> # 顯示時要反正規化：img = img * 0.5 + 0.5
> ```
> 這是最常見的「生成圖片一片黑或一片白」的原因。

### 優化器設定

```python
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
# ★ beta1 = 0.5 而不是 0.9，這是 DCGAN 論文的關鍵設定
```

---

## §4 GAN 的四大失敗模式與診斷

| 症狀 | 名稱 | 原因 | 對策 |
|---|---|---|---|
| 生成的圖全部長一樣 | **Mode collapse** | G 找到一個能騙過 D 的「捷徑」 | minibatch discrimination、WGAN-GP、加大 batch、unrolled GAN |
| D loss → 0，G loss 暴增 | **D 太強** | D 完美分辨，G 沒有梯度 | 降 D 的 lr、少訓練 D、加 D 的 dropout、label smoothing |
| 兩邊 loss 都劇烈震盪 | **不收斂** | 對抗訓練本質不穩 | 降 lr、用 WGAN-GP、加 spectral norm |
| 生成圖有規律方格 | **棋盤格 artifact** | ConvTranspose 的重疊 | kernel 用 4/stride 2，或改 Upsample + Conv |

### 診斷指標（★ GAN 的 loss 值本身沒有意義，一定要看這些）

```python
# 1. D 對真假的平均輸出（最重要的健康指標）
d_real_prob = torch.sigmoid(D(real)).mean().item()
d_fake_prob = torch.sigmoid(D(fake)).mean().item()
# 健康：d_real ~ 0.6-0.8, d_fake ~ 0.2-0.4
# D 太強：d_real ~ 1.0, d_fake ~ 0.0   ← G 拿不到梯度
# G 太強：兩者都 ~ 0.5 但生成品質差   ← D 太弱

# 2. 生成樣本的多樣性（偵測 mode collapse）
with torch.no_grad():
    samples = G(torch.randn(64, nz, 1, 1, device=device))
    pairwise = torch.cdist(samples.flatten(1), samples.flatten(1))
    print(f"樣本間平均距離 {pairwise.mean().item():.4f}")   # 太小 = collapse

# 3. 固定 z 觀察演進（★ 最直觀，每個 epoch 存一次圖）
fixed_z = torch.randn(64, nz, 1, 1, device=device)      # 訓練開始前建立，之後都用它
```

---

## §5 訓練穩定技巧（實務錦囊）

```python
# 1. Label smoothing：真標籤用 0.9 而不是 1.0
d_loss = bce(d_real, ones * 0.9) + bce(d_fake, zeros)

# 2. 給 D 的輸入加噪（instance noise）
real_noisy = real + 0.05 * torch.randn_like(real)

# 3. 調整 D 和 G 的訓練比例
n_critic = 1      # WGAN 通常用 5（D 訓練 5 次才訓練 G 一次）

# 4. Spectral Normalization（★ 簡單有效，強烈推薦）
from torch.nn.utils.parametrizations import spectral_norm
nn.Sequential(spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2))
# 限制 D 的 Lipschitz 常數，讓梯度更平滑

# 5. EMA of G（生成品質明顯提升）
@torch.no_grad()
def update_ema(G_ema, G, decay=0.999):
    for p_e, p in zip(G_ema.parameters(), G.parameters()):
        p_e.data.mul_(decay).add_(p.data, alpha=1 - decay)
# 評估和生成時用 G_ema
```

---

## §6 WGAN-GP（更穩定的替代方案）

**問題**：原始 GAN 用 JS 散度，當兩個分布不重疊時 JS 是常數 `log2` → **梯度為 0**。

**WGAN 的解法**：改用 Wasserstein 距離（推土機距離），
它在分布不重疊時仍有有意義的梯度。
代價是 D（改稱 critic）必須是 1-Lipschitz 函數。

**WGAN-GP** 用梯度懲罰來強制這個約束（比原本的 weight clipping 好很多）：

```python
def gradient_penalty(D, real, fake, device):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)   # ★ 真假之間的內插點
    d_hat = D(x_hat)
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,          # ★★ 必須 True！GP 本身要能再對它求導（二階梯度）
        retain_graph=True,
    )[0]
    grads = grads.flatten(1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# WGAN-GP 訓練
for i, (real, _) in enumerate(loader):
    # --- critic ---
    fake = G(torch.randn(B, nz, 1, 1, device=device))
    d_loss = D(fake.detach()).mean() - D(real).mean() + 10.0 * gradient_penalty(D, real, fake.detach(), device)
    opt_D.zero_grad(); d_loss.backward(); opt_D.step()

    # --- generator（每 n_critic 步一次）---
    if i % 5 == 0:
        fake = G(torch.randn(B, nz, 1, 1, device=device))
        g_loss = -D(fake).mean()
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```

> ★ **`create_graph=True` 是這裡的關鍵**，也是 `07_autograd.md` 講的
> 「對梯度再求梯度」的實際應用。這在 meta-learning（MAML）也會用到。
>
> ⚠️ WGAN-GP 的 critic **不要用 BatchNorm**（GP 是逐樣本的，BN 會讓樣本互相影響）。
> 改用 LayerNorm 或 InstanceNorm。

### GAN loss 變體對照

| 版本 | D loss | G loss |
|---|---|---|
| 原始（saturating） | `−log D(x) − log(1−D(G(z)))` | `log(1−D(G(z)))` |
| ★ Non-saturating | 同上 | `−log D(G(z))` |
| LSGAN | `(D(x)−1)² + D(G(z))²` | `(D(G(z))−1)²` |
| ★ WGAN-GP | `D(G(z)) − D(x) + λ·GP` | `−D(G(z))` |
| Hinge | `relu(1−D(x)) + relu(1+D(G(z)))` | `−D(G(z))` |

---

## §7 條件式 GAN（cGAN）

```python
class ConditionalG(nn.Module):
    def __init__(self, nz, n_classes, ngf=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, nz)        # 類別 → 向量
        self.net = ...                                   # 同 DCGAN

    def forward(self, z, y):
        z = z * self.embed(y)[..., None, None]          # ★ 條件注入（乘法或串接都可）
        return self.net(z)
```

D 也要吃條件（把 label embedding 串成額外的 channel）。
這是 pix2pix、CycleGAN、StyleGAN 的基礎。

---

## §8 評估 GAN

| 指標 | 意義 | 備註 |
|---|---|---|
| 肉眼看 | ★ 最重要 | 固定 z，每個 epoch 存圖對比 |
| **FID** | 用 Inception 特徵比較真假分布的距離 | ★ 業界標準，越低越好 |
| IS | Inception Score | 已知有缺陷，較少用 |
| Precision / Recall | 品質 vs 多樣性分開評 | 能診斷 mode collapse |

```python
# FID 需要 pip install torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=2048).to(device)
fid.update(real_uint8, real=True)
fid.update(fake_uint8, real=False)
print(fid.compute())
```

> **GAN loss 完全不能拿來比較模型好壞。** D 和 G 是動態平衡，
> loss 低不代表生成得好。一定要用 FID 或肉眼。

---

## §9 動手練習

1. 跑 `03_code/08_gan_dcgan.py`（MNIST 或 CIFAR-10），存下每個 epoch 的固定 z 生成圖。
2. **故意拿掉 `fake.detach()`**，印出 G 參數的 grad norm 觀察污染。
3. 故意讓 D 學太快（D 的 lr 設 10 倍），觀察 G loss 暴增、生成崩壞。
4. 實作 mode collapse 的診斷指標，找出 collapse 發生的 epoch。
5. 加上 spectral norm，比較訓練穩定性。
6. 實作 WGAN-GP，比較跟原始 GAN 的收斂曲線。
7. 實作 conditional GAN，指定生成某個數字。
8. 用 latent 內插：在兩個 z 之間取 10 個點生成，看過渡是否平滑。

---

## ✅ 自我檢核

- [ ] 寫出 GAN 的 minimax 目標
- [ ] 說出 non-saturating loss 存在的理由（梯度角度）
- [ ] 說出訓練 D 時哪裡要 detach、訓練 G 時為何不能
- [ ] 說出 mode collapse 的症狀與三種對策
- [ ] 說出 GAN loss 為何不能拿來比較模型好壞
- [ ] 說出 DCGAN 的五條架構準則
- [ ] 說出 Tanh 輸出和資料正規化要怎麼配合
- [ ] 解釋 WGAN-GP 裡 `create_graph=True` 的作用
