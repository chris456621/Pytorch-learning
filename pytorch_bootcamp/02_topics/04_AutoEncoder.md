# 主題 04 · AutoEncoder / VAE

> **對應**：李宏毅 Auto-encoder 篇 · Kingma & Welling (2013) · `03_code/06_autoencoder_vae.py`


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 AutoEncoder 的核心想法

```
x  →  [Encoder]  →  z（瓶頸，維度遠小於 x）  →  [Decoder]  →  x̂
                    訓練目標：最小化 ‖x − x̂‖²
```

**沒有標籤**，模型只是學「壓縮再還原」。
因為 `z` 的維度很小，模型被迫**只保留最重要的資訊** → 學到有用的表徵。

> ★ **這是 Self-supervised learning 的雛形。**
> AE 用「重建自己」當作 pretext task；SSL 則發展出更多樣的 pretext task。

### 最簡單的 AE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=784, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent),           # ★ 瓶頸層，不加激活
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, in_dim),
            nn.Sigmoid(),                    # ★ 輸出對應 [0,1] 的像素
        )

    def forward(self, x):
        z = self.encoder(x.flatten(1))
        return self.decoder(z), z

# 訓練
recon, z = model(x)
loss = F.mse_loss(recon, x.flatten(1))
```

> ⚠️ **輸出激活要跟資料範圍匹配**：
> - 資料在 `[0,1]` → `Sigmoid` + MSE 或 BCE
> - 資料在 `[-1,1]` → `Tanh` + MSE
> - 資料未正規化 → **不加激活** + MSE
>
> 配錯的症狀：重建圖片一片灰，loss 卡住不降。

---

## §2 AE 學到的是什麼

**線性 AE ≈ PCA**：encoder/decoder 都線性且 loss 是 MSE 時，
AE 學到的子空間跟 PCA 主成分張成的空間相同。

**非線性 AE > PCA**：加上激活函數後可以學到彎曲的流形（manifold）。

---

## §3 卷積 AE

```python
class ConvAE(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),     # 28→14
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),    # 14→7
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # 7→4
            nn.Flatten(1),
            nn.Linear(128 * 4 * 4, latent),
        )
        self.fc = nn.Linear(latent, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(),                                # 4→8
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(),                                # 8→16
            nn.Upsample(size=(28, 28)), nn.Conv2d(32, 1, 3, 1, 1),        # →28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        h = self.fc(z).view(-1, 128, 4, 4)
        return self.decoder(h), z
```

> ★ 這裡刻意用 `Upsample + Conv` 而不是 `ConvTranspose2d`：
> 轉置卷積容易產生**棋盤格 artifact**，而且 `output_padding` 很難算對。
> 想用 `ConvTranspose2d` 時，一定要用 `torchinfo.summary` 驗證每層 shape。

---

## §4 Denoising AutoEncoder（DAE）

**想法**：輸入加噪，但要求還原**乾淨**的原圖。
這強迫模型學到資料的結構，而不是學會恆等映射（複製貼上）。

```python
def train_dae_step(model, x, noise_std=0.3):
    x_noisy = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    recon, _ = model(x_noisy)
    return F.mse_loss(recon, x)          # ★ 目標是乾淨的 x，不是 x_noisy
```

其他加噪方式：

```python
mask = (torch.rand_like(x) > 0.3).float()
x_corrupt = x * mask                     # ★ 隨機遮蔽，這就是 MAE 的雛形
```

> ★ **DAE → MAE 的連結**：Masked Autoencoder（He et al. 2021）本質上就是
> 「遮掉 75% 的 patch 再重建」的 DAE，只是用了 Transformer 且遮蔽比例極高。

### 用 AE 做異常偵測（★ 可直接用在你的股票資料）

```python
# 只用「正常」資料訓練 AE。異常資料的重建誤差會明顯偏高。
model.eval()
with torch.no_grad():
    recon, _ = model(x)
    err = (recon - x).pow(2).flatten(1).mean(1)      # 每筆樣本的重建誤差

threshold = err_train.quantile(0.99)                 # 用訓練集 99 分位當門檻
anomalies = err > threshold
```

**應用到 2330**：用平穩期的 K 線視窗訓練 AE，重建誤差大的日子就是「異常波動日」。
可以當成一個新特徵餵給下游模型。

---

## §5 VAE 變分自編碼器 ★★

### 5.1 AE 的問題

AE 的 latent space 是**不連續、有洞**的。
在兩個訓練樣本的 latent 之間取中點解碼，常常得到毫無意義的東西。
所以 **AE 不能拿來生成**。

VAE 的解法：**讓 encoder 輸出一個機率分布而非一個點**，並強迫它接近標準常態。
這樣整個 latent space 被「填滿」，任意採樣都能解碼出合理的東西。

### 5.2 ELBO

想最大化 `log p(x)` 但它不可解。引入近似後驗 `q(z|x)`，推得下界：

```
log p(x) ≥ E_q(z|x)[ log p(x|z) ] − KL( q(z|x) ‖ p(z) )
           └───── 重建項 ─────┘   └──── 正則項 ────┘
```

- **重建項**：解碼出來要像原圖
- **KL 項**：encoder 的分布要接近先驗 `p(z) = N(0, I)`

兩者是**對抗**的：重建想讓每個樣本佔據自己的位置，KL 想把大家壓回原點。

### 5.3 Reparameterization Trick ★（最重要的一個技巧）

**問題**：需要從 `q(z|x) = N(μ, σ²)` 採樣，但「採樣」不可微，
梯度無法從 `z` 傳回 `μ` 和 `σ`。

**解法**：把隨機性移到外面

```
z = μ + σ ⊙ ε ,  其中 ε ~ N(0, I)
```

現在 `z` 對 `μ`、`σ` 是可微的（`∂z/∂μ = 1`、`∂z/∂σ = ε`），
隨機性只存在於 `ε`，而 `ε` 不需要梯度。

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)        # ★ 網路輸出 log σ²，保證 σ>0 且數值穩定
    eps = torch.randn_like(std)
    return mu + eps * std                 # ★ 梯度可以流過 mu 和 std
```

> ★ **為什麼輸出 `logvar` 而不是 `std`？**
> `std` 必須為正，直接輸出需要額外激活且容易數值不穩。
> 輸出 `log σ²` 值域是整個實數軸，`exp(0.5·logvar)` 保證為正。

### 5.4 完整 VAE

```python
class VAE(nn.Module):
    def __init__(self, in_dim=784, hidden=400, latent=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        x = x.flatten(1)
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    x = x.flatten(1)
    B = x.size(0)
    # ★ 用 sum 而不是 mean：ELBO 是對每個維度加總，最後才除以 batch
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = (recon_loss + beta * kl) / B
    return total, recon_loss.item() / B, kl.item() / B
```

### 逐行拆解 `vae_loss`

```python
recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
```
- `reduction="sum"`（不是預設的 `"mean"`）：ELBO 的定義是「對每個像素加總」。
  用 `mean` 的話 recon 項會被除以 784，相對 KL 項就變得太小，兩者失衡
- 最後統一除以 `B`（batch size），所以得到的是「每張圖的 ELBO」

```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```
- 這是 `KL(N(mu, sigma^2) || N(0,1))` 的**解析解**，不用採樣估計，直接套公式
- `logvar.exp()` 就是 `sigma^2`；`mu.pow(2)` 就是 `mu^2`
- 直覺：`mu` 離 0 越遠、`sigma` 離 1 越遠，這一項就越大 →
  它在把 encoder 的輸出分布「拉回」標準常態

```python
return (recon_loss + beta * kl) / B, recon_loss.item() / B, kl.item() / B
```
- 回傳三個東西：**總 loss（要 backward 的）**、以及兩個**純數字**（給 log 用）
- ★ 後兩個一定要 `.item()`：直接回傳 tensor 的話會抓著整張計算圖不放，
  累積幾百個 batch 就 OOM
- **分開記錄 recon 和 KL 是必要的**，因為 posterior collapse 的症狀
  就是「總 loss 看起來在降，但 KL 已經掉到 0」

**KL 散度解析解**（`N(μ,σ²)` 對 `N(0,1)`）：

```
KL = −½ Σ ( 1 + log σ² − μ² − σ² )
```

### 5.5 生成新樣本

```python
model.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim, device=device)     # ★ 直接從先驗採樣
    samples = model.dec(z).view(-1, 1, 28, 28)
```

### 5.6 Posterior Collapse ★（VAE 最著名的失敗模式）

**症狀**：KL 項趨近 0，`μ → 0`、`σ → 1`，encoder 完全忽略輸入。
解碼器變成不看 `z` 的生成器，重建全部長一樣（模糊的平均臉）。

**原因**：KL 項太強，或解碼器太強大（自己就能生成，不需要 z）。

**解法：**

```python
# 1. KL annealing：beta 從 0 慢慢升到 1
beta = min(1.0, epoch / warmup_epochs)

# 2. Free bits：每個維度的 KL 至少保留 lambda
kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # (B, latent)
kl = torch.clamp(kl_per_dim, min=free_bits).sum(1).mean()

# 3. beta-VAE：beta < 1 偏重重建；beta > 1 偏重解耦
```

**監控方式**：訓練時**分開記錄** recon_loss 和 KL。KL 掉到接近 0 就是 collapse 了。

---

## §6 AE 家族總覽

| 模型 | 特點 | 用途 |
|---|---|---|
| AE | 最基本 | 降維、特徵提取 |
| DAE | 輸入加噪 | 更穩健的表徵、去噪 |
| Sparse AE | latent 加 L1 | 學到稀疏可解釋的特徵 |
| **VAE** | 機率式 latent | ★ 生成、latent 可內插 |
| beta-VAE | 加大 KL 權重 | 解耦表徵 |
| VQ-VAE | 離散 latent（碼本） | ★ 影像生成、語音、DALL-E 基礎 |
| **MAE** | 遮蔽 75% 重建 | ★ 現代 SSL 主力（見主題 05） |

### VQ-VAE 的關鍵技巧（跟 autograd 有關，值得看）

```python
# 量化：找碼本中最近的向量（不可微）
d = (z_e.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
idx = d.argmin(1)
z_q = codebook[idx]

# ★ Straight-through：forward 用 z_q，backward 的梯度直接給 z_e
z_q = z_e + (z_q - z_e).detach()
```

這正是 `01_syntax/07_autograd.md` §8 教的那個 trick。

---

## §7 動手練習

1. 訓練基本 AE，latent 設成 2 維，直接畫出 latent 散佈圖（用類別著色）。
2. 訓練 DAE，比較不同噪音強度對重建品質的影響。
3. 用 AE 做異常偵測：用 2330 平穩期訓練，找出重建誤差最大的 10 天，查那幾天發生什麼事。
4. 實作 VAE，畫出 latent **內插**：在兩個數字的 latent 之間取 10 個點解碼。
5. 從 `N(0,I)` 採樣 64 個 z 生成圖片。
6. 故意把 `beta` 設成 10，觀察 posterior collapse，再用 KL annealing 修好。
7. 分開畫 recon_loss 和 KL 的曲線，觀察兩者的拉扯。

---

## ✅ 自我檢核

- [ ] 說出 AE 的瓶頸層為何是關鍵
- [ ] 說出輸出激活函數要怎麼跟資料範圍匹配
- [ ] 說出 AE 為何不能拿來生成，VAE 怎麼解決
- [ ] 寫出 ELBO 並說明兩項的意義與拉扯關係
- [ ] 解釋 reparameterization trick 為何必要（要講到梯度）
- [ ] 說出為何 encoder 輸出 `logvar` 而不是 `std`
- [ ] 說出 posterior collapse 的症狀、原因與三種解法
- [ ] 說出 DAE 和 MAE 的關係
