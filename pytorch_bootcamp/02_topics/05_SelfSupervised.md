# 主題 05 · Self-Supervised Learning 自監督學習

> **對應**：李宏毅 SSL 篇 · SimCLR / BYOL / SimSiam / MAE · `03_code/07_ssl_simclr.py`


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 核心問題：標籤很貴，資料很便宜

| 學習方式 | 需要標籤 | 監督訊號來自 |
|---|---|---|
| Supervised | ✅ 人工標註 | 標籤 |
| Unsupervised | ❌ | 資料的統計結構（分群、降維） |
| **Self-supervised** | ❌ | ★ **從資料自己構造出來的任務** |

**SSL 的兩階段流程：**

```
階段 1（pretext / 預訓練）：用無標籤資料 + 自造任務 → 學到好的表徵
階段 2（downstream / 下游）：用少量標籤 fine-tune 或 linear probe → 解決真實任務
```

> ★ **為什麼你該學這個**：你的股票資料有數十年的無標籤 K 線，
> 但「有意義的標籤」（真正能獲利的漲跌訊號）稀少且雜訊大。
> SSL 正是為這種情況設計的。

---

## §2 SSL 的兩大流派

### 流派 A：生成式 —— 重建被破壞的輸入

| 方法 | pretext task |
|---|---|
| AE / DAE | 重建（加噪的）輸入 |
| **MAE** | 遮掉 75% 的 patch，重建被遮的部分 |
| BERT | 遮掉 15% 的 token，預測它們 |
| GPT | 預測下一個 token |

### 流派 B：對比式 —— 學「什麼是同一個東西」

| 方法 | 核心 |
|---|---|
| **SimCLR** | 大量負樣本 + InfoNCE |
| MoCo | momentum encoder + queue 存負樣本 |
| **BYOL** | ★ 不需要負樣本 |
| **SimSiam** | ★ 不需負樣本、不需 momentum，最簡單 |
| Barlow Twins | 讓兩視角的特徵相關矩陣接近單位矩陣 |

---

## §3 SimCLR ★（對比學習的代表作）

### 3.1 流程

```
一張圖 x
   ├─ 增強 t1 → x1 → encoder f → h1 → projector g → z1
   └─ 增強 t2 → x2 → encoder f → h2 → projector g → z2

目標：讓 (z1, z2) 這一對很像，跟 batch 內其他 2N−2 個都不像
```

### 3.2 InfoNCE / NT-Xent Loss

對第 `i` 個樣本（正樣本是 `j`）：

```
loss(i,j) = − log [ exp(sim(zi,zj)/T) / sum_{k!=i} exp(sim(zi,zk)/T) ]

sim(a,b) = a·b / (||a|| ||b||)        （餘弦相似度）
T = temperature，通常 0.1 ~ 0.5
```

**本質上就是一個 2N 類的分類問題**：從 2N−1 個候選中找出正樣本。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """z1, z2: (N, D)，第 i 列互為正樣本。"""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                    # (2N, D)
    z = F.normalize(z, dim=1)                         # ★ 一定要 L2 正規化

    sim = z @ z.t() / temperature                     # (2N, 2N)

    # ★ 把對角線（自己跟自己）遮掉
    eye = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # 正樣本位置：i 的正樣本是 i+N（後半段則是 i-N）
    targets = torch.cat([torch.arange(N, 2 * N),
                         torch.arange(0, N)]).to(z.device)

    return F.cross_entropy(sim, targets)              # ★ 直接用 CE！
```

### 逐行拆解 `nt_xent_loss`

假設 `N = 3`（batch 裡 3 張圖，各做兩種增強）：

```
z 疊起來後是 6 列：
  列 0,1,2  = view1 的三張圖
  列 3,4,5  = view2 的三張圖
  第 0 列的正樣本是第 3 列（同一張圖的另一個視角）
```

**① `F.normalize(torch.cat([z1, z2], 0), dim=1)`**
- `torch.cat([z1, z2], 0)` 沿第 0 維接起來：`(3,D) + (3,D)` → `(6,D)`
  （`cat` 不增加維度數；如果用 `stack` 會變成 `(2,3,D)`，就錯了）
- `F.normalize(..., dim=1)` 把**每一列**除以自己的長度 → 變成單位向量
- **為什麼要正規化？** 這樣兩個向量的內積就直接等於餘弦相似度，
  不會因為某些向量比較「長」而佔便宜

**② `sim = z @ z.t() / temperature`**
- `z.t()` 是轉置（`(6,D)` → `(D,6)`），`@` 是矩陣乘法
- `(6,D) @ (D,6)` → `(6,6)`，`sim[i][j]` = 第 i 列跟第 j 列的相似度
- 除以 `temperature`（例如 0.5）等於把分數放大兩倍 → softmax 更尖銳、
  更專注在「最像的那個負樣本」上

**③ `sim.masked_fill(eye, float("-inf"))`**
- `torch.eye(6, dtype=torch.bool)` 是 6×6 的單位矩陣，對角線 True
- 對角線是「自己跟自己」，相似度必定是 1（最高），不排除掉答案就永遠是自己
- 填 `-inf` 而不是 0：因為之後要過 softmax，`exp(-inf) = 0` 才是真的排除

**④ `targets`**
- `torch.arange(N, 2*N)` = `[3,4,5]` → 第 0,1,2 列的答案分別是第 3,4,5 欄
- `torch.arange(0, N)` = `[0,1,2]` → 第 3,4,5 列的答案分別是第 0,1,2 欄
- `torch.cat` 接起來 → `[3,4,5,0,1,2]`，剛好是 6 列各自的正確答案

**⑤ `F.cross_entropy(sim, targets)`**
- `sim` 當 logits（每列 6 個分數）、`targets` 當標籤
- CrossEntropyLoss 內部會做 `softmax` + `取負 log`，正好就是 InfoNCE 的公式

> ✅ **整個對比學習被化約成一次 cross_entropy 呼叫**，這就是為什麼這段值得手抄三遍。

> ★ **這段程式碼值得手抄三遍。** 它把「對比學習」漂亮地化約成一個 cross-entropy，
> 是整個 SSL 領域最重要的實作技巧。

### 3.3 三個關鍵設計

| 設計 | 為什麼重要 |
|---|---|
| **強資料增強** | ★ SimCLR 最重要的發現：crop + color jitter 的組合是關鍵。增強太弱 → 模型靠顏色就能配對，學不到語意 |
| **Projection head** | 在 `h` 之後接 MLP 得到 `z`，**loss 算在 `z` 上，下游用 `h`**。因為 `z` 為了對比任務丟掉了一些資訊（例如顏色），那些資訊對下游可能有用 |
| **大 batch size** | 負樣本數 = 2N−2，越大越好（論文用 4096）。你的 8GB 卡用 256 是合理折衷 |

```python
class SimCLR(nn.Module):
    def __init__(self, encoder, feat_dim, proj_dim=128):
        super().__init__()
        self.encoder = encoder                        # 例如 ResNet-18 去掉 fc
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)                # ★ 下游用這個
        z = self.projector(h)                         # ★ 算 loss 用這個
        return h, z
```

### 3.4 SimCLR 的資料增強

```python
from torchvision import transforms as T

simclr_tfm = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),                    # ★ 最重要
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    # ★ 第二重要
    T.RandomGrayscale(p=0.2),                                     # 防止靠顏色作弊
    T.ToTensor(),
    T.Normalize(mean, std),
])

class TwoCropTransform:
    """回傳同一張圖的兩個不同增強。"""
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, x): return self.tfm(x), self.tfm(x)

ds = datasets.CIFAR10(root, train=True, transform=TwoCropTransform(simclr_tfm))
# 取出來會是 ((x1, x2), y)
```

---

## §4 BYOL / SimSiam：不需要負樣本

### 4.1 Collapse 問題

如果只要求「兩個視角要像」，模型有個作弊解：**對所有輸入都輸出同一個常數向量**。
loss = 0，但完全沒學到東西。這叫 **representation collapse**。

SimCLR 用負樣本避免它（因為要跟別人不像）。BYOL/SimSiam 用別的方法。

### 4.2 SimSiam（最簡潔的方案）★

```
x → t1 → f → g → z1 → h → p1
x → t2 → f → g → z2 → h → p2

loss = −cos(p1, stopgrad(z2))/2 − cos(p2, stopgrad(z1))/2

f = encoder, g = projector, h = predictor
```

```python
class SimSiam(nn.Module):
    def __init__(self, encoder, dim=2048, pred_dim=512):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
        )
        self.predictor = nn.Sequential(               # ★ 只有 predictor 這一支不對稱
            nn.Linear(dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim), nn.ReLU(),
            nn.Linear(pred_dim, dim),
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1).flatten(1))
        z2 = self.projector(self.encoder(x2).flatten(1))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()       # ★★ stop-gradient 就在這裡

def simsiam_loss(p1, p2, z1, z2):
    return -(F.cosine_similarity(p1, z2).mean() +
             F.cosine_similarity(p2, z1).mean()) * 0.5
```

> ★★ **`z1.detach()` 這一行就是整篇論文的核心。**
> 拿掉它，模型立刻 collapse。
> **為什麼有效**：stop-gradient 讓問題變成類似 EM 的交替最佳化 ——
> predictor 追著一個「暫時固定」的目標跑，避免兩邊同時往常數解坍縮。
>
> ✍️ **強烈建議動手驗證**：訓練時把 `.detach()` 拿掉，觀察 loss 迅速降到 −1
> 而 latent 的標準差降到 0。這就是 collapse 的樣子。

### 4.3 BYOL = SimSiam + momentum encoder

```python
@torch.no_grad()
def update_target(online, target, m=0.996):
    """target network 用 EMA 更新，不吃梯度。"""
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
```

### 4.4 監控 collapse

```python
# 輸出向量各維度的標準差。正常應接近 1/sqrt(d)，接近 0 就是 collapse
z_norm = F.normalize(z, dim=1)
print(f"std = {z_norm.std(dim=0).mean().item():.4f}  "
      f"(理想約 {1 / z.size(1) ** 0.5:.4f})")
```

---

## §5 MAE：Masked Autoencoder（跟主題 04 連起來）

```
1. 把圖片切成 patch
2. 隨機遮掉 75%（★ 比例極高是關鍵）
3. Encoder 只處理「可見的 25%」→ 計算量只有 1/4，訓練超快
4. 加入 mask token，用一個輕量 Decoder 重建被遮的像素
5. loss 只算「被遮住的 patch」的 MSE
```

**為什麼遮 75% 有效？** 圖片有大量空間冗餘。
遮 15%（像 BERT）太簡單，模型靠內插就能填回來，學不到語意。
遮 75% 逼模型必須理解「這是一隻貓」才能重建。

```python
def random_masking(x, mask_ratio=0.75):
    """x: (B, L, D) patch embeddings"""
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, D))
    mask = torch.ones(B, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)        # (B, L)，1 表示被遮
    return x_kept, mask, ids_restore
```

---

## §6 下游評估協定 ★（做研究一定要照規矩來）

| 協定 | 做法 | 在評估什麼 |
|---|---|---|
| **Linear probing** | 凍結 encoder，只訓練一個 `nn.Linear` | ★ 表徵的**線性可分性**，最能反映表徵品質 |
| **Fine-tuning** | 全部解凍，小 lr 訓練 | 實務上限，但混入了模型容量的影響 |
| **k-NN 評估** | 完全不訓練，用 latent 做 k-NN | 最快，適合訓練中途監控 |
| **Semi-supervised** | 只用 1% / 10% 標籤 fine-tune | ★ SSL 最有說服力的場景 |

```python
# Linear probing 標準寫法
for p in encoder.parameters():
    p.requires_grad_(False)
encoder.eval()                       # ★ 別忘了，否則 BN running stats 會被污染

clf = nn.Linear(feat_dim, n_classes).to(device)
opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)

for x, y in loader:
    with torch.no_grad():
        h = encoder(x.to(device)).flatten(1)
    loss = F.cross_entropy(clf(h), y.to(device))
    opt.zero_grad(); loss.backward(); opt.step()
```

```python
# k-NN 評估（不用訓練，快速看表徵好不好）
@torch.no_grad()
def knn_eval(encoder, train_loader, test_loader, device, k=20, T=0.1):
    encoder.eval()
    feats, labels = [], []
    for x, y in train_loader:
        feats.append(F.normalize(encoder(x.to(device)).flatten(1), dim=1).cpu())
        labels.append(y)
    Ftr = torch.cat(feats); ytr = torch.cat(labels)
    n_cls = int(ytr.max()) + 1

    correct = total = 0
    for x, y in test_loader:
        f = F.normalize(encoder(x.to(device)).flatten(1), dim=1).cpu()
        sim = f @ Ftr.t()
        topk_sim, topk_idx = sim.topk(k, dim=1)
        cand = ytr[topk_idx]                              # (B, k)
        w = (topk_sim / T).exp()
        scores = torch.zeros(len(f), n_cls).scatter_add_(1, cand, w)
        correct += (scores.argmax(1) == y).sum().item()
        total += len(y)
    return correct / total
```

---

## §7 SSL 用在你的時序資料（研究方向）

**增強策略**（相當於影像的 crop + jitter）：

```python
def ts_augment(x):                       # x: (L, F)
    x = x + 0.02 * torch.randn_like(x)                        # jitter
    x = x * (1 + 0.1 * torch.randn(1, x.size(-1)))            # scaling
    m = torch.rand(x.size(0)) < 0.15                          # 隨機遮蔽時間步
    x = x.clone(); x[m] = 0
    return x
```

**專案設計（W25+ 的研究題目）：**

```
1. 用 2330 + 2344 + 8028 的全部歷史（無標籤）做 SimCLR 預訓練
2. Encoder 用你現有的 CNN+Transformer 架構
3. Linear probe 評估：凍結 encoder，只訓練分類頭
4. 對照組：從零訓練同樣的架構
5. 加碼實驗：只用 10% 的標籤，看 SSL 的優勢是否放大
```

如果 SSL 版本在「少標籤情境」明顯勝出，這就是一篇有內容的大專生研究。

---

## §8 動手練習

1. 手刻 `nt_xent_loss`，用 `(4, 8)` 隨機輸入驗證 loss 是合理正數。
2. 訓練 SimCLR（CIFAR-10，100 epoch），用 k-NN 評估監控進度。
3. 做 linear probing，跟「隨機初始化 encoder + linear」比較。
4. 只用 10% 標籤 fine-tune，比較 SSL 預訓練 vs 從零訓練。
5. **拿掉 SimSiam 的 `.detach()`**，畫出 z 的 std 曲線觀察 collapse。
6. 用 t-SNE 畫出預訓練前後的 latent space，看類別有沒有自然分開。
7. 把增強從「crop + jitter」減弱成「只有 flip」，觀察效果掉多少。

---

## ✅ 自我檢核

- [ ] 說出 SSL 的兩階段流程與兩大流派
- [ ] 手寫 InfoNCE / NT-Xent 的 PyTorch 實作
- [ ] 說出 SimCLR 為何需要 projection head，下游該用 `h` 還是 `z`
- [ ] 說出 representation collapse 是什麼，SimCLR 和 SimSiam 各怎麼避免
- [ ] 解釋 SimSiam 的 stop-gradient 為何是關鍵
- [ ] 說出 MAE 為何要遮 75% 而不是 15%
- [ ] 說出 linear probing 和 fine-tuning 各在評估什麼
- [ ] 說出如何監控 collapse
