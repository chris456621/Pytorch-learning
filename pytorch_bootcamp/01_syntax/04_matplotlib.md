# 04 · Matplotlib 視覺化（深度學習專用）

> 只教你在訓練模型時**真正會用到**的圖。學會這 8 種圖就夠應付研究了。

## §0 兩種 API，只用一種

```python
import matplotlib.pyplot as plt

# ❌ pyplot 風格（隱含當前圖，多圖時會混亂）
plt.plot(x, y); plt.title("a"); plt.show()

# ✅ 物件導向風格（★ 一律用這個）
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y)
ax.set_title("a")
fig.tight_layout()
plt.show()
```

### 中文顯示（Windows）

```python
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 微軟正黑體
matplotlib.rcParams["axes.unicode_minus"] = False                # 讓負號正常顯示
```

> 找不到字型時可依序 fallback：`Microsoft YaHei`、`SimHei`、`DejaVu Sans`。
> 建議：**圖表標題和軸標籤一律用英文**，投稿論文時不用重畫。

---

## §1 訓練曲線 ★（最重要，每次訓練都要畫）

```python
def plot_history(history, save_path=None):
    """history 是一個 dict，key 有 train_loss / val_loss / val_acc / lr，值都是 list。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(ep, history["val_acc"], color="green")
    best = max(history["val_acc"]); bi = history["val_acc"].index(best) + 1
    axes[1].scatter([bi], [best], color="red", zorder=5)
    axes[1].annotate(f"best {best:.4f} @ep{bi}", (bi, best),
                     textcoords="offset points", xytext=(5, -12))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Val Accuracy"); axes[1].grid(alpha=.3)

    axes[2].plot(ep, history["lr"], color="orange")
    axes[2].set_yscale("log")                 # ★ lr 一定要用 log 軸
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate"); axes[2].grid(alpha=.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

**怎麼「讀」訓練曲線（★ 這才是重點）：**

| 曲線形狀 | 診斷 | 處方 |
|---|---|---|
| train 降、val 也降 | 正常 | 繼續訓練 |
| train 降但 val 開始上升 | **過擬合** | 加 dropout / weight decay / 資料增強 / early stop |
| train 和 val 都很高且不降 | **欠擬合** | 加大模型 / 提高 lr / 訓練更久 / 檢查特徵 |
| loss 劇烈震盪 | lr 太大 或 batch 太小 | 降 lr / 加大 batch / 用 grad clip |
| loss 一開始就 nan | 數值爆炸 | 降 lr / 檢查輸入有無 nan / 檢查 log(0) |
| loss 完全水平不動 | 沒接上梯度 | 檢查 zero_grad / backward / requires_grad |
| val 比 train 還低 | 有 dropout（正常）或資料洩漏（危險） | 檢查切分方式 |

---

## §2 混淆矩陣

```python
import numpy as np

def confusion_matrix(y_true, y_pred, n_cls):
    return np.bincount(y_true * n_cls + y_pred, minlength=n_cls**2).reshape(n_cls, n_cls)

def plot_confusion(cm, class_names, normalize=True):
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i,j]:.2f}" if normalize else f"{cm[i,j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    return fig
```

> ★ **一定要看正規化後的版本**。類別不平衡時，未正規化的混淆矩陣會騙人：
> 模型全部猜多數類別也能有漂亮的對角線。

---

## §3 顯示圖片與一批樣本

```python
import torch

def show_batch(images, labels=None, class_names=None, n=16, mean=None, std=None):
    """images 形狀為 (B, C, H, W) 的 tensor。mean/std 用來反正規化。"""
    imgs = images[:n].detach().cpu()
    if mean is not None:
        m = torch.tensor(mean).view(1, -1, 1, 1)
        s = torch.tensor(std).view(1, -1, 1, 1)
        imgs = imgs * s + m
    imgs = imgs.clamp(0, 1)

    rows = int(np.ceil(n ** 0.5))
    fig, axes = plt.subplots(rows, rows, figsize=(rows * 1.6, rows * 1.6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(imgs):
            continue
        img = imgs[i]
        img = img.squeeze(0) if img.shape[0] == 1 else img.permute(1, 2, 0)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if labels is not None:
            name = class_names[labels[i]] if class_names else str(labels[i].item())
            ax.set_title(name, fontsize=8)
    fig.tight_layout()
    return fig
```

> ★ 關鍵一行：`img.permute(1, 2, 0)`
> PyTorch 用 `(C, H, W)`，matplotlib 要 `(H, W, C)`。忘記轉就會看到一片亂碼。

torchvision 的懶人版：

```python
from torchvision.utils import make_grid
grid = make_grid(images[:64], nrow=8, normalize=True, value_range=(-1, 1))
plt.imshow(grid.permute(1, 2, 0).cpu())
```

---

## §4 特徵圖視覺化（用 forward hook）

```python
feats = {}

def make_hook(name):
    def fn(module, inp, out):
        feats[name] = out.detach()
    return fn

h = model.features[0].register_forward_hook(make_hook("conv1"))
_ = model(x[:1])
h.remove()                       # ★ 用完一定要移除，否則會一直累積佔記憶體

fmap = feats["conv1"][0]         # (C, H, W)
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.axis("off")
    if i < fmap.shape[0]:
        ax.imshow(fmap[i].cpu(), cmap="viridis")
fig.suptitle("conv1 feature maps")
```

---

## §5 分布與統計圖

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 直方圖：看權重或梯度分布（診斷梯度消失／爆炸）
axes[0].hist(w.detach().cpu().numpy().ravel(), bins=60, alpha=.7)
axes[0].set_title("Weight distribution")

# 箱型圖：比較多組實驗（★ 報告多 seed 結果就用這個）
axes[1].boxplot([accs_a, accs_b, accs_c], tick_labels=["A", "B", "C"])
axes[1].set_ylabel("Val Acc"); axes[1].set_title("5 seeds")

# 散佈圖：預測 vs 真實（迴歸任務）
axes[2].scatter(y_true, y_pred, s=6, alpha=.4)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
axes[2].plot(lims, lims, "r--")
axes[2].set_xlabel("True"); axes[2].set_ylabel("Pred")

fig.tight_layout()
```

---

## §6 latent space 降維視覺化（AE / SSL 主題會用到）★

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

Z = embeddings.detach().cpu().numpy()        # (N, D)
y = labels.cpu().numpy()

Z2 = PCA(n_components=2).fit_transform(Z)                                 # 快
# Z2 = TSNE(n_components=2, perplexity=30, init="pca").fit_transform(Z)   # 慢但漂亮

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=y, cmap="tab10", s=6, alpha=.7)
fig.colorbar(sc, ax=ax)
ax.set_title("Latent space (PCA)")
```

> ★ 評估 Self-supervised 學到的表徵好不好，這張圖是最直觀的證據：
> **好的表徵會讓同類別的點自然聚在一起**，即使訓練時完全沒用到標籤。

---

## §7 時序資料三聯圖（給你的股票資料）

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})

axes[0].plot(df.index, df["Close"], lw=.8, label="Close")
axes[0].plot(df.index, df["Close"].rolling(20).mean(), lw=1, label="MA20")
up = df["label"] == 1
axes[0].scatter(df.index[up], df.loc[up, "Close"], s=8, c="red", label="up", zorder=5)
axes[0].legend(); axes[0].set_ylabel("Price")

axes[1].bar(df.index, df["Volume"], width=1)
axes[1].set_ylabel("Volume")

axes[2].plot(df.index, df["K"], lw=.8, label="K")
axes[2].plot(df.index, df["D"], lw=.8, label="D")
axes[2].axhline(80, ls="--", c="gray", lw=.6)
axes[2].axhline(20, ls="--", c="gray", lw=.6)
axes[2].legend(); axes[2].set_ylabel("KD")

fig.tight_layout()
```

---

## §8 存檔與論文品質設定

```python
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,          # ★ 投稿要 300 dpi 以上
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,    # 去掉上、右框線比較好看
    "axes.spines.right": False,
    "legend.frameon": False,
})

fig.savefig("fig1.png", dpi=300, bbox_inches="tight")
fig.savefig("fig1.pdf", bbox_inches="tight")   # ★ 向量圖，論文首選
plt.close(fig)                                 # ★ 迴圈裡畫圖一定要 close，否則記憶體爆掉
```

在 VS Code 的 `.py` 檔看圖：用 `# %%` 分格執行，圖會顯示在互動視窗。
純腳本執行時最後要 `plt.show()`（會阻塞），或直接 `savefig` 不顯示。

---

## §9 動手練習

1. 把 `plot_history` 放進 `03_code/00_common.py`，之後每次訓練都呼叫它。
2. 故意把 lr 設成 10 訓練 5 epoch，畫出曲線，**記住「爆炸」長什麼樣**。
3. 故意用 20 筆資料訓練 200 epoch，畫出曲線，**記住「過擬合」長什麼樣**。
4. 畫出 2330 的價格 + MA20 + KD 三聯圖。
5. 對你 `Transformer.py` 的第一層 CNN 做特徵圖視覺化。

---

## ✅ 自我檢核

- [ ] 只用物件導向 API（`fig, ax = plt.subplots()`）寫圖
- [ ] 看到一組 loss 曲線能立刻說出「過擬合 / 欠擬合 / lr 太大」
- [ ] 說出為何顯示 tensor 圖片要 `permute(1,2,0)`
- [ ] 說出為何迴圈裡畫圖要 `plt.close(fig)`
- [ ] 說出為何 learning rate 曲線要用 log 軸
