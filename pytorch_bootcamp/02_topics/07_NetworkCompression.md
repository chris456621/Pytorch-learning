# 主題 07 · Network Compression 模型壓縮

> **對應**：李宏毅 Network Compression 篇 · Deep Compression / KD / Lottery Ticket
> **實作**：`03_code/09_compression.py`


> 📘 **看不懂程式碼裡的語法？**
> 先查 [`01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)（Python 語法解碼器）。
> 猶豫「這兩個寫法差在哪」？查 [`01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

---

## §1 為什麼要壓縮

一個 ResNet-50 有 2500 萬參數、100 MB、單張推論 4 GFLOPs。
要放進手機、嵌入式裝置或做即時推論，都撐不住。

| 技術 | 減少什麼 | 難度 | 實際加速 |
|---|---|---|---|
| **剪枝 Pruning** | 參數量 | 中 | 結構化才有 |
| **量化 Quantization** | 每個參數的位元數 | 中 | ★ 有（2-4x） |
| **知識蒸餾 KD** | 用小模型取代大模型 | 低 | ★ 有 |
| **架構設計** | 一開始就設計小模型 | 高 | ★ 有 |

---

## §2 剪枝 Pruning

### 2.1 非結構化 vs 結構化 ★（最關鍵的區別）

| | 非結構化 | 結構化 |
|---|---|---|
| 剪什麼 | 個別權重（設成 0） | 整個 channel / filter / head |
| 稀疏度 | 可到 90%+ | 通常 30-50% |
| 精度損失 | 小 | 較大 |
| **實際加速** | ❌ **沒有**（除非硬體支援稀疏運算） | ✅ **有**（模型真的變小） |

> 🔥 **最重要的觀念**：
> 非結構化剪枝把權重設成 0，但矩陣的**形狀沒變**，
> GPU 還是照樣做 `1000×1000` 的矩陣乘法（只是很多項是 0 乘 x）。
> **所以「稀疏度 90%」不等於「快 10 倍」。**
> 想真的加速，必須做結構化剪枝，把整個 channel 拿掉。

### 2.2 PyTorch 內建剪枝 API

```python
import torch.nn as nn
import torch.nn.utils.prune as prune

# 非結構化：把該層 30% 最小的權重歸零
prune.l1_unstructured(model.conv1, name="weight", amount=0.3)

# 結構化：整個 output channel 剪掉（dim=0）
prune.ln_structured(model.conv1, name="weight", amount=0.5, n=2, dim=0)

# 全域剪枝（★ 更好：跨層一起排序，不重要的層自然剪多一點）
params = [(m, "weight") for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.5)

# ★ 剪枝後 weight 是 weight_orig * weight_mask 算出來的
#    要「固定」下來必須 remove
for m, n in params:
    prune.remove(m, n)          # 把 mask 永久套用並移除 hook

def sparsity(model):
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    return zeros / total
```

### 2.3 Iterative Magnitude Pruning ★

```
訓練 → 剪 20% → fine-tune → 剪 20% → fine-tune → ... → 目標稀疏度
```

**一次剪 80% 效果會很差，分五次每次 20% 效果好很多。**

```python
for round_i in range(5):
    prune.global_unstructured(params, prune.L1Unstructured, amount=0.2)
    for ep in range(fine_tune_epochs):
        train_one_epoch(model, ...)
    print(f"round {round_i}: sparsity {sparsity(model):.2%}, acc {evaluate(...):.4f}")
```

### 2.4 Lottery Ticket Hypothesis（觀念很酷）

**假說**：一個隨機初始化的大網路裡，存在一個小子網路（「中獎彩券」），
如果**用原本的初始值**單獨訓練它，能達到跟大網路差不多的效果。

```
1. 隨機初始化 theta_0，存起來
2. 訓練到收斂 → theta_T
3. 剪掉 |theta_T| 最小的 p%，得到 mask
4. ★ 把剩下的權重「重置回 theta_0」（不是隨機重來，是回到最初的值）
5. 用同一個 mask 重新訓練 → 效果應該不輸原網路
```

> ★ 關鍵在第 4 步。如果重置成**新的**隨機值，效果會差很多 ——
> 這說明「初始值」和「結構」是配對的。

---

## §3 量化 Quantization

### 3.1 原理

把 float32 換成 int8：**檔案小 4 倍，整數運算快 2-4 倍**。

```
量化：   q = round(x / scale) + zero_point
反量化： x_hat = (q − zero_point) * scale

對稱量化（zero_point = 0）:  scale = max|x| / 127
非對稱量化: scale = (max − min) / 255,  zero_point = round(−min / scale)
```

```python
import torch

def quantize_tensor(x, num_bits=8):
    qmin, qmax = 0, 2 ** num_bits - 1
    mn, mx = x.min().item(), x.max().item()
    scale = (mx - mn) / (qmax - qmin)
    zero_point = round(qmin - mn / scale)
    q = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
    return q.to(torch.uint8), scale, zero_point

def dequantize(q, scale, zero_point):
    return (q.float() - zero_point) * scale
```

### 3.2 PTQ vs QAT ★

| | Post-Training Quantization | Quantization-Aware Training |
|---|---|---|
| 何時做 | 訓練完之後 | 訓練過程中 |
| 需要資料 | 少量校準資料 | 完整訓練集 |
| 成本 | 低（幾分鐘） | 高（要重訓） |
| 精度損失 | 中（1-3%） | ★ 小（< 1%） |
| 何時用 | ★ 先試這個 | PTQ 掉太多才用 |

```python
# 動態量化（最簡單，適合 Linear / LSTM 為主的模型）
qmodel = torch.ao.quantization.quantize_dynamic(
    model.cpu().eval(), {nn.Linear, nn.LSTM}, dtype=torch.qint8)

import os
def model_size_mb(m, path="tmp.pt"):
    torch.save(m.state_dict(), path)
    return os.path.getsize(path) / 1e6
```

> ⚠️ PyTorch 的量化推論**主要在 CPU 上**（fbgemm / qnnpack backend）。
> GPU 的 int8 要走 TensorRT。在筆電做實驗時記得 `.cpu()`。

### 3.3 QAT 的核心：Straight-Through Estimator

量化的 `round()` 導數處處為 0，梯度傳不回去。
解法就是 `01_syntax/07_autograd.md` §8 教的那個 trick：

```python
class FakeQuantize(nn.Module):
    """訓練時模擬量化誤差，但梯度直接穿透。"""
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):
        if not self.training:
            return x
        qmax = 2 ** (self.num_bits - 1) - 1
        scale = x.abs().max() / qmax
        xq = torch.clamp((x / scale).round(), -qmax - 1, qmax) * scale
        return x + (xq - x).detach()      # ★★ STE：forward 值是 xq，backward 梯度是 1
```

---

## §4 知識蒸餾 Knowledge Distillation ★（最實用）

### 4.1 核心想法

大模型（teacher）的 **soft targets** 比 one-hot 標籤含有更多資訊。

```
硬標籤：  [0, 0, 1, 0]              「這是貓」
軟標籤：  [0.02, 0.05, 0.88, 0.05]  「這是貓，但有點像狗，完全不像車」
                    ↑ 這個「暗知識 dark knowledge」是額外的監督訊號
```

### 4.2 Loss

```
L = alpha * T² * KL( softmax(z_s/T) || softmax(z_t/T) ) + (1−alpha) * CE(z_s, y)
      └─────────── 蒸餾項（學 teacher）───────────┘      └── 學真實標籤 ──┘
```

```python
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    # ★ KLDivLoss 第一個參數要 log_softmax，第二個要 softmax
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)                       # ★★ 一定要乘 T 平方
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
```

### 逐行拆解 `kd_loss`

```python
soft = F.kl_div(
    F.log_softmax(student_logits / T, dim=1),   # ← 第一個參數：log 機率
    F.softmax(teacher_logits / T, dim=1),       # ← 第二個參數：一般機率
    reduction="batchmean",
)
```

**三個容易寫錯的地方：**

1. **兩個參數的形式不對稱**。`F.kl_div` 的第一個參數要**已經取過 log** 的，
   第二個要**沒取 log** 的。寫反了不會報錯，但算出來的東西沒有意義。
2. **`/ T` 要對兩邊都做**。T 是 temperature，除以它讓分布變平緩，
   把「貓有點像狗」這種暗知識放大出來。只除一邊等於在比兩個不同溫度的分布。
3. **`reduction="batchmean"` 不是 `"mean"`**。
   `"mean"` 會除以「batch × 類別數」，那不是 KL 散度的定義；
   `"batchmean"` 只除以 batch，才是數學上正確的。

```python
hard = F.cross_entropy(student_logits, labels)      # ★ 這裡不除 T
return alpha * soft + (1 - alpha) * hard
```
- 硬標籤那一項用**原始的** logits（不除 T），因為它學的是真實答案
- `alpha` 控制「聽 teacher 的」和「看標準答案」的比重，常用 0.5~0.9

> ★★ **為什麼要乘 T²？**
> `softmax(z/T)` 對 `z` 微分會多出一個 `1/T` 因子，KL 兩邊都被縮放，
> 導致蒸餾項的梯度大約是原本的 `1/T²`。乘回 `T²` 讓蒸餾項和硬標籤項的
> 梯度量級相當，這樣 `alpha` 才是有意義的權重。
> **這個理由是正確的，乘 T² 也是文獻與各家實作的標準做法。**
>
> ⚠️ **但它是「尺度校正」，不是「效果加成」。**

### 4.2.1 一個真實的實驗教訓（★ 這節比公式重要）

我在你的機器上實測了兩次（MNIST，student 只有 7K 參數，**每個設定都各自掃過 lr 取最佳**）：

| teacher 準確率 | student 從零訓練 | KD 乘 T² | KD 不乘 T² |
|---|---|---|---|
| 93.1%（弱 teacher） | 0.9154 | **0.9149** | 0.8865 |
| 98.7%（強 teacher） | 0.9794 | 0.9788 | **0.9832** |

**兩次的結論方向完全相反。**

這不是程式有 bug，而是：

1. **單一 seed 的差異（0.4%~1%）跟這個比較的效果量是同一個量級。**
   換句話說，這兩次實驗**都沒有統計效力**，任何一次拿來下結論都是錯的。
2. 要得到可信的結論，必須 **跑 3~5 個 seed 報告 mean ± std**，
   而且 lr 網格要夠密（我只掃了 2 個點）。

> 🔥 **這正是為什麼 `01_syntax/11_訓練迴圈範本.md` §8 一直強調多 seed。**
> 我自己第一次跑完就寫下「T² 勝出 +2.8%」，第二次換個 teacher 就被推翻。
> **你在讀論文和寫論文時，都要對「只跑一次就宣稱改進 1%」保持警覺。**

**那到底要不要乘 T²？** 要 —— 因為它的理論依據（梯度量級對齊）是對的，
且所有主流實作都這樣寫，不乘會讓你的 `alpha` 失去意義、無法跟別人比較。
但你要知道：**改了 T 或 T² 就要重新調 lr**，而且它不會自動帶來準確率提升。

`03_code/09_compression.py --kd` 已經改成「每個設定各自掃 lr 取最佳」的公平比較。
**動手練習**：把它加上 `--seeds 5` 的迴圈，看看誤差棒有多大。

> 📌 **另一個觀察**：上表兩次的 KD 都沒有明顯贏過「從零訓練」。
> **KD 的增益來自「teacher 比 student 強很多」，而且 student 要有足夠容量吸收。**
> 7K 參數的 student 在 MNIST 上本來就已接近上限（97.9%），沒有多少空間給 KD 發揮。
> 想看到教科書等級的 KD 增益，要用**任務更難、student 容量落差更明顯**的設定
> （例如 CIFAR-100、ResNet-34 蒸餾到 ResNet-8）。

### 4.3 訓練迴圈

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)           # ★ teacher 完全不更新

for x, y in loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():             # ★ teacher 的 forward 不建圖
        t_logits = teacher(x)
    s_logits = student(x)
    loss = kd_loss(s_logits, t_logits, y, T=4.0, alpha=0.7)
    opt.zero_grad(); loss.backward(); opt.step()
```

### 4.4 蒸餾的變體

| 方法 | 蒸餾什麼 |
|---|---|
| Logit KD（原始） | 輸出機率分布 |
| **Feature KD** | 中間層的特徵圖（用 hook 抓） |
| Attention Transfer | attention map |
| Relational KD | 樣本之間的關係 |
| **Self-distillation** | teacher = student 自己的前一版（不用大模型！） |

```python
# Feature KD：用 hook 抓中間特徵
feats = {}
def hook(name):
    def fn(m, i, o): feats[name] = o
    return fn
teacher.layer3.register_forward_hook(hook("t3"))
student.layer3.register_forward_hook(hook("s3"))
# loss 加上：F.mse_loss(adapter(feats["s3"]), feats["t3"].detach())
#            adapter 是 1x1 conv，用來對齊 channel 數
```

---

## §5 完整壓縮流程（W21 的作業）

```
1. 訓練大模型（teacher），例如 ResNet-18 → CIFAR-10 準確率 94%
2. 設計小 student（1/4 寬度），從零訓練 → 89%          ← baseline
3. 用 KD 訓練同一個 student                          → 92%   ★ 進步
4. 對 student 做 iterative pruning 到 50% 稀疏        → 91%
5. 做 PTQ 量化到 int8                                → 90.5%
6. 報告：參數量、檔案大小、CPU 推論延遲、準確率
```

**評估表格範本：**

| 模型 | 參數量 | 大小(MB) | CPU延遲(ms) | Acc |
|---|---|---|---|---|
| Teacher ResNet-18 | 11.2M | 44.8 | 38.2 | 94.1% |
| Student 從零訓練 | 0.7M | 2.8 | 6.1 | 89.3% |
| Student + KD | 0.7M | 2.8 | 6.1 | 92.0% |
| + Pruning 50% | 0.7M(稀疏) | 2.8 | 6.1 | 91.4% |
| + INT8 量化 | 0.7M | **0.7** | **2.4** | 90.6% |

```python
# 測量推論延遲的正確方法
import time
model.eval()
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    for _ in range(10):
        model(x)                        # ★ warmup，第一次很慢不能算
    t0 = time.perf_counter()
    for _ in range(100):
        model(x)
    print(f"{(time.perf_counter() - t0) / 100 * 1000:.2f} ms")
```

---

## §6 高效架構設計（從源頭壓縮）

| 技巧 | 參數量變化 | 用在 |
|---|---|---|
| Depthwise separable conv | ÷ 8~9 | MobileNet |
| Bottleneck（1×1 降維再升維） | ÷ 3~4 | ResNet-50+ |
| Group convolution | ÷ g | ResNeXt |
| 低秩分解（W ≈ UV） | 依 rank | 全連接層壓縮 |
| Global Average Pooling 取代大 FC | ÷ 100+ | ★ 所有現代 CNN |

```python
# 低秩分解：把 Linear(1024, 1024) 拆成兩層
class LowRankLinear(nn.Module):
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.u = nn.Linear(in_f, rank, bias=False)
        self.v = nn.Linear(rank, out_f)
    def forward(self, x):
        return self.v(self.u(x))
# 參數量：1024*1024 = 1M → 1024*64 + 64*1024 = 131K（rank=64）
```

---

## §7 動手練習

1. 對你的 `CNN.py` 做非結構化剪枝到 80%，測量稀疏度、檔案大小、推論時間、準確率，
   **並解釋為何推論時間沒變**。
2. 改做結構化剪枝，觀察推論時間的差異。
3. 實作 iterative pruning（5 輪 × 20%），跟「一次剪 80%」比較。
4. 用 `quantize_dynamic` 量化，比較大小與 CPU 延遲。
5. 手刻 `quantize_tensor` / `dequantize`，測量量化誤差。
6. 實作 KD，用你的 `Transformer.py` 當 teacher 蒸餾一個小模型。
7. **故意拿掉 `T²`**，用 5 個 seed 各自掃 lr，報告 mean ± std，
   然後回答：這個差異在統計上站得住腳嗎？
8. 做出 §5 的完整表格，這可以直接當成一份研究報告。

---

## ✅ 自我檢核

- [ ] 說出非結構化剪枝為何「檔案變小但速度沒變快」
- [ ] 說出 iterative pruning 為何優於一次剪完
- [ ] 解釋 Lottery Ticket Hypothesis 的第 4 步為何是關鍵
- [ ] 寫出量化的 scale / zero_point 公式
- [ ] 說出 PTQ 和 QAT 的差別與各自適用時機
- [ ] 寫出 KD loss，並解釋為何要乘 T²（理論依據），以及為何它不保證提升準確率
- [ ] 說出「單次實驗差 1% 就宣稱改進」錯在哪裡
- [ ] 說出 soft target 為何比 hard label 含更多資訊
- [ ] 說出 QAT 用什麼技巧讓 round() 可微
