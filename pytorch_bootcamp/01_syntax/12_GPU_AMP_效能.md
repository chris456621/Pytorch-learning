# 12 · GPU / 混合精度 / 效能優化

> 針對你的硬體（**RTX 4060 Laptop 8GB, Ada 架構, CUDA 12.6, torch 2.12**）量身寫的。
> 我已實測確認：你的卡**支援 bfloat16**，所以 AMP 可以用最好的設定。

---

## §1 device 基本操作

```python
import torch

torch.cuda.is_available()                       # True
torch.cuda.device_count()                       # 1
torch.cuda.get_device_name(0)                   # NVIDIA GeForce RTX 4060 Laptop GPU
torch.cuda.is_bf16_supported()                  # True（你的卡支援）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 記憶體
torch.cuda.memory_allocated() / 1e9             # 目前 tensor 佔用 (GB)
torch.cuda.memory_reserved() / 1e9              # PyTorch 向驅動要的總量
torch.cuda.max_memory_allocated() / 1e9         # ★ 峰值，調 batch size 靠它
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()                        # 把快取還給系統（不會釋放正在用的）
```

**測量峰值記憶體的正確方式：**

```python
torch.cuda.reset_peak_memory_stats()
loss = criterion(model(x), y); loss.backward()
print(f"峰值 {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

---

## §2 混合精度 AMP ★（一定要開）

**效果**：速度提升 1.5～3 倍，記憶體省 30～50%，準確率幾乎不變。

```python
# ★ torch 2.x 的新 API（舊的 torch.cuda.amp.* 已棄用）
from torch.amp import autocast, GradScaler

# ---- bfloat16 版本（★ 你的 4060 支援，推薦這個）----
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    loss.backward()                       # ★ bf16 不需要 GradScaler
    optimizer.step()

# ---- float16 版本（舊卡用，需要 GradScaler 防梯度下溢）----
scaler = GradScaler("cuda")
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                                  # 要裁剪梯度時先反縮放
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### bf16 vs fp16 該選哪個

| | float16 | bfloat16 |
|---|---|---|
| 尾數位元 | 10 | 7（精度較低） |
| 指數位元 | 5 | 8（動態範圍同 float32） |
| 需要 GradScaler | ✅ 需要 | ❌ 不需要 |
| 容易 overflow/nan | 較容易 | 很少 |
| 你的 4060 | 支援 | ★ **支援，選這個** |

> ★ **結論：你直接用 bfloat16，不用管 GradScaler，程式碼更簡單也更穩。**

### AMP 的注意事項

```python
# 1. 驗證時也可以開，但不需要 scaler
@torch.no_grad()
def evaluate(...):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)

# 2. 某些運算要強制用 fp32（數值敏感）
with autocast(device_type="cuda", enabled=False):
    loss = my_numerically_sensitive_loss(logits.float(), y)

# 3. loss 記錄時轉回 float
total_loss += loss.float().item()
```

---

## §3 torch.compile（免費加速）

```python
model = MyModel().to(device)
model = torch.compile(model)         # ★ 一行加速 10~50%

# 常用模式
model = torch.compile(model, mode="reduce-overhead")   # 小模型/小 batch 用
model = torch.compile(model, mode="max-autotune")      # 編譯久但最快
```

> ⚠️ 注意事項：
> - **第一個 batch 會很慢**（編譯中），之後才會加速
> - 輸入 shape 一變就會重新編譯 → 用 `drop_last=True` 讓 batch 固定
> - Windows 上支援度不如 Linux，如果報錯就先不要用
> - 存 checkpoint 時要存 `model._orig_mod.state_dict()`，或先 compile 再 load

---

## §4 記憶體不夠（OOM）時的處方，依序試

```
1. 降 batch_size                              最直接
2. 開 AMP（bfloat16）                          省 30~50%
3. 梯度累積模擬大 batch                          見 11 章 §6
4. optimizer.zero_grad(set_to_none=True)      省一份梯度的記憶體（2.x 已是預設）
5. 驗證迴圈加 @torch.no_grad()                  ★ 最常被漏掉
6. 用 loss.item() 不要 loss                     ★ 最常見的洩漏
7. gradient checkpointing                     用時間換記憶體，見下
8. 降低模型寬度 / 輸入解析度
```

### Gradient Checkpointing（記憶體換時間，最後手段）

```python
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x):
        return self.mlp(self.attn(x))

# 在 forward 裡用
def forward(self, x):
    for blk in self.blocks:
        x = checkpoint(blk, x, use_reentrant=False)   # ★ 不存中間激活，backward 時重算
    return x
```

省下 ~60% 記憶體，慢 ~30%。訓練大 Transformer 時很有用。

### 診斷記憶體洩漏

```python
# 症狀：每個 epoch 記憶體都增加一點，跑幾個 epoch 後 OOM
# 頭號兇手：
total_loss += loss                    # ❌ 抓住整張圖
outputs.append(logits)                # ❌ 沒 detach
self.history.append(feat)             # ❌ 同上

# ✅
total_loss += loss.item()
outputs.append(logits.detach().cpu())
```

---

## §5 速度優化清單

```python
# 1. cudnn benchmark（輸入尺寸固定時）★ 免費 5~15%
torch.backends.cudnn.benchmark = True

# 2. TF32（Ampere 以後的卡，你的 4060 支援）★ 矩陣乘法加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 3. 設定 float32 矩陣乘法精度（torch 2.x 的新寫法）
torch.set_float32_matmul_precision("high")     # 'highest' | 'high' | 'medium'

# 4. DataLoader
DataLoader(ds, num_workers=4, pin_memory=True, persistent_workers=True,
           prefetch_factor=2, drop_last=True)
x = x.to(device, non_blocking=True)

# 5. 用 channels_last 記憶體格式（CNN 專用，配合 AMP 效果好）
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)

# 6. Transformer 用內建的 SDPA（會自動選 FlashAttention）★
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
# 比手刻的 softmax(QK^T/sqrt(d))V 快很多且省記憶體
```

**建議放在每個訓練腳本開頭的樣板：**

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

---

## §6 找出瓶頸

```python
# 1. 簡單計時法：分別測資料和計算
import time
t0 = time.perf_counter()
for x, y in loader: pass
t_data = time.perf_counter() - t0
print(f"純讀資料 {t_data:.1f}s")
# 跟完整 epoch 時間比較：接近 → 瓶頸在資料；差很多 → 瓶頸在 GPU

# 2. GPU 計時要同步！
torch.cuda.synchronize()          # ★ CUDA 是非同步的，不同步會測到假時間
t0 = time.perf_counter()
out = model(x)
torch.cuda.synchronize()
print(time.perf_counter() - t0)

# 3. PyTorch Profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(10):
        loss = criterion(model(x), y); loss.backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

**在另一個終端機監控 GPU：**

```bash
nvidia-smi -l 1
```

如果 GPU 使用率長期低於 70%，瓶頸在資料管線，不是模型。

---

## §7 你的 8GB 顯卡能跑什麼（實務參考）

| 任務 | 建議設定 | 大約 VRAM |
|---|---|---|
| MNIST MLP | batch 256 | < 1 GB |
| CIFAR-10 小 CNN | batch 128 | ~2 GB |
| CIFAR-10 ResNet-18 | batch 128 + AMP | ~3 GB |
| ImageNet ResNet-50 微調 224px | batch 32 + AMP | ~7 GB |
| ViT-Small 224px | batch 32 + AMP | ~6 GB |
| 小型 Transformer (d=256, L=512) | batch 32 + AMP | ~3 GB |
| DCGAN 64×64 | batch 128 | ~2 GB |
| 你的股票 CNN+Transformer | batch 64 | < 1 GB |

> 你的股票專案完全不吃記憶體，可以放心把模型做大或用更大的 batch。

---

## §8 動手練習

1. 對同一個 CIFAR-10 訓練，比較「有無 AMP」的 epoch 時間與峰值記憶體。
2. 測量你的資料管線時間，判斷瓶頸在哪。
3. 開啟 `torch.compile`，測量加速比（記得跳過第一個 epoch）。
4. 故意寫 `total_loss += loss`（不加 item），觀察記憶體怎麼漲。
5. 用 profiler 找出你的模型最慢的三個運算。
6. 逐步加大 batch size 直到 OOM，記錄你的卡的極限。

---

## ✅ 自我檢核

- [ ] 說出 bf16 和 fp16 的差別，以及為何 bf16 不需要 GradScaler
- [ ] 寫出 torch 2.x 的 AMP 標準寫法
- [ ] 列出 OOM 時的處方順序
- [ ] 說出為何 `total_loss += loss` 會造成記憶體洩漏
- [ ] 說出測 GPU 時間為何要 `torch.cuda.synchronize()`
- [ ] 說出 `cudnn.benchmark = True` 的前提條件
