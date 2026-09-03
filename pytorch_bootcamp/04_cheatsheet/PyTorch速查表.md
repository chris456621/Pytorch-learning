# PyTorch 速查表（印出來貼在桌上）

## 建立 tensor

```python
torch.tensor([1., 2.])            torch.zeros(2,3)      torch.ones(2,3)
torch.full((2,3), 7.)             torch.eye(3)          torch.empty(2,3)
torch.arange(0,10,2)              torch.linspace(0,1,5)
torch.rand(2,3)   torch.randn(2,3)   torch.randint(0,10,(2,3))
torch.randperm(10)                torch.multinomial(p, 1)
torch.zeros_like(x)  torch.randn_like(x)   # ★ 自動同 device / dtype
```

## dtype 與 device

```python
x.float() x.long() x.bool() x.half()
x.to(dtype=torch.float32, device="cuda")      x.type_as(y)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)          # ★ tensor 要重新賦值
model.to(device)          # ★ Module 是就地搬移
```

| 用途 | dtype |
|---|---|
| 特徵 / 權重 | `float32` |
| 分類標籤 | `int64` (`.long()`) |
| mask | `bool` |
| 混合精度 | `bfloat16`（你的 4060 支援） |

## 改形狀

| 操作 | 說明 |
|---|---|
| `x.view(2,-1)` | 需連續，一定是 view |
| `x.reshape(2,-1)` | ★ 不確定就用這個 |
| `x.flatten(1)` | 從 dim1 起攤平（CNN→FC） |
| `x.permute(0,2,3,1)` | 重排多個維度 |
| `x.transpose(1,2)` | 交換兩個維度 |
| `x.unsqueeze(0)` / `x[None]` | 加一維 |
| `x.squeeze(0)` | 移除長度 1 的維度（★ 一定指定 dim） |
| `x.contiguous()` | permute 後、view 前 |

**黃金法則**：`permute` 之後改形狀用 `reshape`，不要用 `view`。

## 合併 / 分割 / 擴張

```python
torch.cat([a,b], dim=0)     # 不增加維度數
torch.stack([a,b], dim=0)   # ★ 一定多一維
x.chunk(3, dim=1)   x.split(2, dim=1)   torch.unbind(x, 0)
x.expand(4,3)       # 不複製記憶體（唯讀），只能擴張長度 1 的維度
x.repeat(4,1)       # 真的複製
```

## 歸約（PyTorch 用 `dim`，NumPy 用 `axis`）

```python
x.sum(dim=1)  x.mean(dim=(0,1), keepdim=True)   # ★ keepdim 沒有 s
v, i = x.max(dim=1)      # ★ 回傳兩個東西
x.argmax(dim=1)   x.amax(dim=1)   x.topk(5, dim=1)
torch.linalg.norm(x, ord=2)
```

## 索引與遮罩

```python
x[x > 0]                              x[x > 0] = 0
torch.where(cond, a, b)
scores.masked_fill(mask, float("-inf"))          # ★ attention mask
q.gather(1, actions[:,None])                     # ★ 取每列指定索引
F.one_hot(labels, num_classes=10).float()
torch.triu(torch.ones(L,L,dtype=torch.bool), 1)  # ★ causal mask
```

## einsum

```python
"ij,jk->ik"      矩陣乘法          "ij->ji"     轉置
"bij,bjk->bik"   batch 矩陣乘法     "bi,bi->b"   batch 內積
"bhqd,bhkd->bhqk"  attention 分數
```

## Autograd

```python
x.requires_grad_(True)      x.grad      x.grad_fn      x.is_leaf
loss.backward()                          # 只能對純量
y.backward(torch.ones_like(y))           # 向量要給上游梯度

x.detach()                  # 切斷單一 tensor
with torch.no_grad(): ...   # 整個區塊不建圖
@torch.no_grad()            # 裝飾器版本
@torch.inference_mode()     # 純推論，更快
p.requires_grad_(False)     # 凍結參數

optimizer.zero_grad(set_to_none=True)    # ★ 梯度是累加的
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
torch.autograd.set_detect_anomaly(True)  # debug nan
```

**Straight-through estimator**（量化 / VQ-VAE / Gumbel）：

```python
y = x + (torch.round(x) - x).detach()    # forward 是 round(x)，梯度是 1
```

## nn.Module

```python
class M(nn.Module):
    def __init__(self):
        super().__init__()                       # ★ 必須
        self.fc = nn.Linear(10, 5)
        self.register_buffer("pe", torch.zeros(100, 10))   # ★ 不是 parameter 但要跟著 to(device)
    def forward(self, x): ...

model(x)                    # ★ 不要寫 model.forward(x)
model.train() / model.eval()             # 影響 Dropout 與 BatchNorm
model.parameters()  model.named_parameters()  model.named_modules()
model.apply(init_fn)
nn.ModuleList([...])  nn.ModuleDict({...})     # ★ 不能用 Python list
```

## 常用層

```python
nn.Linear(i,o)   nn.Dropout(0.2)   nn.Flatten(1)   nn.Identity()
nn.Conv2d(i,o,k,stride,padding, bias=False)      nn.Conv1d(...)
nn.MaxPool2d(2)  nn.AdaptiveAvgPool2d(1)
nn.BatchNorm2d(c)   nn.LayerNorm(d)   nn.GroupNorm(g,c)
nn.ReLU(inplace=True)  nn.LeakyReLU(0.2)  nn.GELU()  nn.SiLU()  nn.Tanh()
nn.Embedding(n, d)
nn.LSTM(i,h,layers, batch_first=True)            # ★ batch_first
nn.TransformerEncoderLayer(d, nhead, dim_ff, batch_first=True, norm_first=True)
```

**卷積輸出尺寸**：`H_out = (H + 2p - k) // s + 1`

| k | s | p | 效果 |
|---|---|---|---|
| 3 | 1 | 1 | 尺寸不變 |
| 3 | 2 | 1 | 減半 |
| 4 | 2 | 1 | 減半（GAN 用） |
| 1 | 1 | 0 | 只改 channel |

## Loss

```python
nn.CrossEntropyLoss(label_smoothing=0.1)   # ★ 吃 logits，labels 是 int64
nn.BCEWithLogitsLoss(pos_weight=w)         # ★ 二分類用這個
nn.MSELoss()  nn.L1Loss()  nn.SmoothL1Loss()
nn.KLDivLoss(reduction="batchmean")        # ★ 第一個參數要 log_softmax
F.cross_entropy(logits, y, reduction="none")   # 每筆的 loss
```

## Optimizer 與 Scheduler

```python
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)     # ★ 預設選擇
optim.SGD(..., lr=0.1, momentum=0.9, nesterov=True)
optim.Adam(..., betas=(0.5, 0.999))                             # GAN

lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)   # 每 epoch step()
lr_scheduler.OneCycleLR(opt, max_lr, steps_per_epoch=..., epochs=...)  # 每 batch
lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5)          # step(metric)
opt.param_groups[0]["lr"]
```

## 訓練迴圈

```python
for x, y in loader:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    total += loss.item() * y.size(0)          # ★ 一定要 .item()
```

## 存讀

```python
torch.save(model.state_dict(), "m.pt")
model.load_state_dict(torch.load("m.pt", map_location="cpu", weights_only=True))
missing, unexpected = model.load_state_dict(sd, strict=False)   # ★ 一定要印出來檢查
```

## 效能樣板（放腳本開頭）

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

## Debug 三件套

```python
print(f"{x.shape=} {x.dtype=} {x.device=} {x.requires_grad=}")
print(x.min().item(), x.max().item(), x.mean().item())
print(torch.isnan(x).any().item(), torch.isinf(x).any().item())
```
