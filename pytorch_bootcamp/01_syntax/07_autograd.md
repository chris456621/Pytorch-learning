# 07 · Autograd 自動微分 ★★★

> **這章是 PyTorch 的靈魂。** 你之後寫 GAN、RL、知識蒸餾、meta-learning
> 會不會卡住，完全取決於你有沒有真正懂這一章。
>
> 讀完這章，你要能回答：「這個 tensor 的梯度會流到哪裡？」

---

## §1 核心心智模型

PyTorch 是 **define-by-run**：你每做一次運算，它就**動態地**在背後多接一個節點，
建出一張計算圖（DAG）。呼叫 `backward()` 時，它從輸出往回走這張圖，用連鎖律算梯度。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x * y          # z = 6，同時記錄「z 是由 x 和 y 相乘來的」
w = z + x          # w = 8
w.backward()       # 從 w 往回走

x.grad             # dw/dx = y + 1 = 4
y.grad             # dw/dy = x = 2
```

### 三個關鍵屬性

```python
x.requires_grad    # True  → 這個 tensor 要算梯度
x.grad             # 梯度值（呼叫 backward 後才有）
x.grad_fn          # 這個 tensor 是由哪個運算產生的（葉節點為 None）

z.requires_grad    # True   ← 只要有一個輸入要梯度，輸出就要
z.grad_fn          # <MulBackward0>
z.is_leaf          # False  ← z 是運算結果，不是葉節點
x.is_leaf          # True   ← x 是使用者建立的，是葉節點
```

> ★ **只有葉節點（leaf）且 `requires_grad=True` 的 tensor 才會累積 `.grad`。**
> 中間結果的 `.grad` 預設是 `None`（省記憶體）。想看中間梯度要用 `z.retain_grad()`。

### 圖長什麼樣

```
   x (leaf, requires_grad)  ──┐
                              ├─→ Mul → z ──┐
   y (leaf, requires_grad)  ──┘             ├─→ Add → w
   x ─────────────────────────────────────  ┘

backward() 從 w 出發，沿著 grad_fn 反向傳播，
把梯度累加到每個葉節點的 .grad
```

---

## §2 手算驗證（一定要自己算一次）

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()          # y = 1 + 4 + 9 = 14
y.backward()
print(x.grad)               # tensor([2., 4., 6.])   因為 dy/dx_i = 2*x_i
```

**為什麼 `backward()` 只能對純量呼叫？**

因為梯度的定義是「純量對向量的偏導」。若輸出是向量，PyTorch 不知道你要哪個方向：

```python
y = x ** 2                  # y 是向量 (3,)
y.backward()                # ❌ RuntimeError: grad can be implicitly created
                            #    only for scalar outputs

y.backward(torch.ones_like(y))     # ✅ 明確指定「上游梯度」（vector-Jacobian product）
y.sum().backward()                 # ✅ 更常見的做法：先變純量
```

> ★ **這就是為什麼所有 loss function 最後都會 `.mean()` 或 `.sum()`。**

---

## §3 梯度累加（新手最大的坑）★

**PyTorch 的 `.grad` 是「累加」而不是「覆寫」**。

```python
x = torch.tensor(2.0, requires_grad=True)

y1 = x ** 2; y1.backward()
print(x.grad)       # 4.0

y2 = x ** 3; y2.backward()
print(x.grad)       # 4 + 12 = 16.0   ← 累加了！不是 12
```

所以訓練迴圈**一定**要清零：

```python
for x, y in loader:
    optimizer.zero_grad()        # ★ 清空上一步的梯度
    loss = criterion(model(x), y)
    loss.backward()              # 算梯度並累加到 .grad
    optimizer.step()             # 用 .grad 更新參數
```

> ❓ **既然這麼容易忘，為何要設計成累加？**
> 因為這讓 **gradient accumulation（梯度累積）** 變得很自然 ——
> 顯示卡記憶體不夠跑大 batch 時，可以拆成小 batch 累積梯度再更新一次：
>
> ```python
> ACC = 4
> optimizer.zero_grad()
> for i, (x, y) in enumerate(loader):
>     loss = criterion(model(x), y) / ACC      # ★ 要除以累積步數
>     loss.backward()
>     if (i + 1) % ACC == 0:
>         optimizer.step()
>         optimizer.zero_grad()
> ```
> 這對你的 8GB 顯卡非常實用。

**更快的清零寫法：**

```python
optimizer.zero_grad(set_to_none=True)    # ★ PyTorch 2.x 的預設，把 .grad 設成 None 而非填 0
```

---

## §4 detach / no_grad / requires_grad_ 三兄弟 ★★

這三個都是「不要算梯度」，但**時機和作用範圍完全不同**：

| 寫法 | 作用 | 典型場景 |
|---|---|---|
| `x.detach()` | 從計算圖**切一刀**，回傳一個不帶梯度歷史的新 tensor（共享記憶體） | GAN 訓練 D、target network、記錄數值 |
| `with torch.no_grad():` | **整個區塊**都不建圖 | 驗證、推論、手動更新參數 |
| `p.requires_grad_(False)` | 讓某些**參數**永久不參與訓練 | 凍結預訓練骨幹、linear probing |
| `@torch.inference_mode()` | 比 `no_grad` 更徹底也更快 | 純推論（結果不能再參與訓練） |

```python
# 1. detach —— 切斷單一 tensor
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
c = b.detach() * 4         # c 不會把梯度傳回 a
c.backward()               # ❌ RuntimeError: does not require grad

# 2. no_grad —— 整個區塊
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    for x, y in loader:
        logits = model(x)          # 不建圖 → 省一半記憶體、快很多
        ...

# 3. requires_grad_ —— 凍結參數
for p in model.backbone.parameters():
    p.requires_grad_(False)         # ★ 只訓練 head
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)
```

### 🔥 `detach()` 的三個經典用途（一定要理解）

```python
# (1) GAN 訓練判別器：不要讓梯度流回生成器
fake = G(z)
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)     # ★ detach！
d_loss.backward()          # 只更新 D
d_opt.step()

# 訓練生成器時反而「不能」detach，因為梯度必須流回 G
g_loss = bce(D(fake), ones)          # ← 不 detach
g_loss.backward()
g_opt.step()

# (2) RL 的 target network：TD target 是常數，不能有梯度
with torch.no_grad():
    target = r + gamma * q_target(s_next).max(1).values * (1 - done)
loss = F.mse_loss(q(s).gather(1, a), target)     # target 是常數

# (3) 記錄數值
running_loss += loss.detach()      # 或 loss.item()
```

---

## §5 手刻一個線性迴歸（W4 的必做作業）

**不用 `nn.Linear`、不用 `optim`，純 autograd。**

```python
import torch

torch.manual_seed(0)
X = torch.randn(100, 1)
y_true = 3.0 * X + 2.0 + 0.1 * torch.randn(100, 1)

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

for step in range(200):
    y_pred = X @ w + b                    # forward
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()                       # 算梯度

    with torch.no_grad():                 # ★ 更新參數時不能建圖！
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()                    # ★ 手動清零
        b.grad.zero_()

    if step % 40 == 0:
        print(f"step {step:3d} loss {loss.item():.5f} "
              f"w {w.item():.3f} b {b.item():.3f}")

print(f"最終 w={w.item():.3f} (真值 3.0), b={b.item():.3f} (真值 2.0)")
```

**這段程式碼藏著三個關鍵觀念，一定要懂：**

1. **為什麼更新參數要包在 `no_grad` 裡？**
   因為 `w -= lr * w.grad` 本身也是一個運算。若不關掉梯度追蹤，
   PyTorch 會把它記進計算圖，導致圖無限增長 → 記憶體爆炸。

2. **為什麼要手動 `zero_()`？**
   §3 講過，`.grad` 是累加的。`optimizer.zero_grad()` 做的就是這件事。

3. **`w -= ...` 和 `w = w - ...` 差在哪？**
   `w -= ...`（就地）保持 `w` 是葉節點；
   `w = w - ...` 會產生新 tensor，`w` 不再是葉節點，下次 `w.grad` 就是 `None`。

> ✍️ **作業**：把上面改成兩層 MLP（手寫 forward + 手寫 backward 的 NumPy 版），
> 跟 PyTorch 算出的梯度用 `torch.allclose` 比對。
> 這個作業做完，你對反向傳播的理解會超過 90% 的同學。

---

## §6 計算圖的生命週期

```python
loss.backward()             # 預設會「釋放」整張圖以省記憶體
loss.backward()             # ❌ RuntimeError: Trying to backward through the graph
                            #    a second time
loss.backward(retain_graph=True)    # ✅ 保留圖，可以再 backward 一次
```

**什麼時候真的需要 `retain_graph=True`？**

```python
# 場景：一次 forward，兩個 loss 各自更新不同的網路
feat = encoder(x)
loss_a = head_a(feat).mean()
loss_b = head_b(feat).mean()

loss_a.backward(retain_graph=True)   # 圖還要給 loss_b 用
loss_b.backward()

# ✅ 但更好的做法是：直接加起來一次 backward
(loss_a + loss_b).backward()
```

> ⚠️ **看到 `retain_graph=True` 先懷疑自己寫錯了。**
> 90% 的情況都可以用「把 loss 加起來」或「重新 forward」來避免。
> 濫用它會讓記憶體暴增。

### 🔥 最常見的「backward through graph a second time」真兇

```python
# ❌ 在 RNN / 累積統計時，不小心讓 hidden state 跨 batch 保留計算圖
hidden = torch.zeros(...)
for x in sequence:
    hidden = rnn_cell(x, hidden)      # 圖越接越長
    loss = criterion(hidden, y)
    loss.backward()                   # 第二次就爆

# ✅ 截斷反向傳播（TBPTT）
hidden = hidden.detach()              # 每個 batch 開始前切一刀
```

---

## §7 梯度診斷與控制

### 7.1 看梯度大小（診斷梯度消失/爆炸）

```python
def grad_stats(model):
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name:40s} grad=None  ← ★ 這層沒接到梯度！")
            continue
        g = p.grad
        total += g.pow(2).sum().item()
        print(f"{name:40s} |g|={g.norm():.3e} max={g.abs().max():.3e}")
    print(f"total grad norm = {total ** 0.5:.4f}")
```

| 現象 | 診斷 | 處方 |
|---|---|---|
| `grad = None` | 這層沒參與 forward，或被 detach 切斷 | 檢查 forward 路徑 |
| grad norm ~ 1e-8 | **梯度消失** | 換 ReLU/GELU、加 BatchNorm、加殘差連接、改初始化 |
| grad norm ~ 1e+4 | **梯度爆炸** | 梯度裁剪、降 lr、加 LayerNorm |
| grad norm 正常但 loss 不降 | lr 太小 或 模型容量不足 | 調 lr、加大模型 |

### 7.2 梯度裁剪（訓練 RNN / Transformer 必備）

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # ★ 放在 backward 之後、step 之前
optimizer.step()
```

### 7.3 偵測 nan 的來源

```python
torch.autograd.set_detect_anomaly(True)     # debug 時開，會慢很多
# 出現 nan 時會直接指出是「哪一個 forward 運算」造成的
```

---

## §8 進階：自訂 autograd Function

當你需要自訂反向傳播（例如量化的 Straight-Through Estimator），就要寫這個。

```python
class StraightThroughRound(torch.autograd.Function):
    """forward 做四捨五入（不可微），backward 直接把梯度傳過去。"""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output          # ← 假裝 round 的導數是 1

ste_round = StraightThroughRound.apply

x = torch.tensor([1.3, 2.7], requires_grad=True)
y = ste_round(x).sum()
y.backward()
print(x.grad)        # tensor([1., 1.])  ← 梯度成功穿透
```

> ★ 這正是 **Network Compression 主題（W21）裡 QAT 量化感知訓練**的核心技巧。
> 也可以用一行 trick 達成同樣效果，不用寫 Function：
>
> ```python
> y = x + (torch.round(x) - x).detach()    # forward 值是 round(x)，梯度是 1
> ```
> 這個 trick 在 VQ-VAE、Gumbel-Softmax 等地方也會看到，**值得背下來**。

### 檢查你的自訂梯度對不對

```python
from torch.autograd import gradcheck
x = torch.randn(5, dtype=torch.double, requires_grad=True)
print(gradcheck(MyFunction.apply, (x,), eps=1e-6, atol=1e-4))
```

---

## §9 一個完整的「梯度流」思考練習

看這段程式碼，回答：`loss.backward()` 之後，哪些參數會有梯度？

```python
feat = encoder(x)                 # encoder 的參數 requires_grad=True
feat_d = feat.detach()
out1 = head1(feat)
out2 = head2(feat_d)
loss = out1.mean() + out2.mean()
loss.backward()
```

<details>
<summary>答案</summary>

- `head1` 的參數：✅ 有梯度
- `head2` 的參數：✅ 有梯度（`feat_d` 只是切斷「往上游」的路，head2 自己的參數還是在圖上）
- `encoder` 的參數：✅ **有**梯度，但**只來自 out1 這條路**。
  out2 那條路被 `detach()` 切斷了，對 encoder 沒有貢獻。

★ 這正是 **SimSiam / BYOL 的 stop-gradient** 在做的事，也是 **知識蒸餾中 teacher 不更新** 的做法。
</details>

---

## §10 動手練習

1. 手算 `z = (x * y).sum()` 對 `x` 的梯度，再用 PyTorch 驗證。
2. 完成 §5 的線性迴歸，然後**故意拿掉 `w.grad.zero_()`**，觀察會怎麼壞。
3. 故意拿掉 `with torch.no_grad():`，看會發生什麼錯誤，並解釋原因。
4. 手刻兩層 MLP 的 NumPy backward，跟 PyTorch 的 `.grad` 用 `np.allclose` 比對。
5. 實作 `StraightThroughRound`，並驗證梯度確實是 1。
6. 寫一個 `grad_stats(model)` 放進 `03_code/00_common.py`。
7. 用 `register_hook` 印出某個中間 tensor 的梯度：
   ```python
   h = feat.register_hook(lambda g: print("feat grad norm:", g.norm().item()))
   loss.backward()
   h.remove()
   ```

---

## ✅ 自我檢核（★ 這 8 題答不出來就不要往下讀）

- [ ] 說出什麼是葉節點，為何中間 tensor 的 `.grad` 是 `None`
- [ ] 解釋為何 `backward()` 只能對純量呼叫
- [ ] 解釋 `.grad` 為何是累加的，以及這個設計的好處
- [ ] 說出 `detach()` / `no_grad()` / `requires_grad_(False)` 三者的差別與使用場景
- [ ] 說出 GAN 訓練 D 時為何要 `fake.detach()`，訓練 G 時為何不能
- [ ] 解釋手動更新參數為何要包在 `no_grad` 裡
- [ ] 說出 `retain_graph=True` 什麼時候需要、為何通常代表你寫錯了
- [ ] 寫出 straight-through estimator 的一行 trick 並解釋原理
