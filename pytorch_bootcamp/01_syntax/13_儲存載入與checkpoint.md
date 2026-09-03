# 13 · 模型儲存、載入與 Checkpoint

---

## §1 只存權重（★ 標準做法）

```python
# 存
torch.save(model.state_dict(), "model.pt")

# 讀
model = MyModel(**same_args)             # ★ 必須先建出結構一樣的模型
model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
model.to(device).eval()
```

> ❌ **不要用 `torch.save(model, ...)` 存整個模型物件。**
> 它用 pickle 序列化，會綁死你的類別路徑和檔案結構，
> 換個資料夾或改個類別名就讀不回來。**只存 `state_dict`。**

### 【新版注意】`weights_only`

PyTorch **2.6 起 `torch.load` 的 `weights_only` 預設為 `True`**（你的 2.12 也是）。
這是安全性改動：避免載入惡意 pickle 執行任意程式碼。

```python
# 只有純 tensor 的 state_dict → 直接可讀
sd = torch.load("model.pt", weights_only=True)

# checkpoint 裡有 dataclass、numpy 陣列等自訂物件 → 會報錯
# 解法 1（推薦）：存的時候就只存基本型別
torch.save({"model": model.state_dict(), "epoch": ep, "acc": float(acc)}, "ck.pt")

# 解法 2：明確允許特定類別
torch.serialization.add_safe_globals([MyConfig])
sd = torch.load("ck.pt", weights_only=True)

# 解法 3：你完全信任這個檔案（★ 只對自己產生的檔案這樣做）
sd = torch.load("ck.pt", weights_only=False)
```

> ⚠️ **絕對不要對網路上下載的 `.pth` 用 `weights_only=False`**，
> 那等於執行來路不明的程式碼。

---

## §2 完整 checkpoint（可續訓）★

```python
def save_checkpoint(path, epoch, model, optimizer, scheduler=None,
                    scaler=None, best_metric=None, config=None):
    ck = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),      # ★ Adam 的動量狀態，續訓一定要
        "best_metric": best_metric,
        "config": config,                          # 用 asdict() 轉成純 dict
        "torch_version": torch.__version__,
        "rng_state": torch.get_rng_state(),        # 讓續訓完全可重現
    }
    if scheduler is not None:
        ck["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ck["scaler"] = scaler.state_dict()
    torch.save(ck, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    scaler=None, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and "scheduler" in ck:
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    return ck["epoch"], ck.get("best_metric")
```

**續訓的主迴圈：**

```python
start_epoch = 1
if resume_path and Path(resume_path).exists():
    start_epoch, best = load_checkpoint(resume_path, model, optimizer, scheduler)
    start_epoch += 1
    print(f"從 epoch {start_epoch} 續訓")

for ep in range(start_epoch, EPOCHS + 1):
    ...
```

---

## §3 只存最好的 + 定期存快照

```python
out = Path("runs/exp001"); out.mkdir(parents=True, exist_ok=True)

if va_acc > best_acc:
    best_acc = va_acc
    save_checkpoint(out / "best.pt", ep, model, optimizer, scheduler,
                    best_metric=best_acc, config=asdict(cfg))

save_checkpoint(out / "last.pt", ep, model, optimizer, scheduler)   # 每個 epoch 覆蓋

if ep % 20 == 0:                                    # 定期快照
    save_checkpoint(out / f"epoch_{ep:03d}.pt", ep, model, optimizer, scheduler)
```

> 磁碟空間有限時只留 `best.pt` 和 `last.pt`。
> 你的 `strong_net.pth` 有 12MB，`best_cnn_transformer.pth` 只有 178KB ——
> 建議統一改成上面的格式，之後才知道哪個 checkpoint 是什麼設定跑出來的。

---

## §4 載入時的常見狀況

### 4.1 部分載入（遷移學習 / SSL 預訓練 → 下游）★

```python
sd = torch.load("pretrained.pt", map_location="cpu", weights_only=True)

# 只取 encoder 的部分
enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
model.encoder.load_state_dict(enc_sd)

# 允許不完全吻合（★ 換了分類頭時用）
missing, unexpected = model.load_state_dict(sd, strict=False)
print("模型有但檔案沒有:", missing)          # 通常是新的分類頭
print("檔案有但模型沒有:", unexpected)        # 通常是舊的分類頭
```

> ★ **一定要把 `missing` 和 `unexpected` 印出來檢查。**
> `strict=False` 會安靜地忽略不吻合的鍵，
> 如果你打錯前綴導致**一個權重都沒載到**，模型還是會正常跑，只是效果爛。
> 這種 bug 極難發現。

**驗證真的載入成功的方法：**

```python
before = model.encoder.conv1.weight.clone()
model.encoder.load_state_dict(enc_sd)
print("有變:", not torch.equal(before, model.encoder.conv1.weight))   # 應該 True
```

### 4.2 鍵名不吻合

```python
# DataParallel 存的檔案會多 module. 前綴
sd = {k.replace("module.", ""): v for k, v in sd.items()}

# torch.compile 存的會多 _orig_mod. 前綴
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
```

### 4.3 跨裝置載入

```python
torch.load(path, map_location="cpu")                  # ★ 最保險，載完再 .to(device)
torch.load(path, map_location=device)
torch.load(path, map_location={"cuda:1": "cuda:0"})
```

---

## §5 匯出模型做部署

```python
# TorchScript（不需要原始碼就能載入）
scripted = torch.jit.script(model.eval())
scripted.save("model_scripted.pt")
loaded = torch.jit.load("model_scripted.pt")

# trace 版本（模型有 if/for 分支時不要用）
traced = torch.jit.trace(model.eval(), example_input)

# ONNX（跨框架部署）
torch.onnx.export(
    model.eval(), example_input, "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

---

## §6 實驗管理慣例（★ 做研究一定要建立）

```
runs/
├── exp001_baseline/
│   ├── config.json          超參數
│   ├── best.pt              最佳權重
│   ├── last.pt              最新權重（續訓用）
│   ├── history.json         每個 epoch 的指標
│   ├── curves.png           訓練曲線
│   └── log.txt              完整輸出
├── exp002_add_bn/
└── exp003_cosine_lr/
```

```python
import json
from dataclasses import asdict

out = Path(f"runs/{cfg.exp_name}"); out.mkdir(parents=True, exist_ok=True)

# 存設定
(out / "config.json").write_text(
    json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

# 存歷程
(out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

# 同時輸出到檔案和終端機
import sys
class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.file.write(s); self.file.flush()
    def flush(self):
        self.stdout.flush(); self.file.flush()

sys.stdout = Tee(out / "log.txt")
```

> 🔥 **三個月後你一定會問「這個 0.9123 是怎麼跑出來的？」**
> 有 `config.json` 就答得出來，沒有就只能重跑。

---

## §7 動手練習

1. 把 `save_checkpoint` / `load_checkpoint` 加進 `03_code/00_common.py`。
2. 訓練 10 epoch 存檔，中斷後續訓，確認 loss 曲線是接續的而不是重來。
3. 存一個模型後改變模型結構，用 `strict=False` 載入，印出 missing/unexpected。
4. 驗證「權重真的有載入」（用 §4.1 的 clone 比對法）。
5. 把你現有的 `strong_net.pth`、`best_cnn_transformer.pth` 改存成標準 checkpoint 格式。
6. 建立 `runs/` 的實驗資料夾慣例，跑三個不同設定的實驗。

---

## ✅ 自我檢核

- [ ] 說出為何要存 `state_dict` 而不是整個 model 物件
- [ ] 說出 `weights_only=True` 是在防什麼，什麼時候需要關掉
- [ ] 說出續訓時除了 model 還要載入什麼（提示：至少三樣）
- [ ] 說出 `strict=False` 的風險與驗證方法
- [ ] 說出 `map_location="cpu"` 為何是最保險的做法
- [ ] 列出你的實驗資料夾應該包含哪些檔案
