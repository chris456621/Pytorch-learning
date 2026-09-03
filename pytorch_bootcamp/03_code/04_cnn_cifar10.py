"""CNN 完整流程：資料增強 + 殘差 + cosine 排程 + Grad-CAM（對應 02_topics/02_CNN.md）

用法：
    python 04_cnn_cifar10.py --epochs 60            # 目標 90%+
    python 04_cnn_cifar10.py --no-aug --epochs 30   # 對照組：沒有資料增強
    python 04_cnn_cifar10.py --model plain          # 對照組：沒有殘差連接
    python 04_cnn_cifar10.py --synthetic --epochs 2 # 沒網路時快速驗證流程
    python 04_cnn_cifar10.py --gradcam ckpt.pt      # 視覺化模型在看哪裡

===========================================================================
本檔用到的語法
---------------------------------------------------------------------------
  nn.Identity()              「什麼都不做」的層，用來當佔位符
  register_forward_hook      在某一層前後插一段自己的程式（抓中間結果）
  @staticmethod              不需要 self 的方法             -> 解碼器 §4
  isinstance(m, nn.Conv2d)   判斷型別（相當於 Java 的 instanceof）
  model.apply(fn)            對模型的「每一個子模組」都跑一次 fn（遞迴）
  參數分組 [{...}, {...}]     讓不同參數用不同的 weight_decay
===========================================================================
"""

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from common import (set_seed, get_device, speed_setup, sanity_check, plot_history,
                    print_model_summary, train_one_epoch, evaluate, evaluate_full,
                    EarlyStopping, save_checkpoint, save_json, plot_confusion)

OUT = Path(__file__).resolve().parent / "outputs"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@dataclass
class Cfg:
    width: int = 64
    model: str = "resnet"          # resnet | plain
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 5e-2
    batch_size: int = 128
    epochs: int = 60
    label_smoothing: float = 0.1
    augment: bool = True
    seed: int = 42
    patience: int = 20
    amp: bool = True


# ============================================================
# 模型
# ============================================================
def conv_bn(in_c, out_c, k=3, s=1, p=1):
    """Conv -> BN -> ReLU 的積木。

    參數：in_c 輸入通道數、out_c 輸出通道數、k 卷積核大小、s 步幅、p 補邊
    輸出尺寸公式：H_out = (H + 2p - k) // s + 1
      k=3,s=1,p=1 -> 尺寸不變（最常用）
      k=3,s=2,p=1 -> 尺寸減半
    ★ bias=False：後面接 BN 時 conv 的 bias 會被 BN 的平移項抵消，是多餘的
    ★ inplace=True：ReLU 直接改寫輸入省記憶體（ReLU 是少數這樣做安全的層）
    """
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()
        if residual and (stride != 1 or in_c != out_c):
            self.shortcut = nn.Sequential(          # ★ shape 不合時用 1x1 conv 對齊
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c))

    def forward(self, x):
        # 由內往外讀：conv1(x) -> bn1(...) -> relu(...)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.residual:
            # ★ 殘差連接：把輸入「加」回來（不是串接）。
            #   梯度因此多了一條 1 的直達路徑，不會因連乘而消失。
            # ★ 一定要寫 out = out + ...，不能寫 out += ...
            #   後者是就地修改，會破壞 autograd 需要的中間值 -> RuntimeError
            out = out + self.shortcut(x)
        return F.relu(out, inplace=True)            # ★ 相加「之後」才 ReLU


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, width=64, dropout=0.1, residual=True):
        super().__init__()
        w = width
        self.stem = conv_bn(3, w)                                   # (B,w,32,32)
        self.layer1 = nn.Sequential(BasicBlock(w, w, 1, residual),
                                    BasicBlock(w, w, 1, residual))
        self.layer2 = nn.Sequential(BasicBlock(w, w * 2, 2, residual),
                                    BasicBlock(w * 2, w * 2, 1, residual))   # 16x16
        self.layer3 = nn.Sequential(BasicBlock(w * 2, w * 4, 2, residual),
                                    BasicBlock(w * 4, w * 4, 1, residual))   # 8x8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # ★ 不管輸入多大都輸出 1x1
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(w * 4, num_classes),
        )
        self.apply(self._init)

    @staticmethod          # 不需要 self，所以標成靜態方法
    def _init(m):
        """權重初始化。由 self.apply(self._init) 對每個子模組呼叫一次。"""
        # isinstance = Java 的 instanceof，判斷 m 是不是某種層
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def features(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return x                                    # (B, 4w, 8, 8)

    def forward(self, x):
        return self.head(self.features(x))


# ============================================================
# 資料
# ============================================================
def get_data(cfg, synthetic=False, root="./data"):
    from torch.utils.data import DataLoader, TensorDataset, random_split

    if not synthetic:
        try:
            from torchvision import datasets, transforms as T
            train_tfm = ([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
                         if cfg.augment else [])
            train_tfm += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
            if cfg.augment:
                train_tfm.append(T.RandomErasing(p=0.25))   # ★ 要放在 Normalize 之後
            test_tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

            full = datasets.CIFAR10(root, True, download=True,
                                    transform=T.Compose(train_tfm))
            val_src = datasets.CIFAR10(root, True, download=True, transform=test_tfm)
            test = datasets.CIFAR10(root, False, download=True, transform=test_tfm)
            g = torch.Generator().manual_seed(cfg.seed)
            idx = torch.randperm(len(full), generator=g)
            tr_idx, va_idx = idx[:45000], idx[45000:]
            from torch.utils.data import Subset
            train = Subset(full, tr_idx.tolist())
            # ★ 驗證集用 test_tfm（不做隨機增強），否則每次評估結果都不同
            val = Subset(val_src, va_idx.tolist())
            print("[data] CIFAR-10 載入成功"
                  f"（augment={'on' if cfg.augment else 'off'}）")
            mk = lambda d, sh: DataLoader(d, cfg.batch_size, shuffle=sh, num_workers=0,
                                          pin_memory=torch.cuda.is_available(),
                                          drop_last=sh)
            return mk(train, True), mk(val, False), mk(test, False)
        except Exception as e:
            print(f"[data] CIFAR-10 載入失敗（{type(e).__name__}: {e}），改用合成資料")

    g = torch.Generator().manual_seed(cfg.seed)
    n = 3000
    proto = torch.randn(10, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    X = proto[y] + 0.6 * torch.randn(n, 3, 32, 32, generator=g)
    ds = TensorDataset(X, y)
    train, val, test = random_split(ds, [2000, 500, 500], generator=g)
    print("[data] 使用合成資料")
    mk = lambda d, sh: DataLoader(d, cfg.batch_size, shuffle=sh, drop_last=sh)
    return mk(train, True), mk(val, False), mk(test, False)


# ============================================================
# Grad-CAM（看模型在看哪裡）
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = self.grads = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    # hook 的固定簽名：forward hook 收到 (模組, 輸入, 輸出)
    def _save_act(self, m, i, o):
        self.acts = o.detach()         # detach 避免把中間結果留在計算圖上

    # backward hook 收到 (模組, 輸入的梯度, 輸出的梯度)，兩者都是 tuple
    def _save_grad(self, m, gi, go):
        self.grads = go[0].detach()    # [0] 取出 tuple 裡的第一個

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        # Grad-CAM 的核心三行：
        # ① 對梯度在空間維度 (H,W) 取平均 -> 得到「每個 channel 有多重要」
        w = self.grads.mean(dim=(2, 3), keepdim=True)     # (1, C, 1, 1)
        # ② 用這個重要度對特徵圖加權求和，再 ReLU 只留正貢獻
        cam = F.relu((w * self.acts).sum(1, keepdim=True))  # (1, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam[0, 0].cpu().numpy(), class_idx


# ============================================================
def run(cfg, synthetic=False, tag=None):
    set_seed(cfg.seed); speed_setup()
    device = get_device()
    tr, va, te = get_data(cfg, synthetic)
    sanity_check(tr, "train")

    model = SmallResNet(10, cfg.width, cfg.dropout,
                        residual=(cfg.model == "resnet")).to(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    # ★ bias 與 norm 的權重不做 weight decay
    # ★ 常見技巧：bias 和 Norm 層的權重「不要」做 weight decay。
    #   判斷方式：這些參數都是一維的（p.ndim <= 1），權重矩陣則是二維以上。
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        # (A if cond else B).append(p) -> 先用三元運算式選出要放進哪個 list，再 append
        (no_decay if p.ndim <= 1 else decay).append(p)
    # 優化器可以吃「參數分組」：一個 list，每個元素是一個字典，
    # 字典裡除了 params 之外還能覆寫該組專屬的超參數（這裡是 weight_decay）。
    # 沒寫在字典裡的（例如 lr）就用外層的預設值。
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    stopper = EarlyStopping(patience=cfg.patience, mode="max")
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None

    tag = tag or f"{cfg.model}_aug{int(cfg.augment)}"
    out_dir = OUT / f"cnn_{tag}"
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best = 0.0

    for ep in range(1, cfg.epochs + 1):
        trl, tra = train_one_epoch(model, tr, criterion, optimizer, device,
                                   amp_dtype=amp_dtype, clip=1.0)
        vl, vacc = evaluate(model, va, criterion, device, amp_dtype=amp_dtype)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        for k, v in zip(hist, [trl, tra, vl, vacc, lr_now]):
            hist[k].append(v)
        print(f"  ep {ep:3d}/{cfg.epochs} | train {trl:.4f}/{tra:.2%} | "
              f"val {vl:.4f}/{vacc:.2%} | lr {lr_now:.2e}")
        if stopper(vacc):
            best = vacc
            save_checkpoint(out_dir / "best.pt", ep, model, optimizer, scheduler,
                            best_metric=best, config=cfg)
        if stopper.should_stop:
            print(f"  early stop @ep{ep}"); break

    plot_history(hist, out_dir / "curves.png")
    save_json(out_dir / "config.json", asdict(cfg))
    m = evaluate_full(model, te, device, 10)
    plot_confusion(m["confusion"], CLASSES, save_path=out_dir / "confusion.png")
    print(f"\n[test] acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
    print(f"每類 F1: {dict(zip(CLASSES, m['per_class_f1']))}")
    return best, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=64)
    # choices=[...]：限制只能傳這幾個字串之一，傳別的值 argparse 會自動報錯並提示
    ap.add_argument("--model", choices=["resnet", "plain"], default="resnet")
    # dest="augment" + store_false：命令列寫 --no-aug 時，args.augment 變 False；
    #   不寫的話 args.augment 預設是 True（有資料增強）。
    #   這樣寫的好處：程式裡用 if cfg.augment 判斷比 if not cfg.no_aug 好讀。
    ap.add_argument("--no-aug", dest="augment", action="store_false")
    ap.add_argument("--lr", type=float, default=1e-3)          # type=float，不是 int！
    ap.add_argument("--batch-size", type=int, default=128)     # 命令列的 - 會自動變底線：
    ap.add_argument("--seed", type=int, default=42)            #   --batch-size -> args.batch_size
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = Cfg(epochs=args.epochs, width=args.width, model=args.model,
              augment=args.augment, lr=args.lr, batch_size=args.batch_size,
              seed=args.seed)
    best, _ = run(cfg, args.synthetic)
    print(f"\n最佳驗證準確率 {best:.4f}")
    print("""
對照實驗建議（一次只改一個變因）：
  python 04_cnn_cifar10.py --epochs 60                 baseline
  python 04_cnn_cifar10.py --epochs 60 --no-aug        看資料增強值多少分
  python 04_cnn_cifar10.py --epochs 60 --model plain   看殘差連接值多少分
  python 04_cnn_cifar10.py --epochs 60 --width 32      看模型容量的影響
""")


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()