"""DCGAN + WGAN-GP（對應 02_topics/06_GAN.md）

用法：
    python 08_gan_dcgan.py --epochs 30                # DCGAN on MNIST
    python 08_gan_dcgan.py --loss wgangp --epochs 30  # WGAN-GP
    python 08_gan_dcgan.py --no-detach                # ★ 觀察忘記 detach 的後果
    python 08_gan_dcgan.py --synthetic --epochs 2     # 快速驗證流程

===========================================================================
讀這份程式碼前先弄懂的三件事
---------------------------------------------------------------------------
1. G（生成器）吃隨機雜訊 z，吐出假圖；D（判別器）吃圖，吐出「是真的」的分數。
2. 兩者交替訓練：先固定 G 訓練 D，再固定 D 訓練 G。
3. ★ 最關鍵的一行是訓練 D 時的 fake.detach()。
   detach 切斷梯度往 G 回流，讓這一步只更新 D。
   訓練 G 時「不能」detach，因為 G 的梯度必須經過 D 才算得出來。
   用 --no-detach 可以親眼看到梯度被污染。
===========================================================================
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))
from common import set_seed, get_device, speed_setup, print_model_summary, show_batch, save_json

OUT = Path(__file__).resolve().parent / "outputs"


# ============================================================
class Generator(nn.Module):
    """z (B, nz, 1, 1) -> 圖片 (B, nc, 32, 32)，值域 [-1, 1]

    ConvTranspose2d 是「反過來的卷積」，用途是把小的特徵圖放大。
    參數 (in, out, kernel=4, stride=2, padding=1) 的組合會讓尺寸剛好加倍。
    ★ 最後用 Tanh 把輸出壓到 [-1,1]，所以真實資料也必須正規化到 [-1,1]
      （Normalize([0.5],[0.5]) 就是在做這件事）。配錯會生成一片黑或一片白。
    """
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),              # 4x4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),              # 8x8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),                  # 16x16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),                                            # ★ 輸出 [-1,1]
        )                                                         # 32x32

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """圖片 -> 一個分數（logit）。分數越大代表「越像真的」。

    ★ 輸出「不加 sigmoid」，因為後面用 BCEWithLogitsLoss（內建 sigmoid 且數值穩定）。
    ★ D 用 LeakyReLU 而非 ReLU：ReLU 把負值全砍成 0，梯度也一起沒了，
      GAN 的 D 需要讓梯度順暢回傳給 G，所以留一條 0.2 的斜坡。
    """
    def __init__(self, ndf=64, nc=1, use_bn=True):
        super().__init__()
        # 巢狀函式：只在這個 __init__ 裡用得到，不需要變成類別的方法
        def blk(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
            if norm:
                # ★ WGAN-GP 不能用 BatchNorm（GP 是逐樣本的），改用 InstanceNorm
                layers.append(nn.BatchNorm2d(o) if use_bn else nn.InstanceNorm2d(o, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))        # ★ D 用 LeakyReLU
            return layers
        self.net = nn.Sequential(
            *blk(nc, ndf, norm=False),          # ★ D 的第一層不加 norm
            *blk(ndf, ndf * 2),
            *blk(ndf * 2, ndf * 4),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),           # ★ 輸出 logits
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def dcgan_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in cls and getattr(m, "weight", None) is not None:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ============================================================
def gradient_penalty(D, real, fake, device):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_hat = D(x_hat)
    # torch.autograd.grad(...) 手動求梯度，回傳一個 tuple，[0] 取出第一個
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,          # 對誰求、關於誰求
        grad_outputs=torch.ones_like(d_hat),  # 上游梯度（因為 d_hat 不是純量）
        # ★★ create_graph=True：讓「這個梯度」本身也留在計算圖上，
        #   因為 GP 是 loss 的一部分，等一下還要對它再求一次導（二階梯度）。
        #   忘了寫的話 GP 這一項完全不會產生梯度，等於沒加。
        create_graph=True,
        retain_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def get_data(args):
    from torch.utils.data import DataLoader, TensorDataset
    if not args.synthetic:
        try:
            from torchvision import datasets, transforms as T
            tfm = T.Compose([T.Resize(32), T.ToTensor(),
                             T.Normalize([0.5], [0.5])])   # ★ 配合 Tanh：[0,1]->[-1,1]
            ds = datasets.MNIST("./data", True, download=True, transform=tfm)
            print("[data] MNIST 載入成功")
            return DataLoader(ds, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=torch.cuda.is_available()), 1
        except Exception as e:
            print(f"[data] MNIST 載入失敗（{type(e).__name__}），改用合成資料")
    g = torch.Generator().manual_seed(args.seed)
    n = 2000
    proto = torch.rand(8, 1, 32, 32, generator=g) * 2 - 1
    y = torch.randint(0, 8, (n,), generator=g)
    X = (proto[y] + 0.15 * torch.randn(n, 1, 32, 32, generator=g)).clamp(-1, 1)
    print("[data] 使用合成資料")
    return DataLoader(TensorDataset(X, y), args.batch_size, shuffle=True,
                      drop_last=True), 1


# ============================================================
def train(args):
    set_seed(args.seed); speed_setup()
    device = get_device()
    loader, nc = get_data(args)

    is_wgan = args.loss == "wgangp"
    G = Generator(args.nz, args.ngf, nc).to(device)
    D = Discriminator(args.ndf, nc, use_bn=not is_wgan).to(device)
    G.apply(dcgan_init); D.apply(dcgan_init)
    print_model_summary(G); print_model_summary(D)

    # ★ DCGAN 論文的關鍵設定：beta1 = 0.5（不是預設的 0.9）
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, args.nz, 1, 1, device=device)   # ★ 固定 z 觀察演進
    tag = f"gan_{args.loss}{'' if args.detach else '_nodetach'}"
    out_dir = OUT / tag
    hist = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": [], "diversity": []}

    for ep in range(1, args.epochs + 1):
        dl = gl = dr = df = 0.0
        steps = 0
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            B = real.size(0)
            z = torch.randn(B, args.nz, 1, 1, device=device)
            fake = G(z)

            # ---------- 訓練 D ----------
            # ★★ detach：不讓 D 的梯度流回 G
            # ★★★ 這一行是整份程式碼最重要的地方。
            #   fake.detach() 產生一個「值一樣但不帶計算圖」的張量，
            #   於是 d_loss.backward() 的梯度走到這裡就停住，不會流回 G。
            #   不 detach 的話：G 的 .grad 會被這一步污染（雖然 optD.step() 不更新 G，
            #   但下一步訓練 G 時若沒清乾淨就會錯），而且白白多算一次反向傳播。
            fake_for_d = fake.detach() if args.detach else fake
            if is_wgan:
                gp = gradient_penalty(D, real, fake.detach(), device)
                d_loss = D(fake_for_d).mean() - D(real).mean() + args.gp_weight * gp
            else:
                d_loss = (bce(D(real), torch.ones(B, 1, device=device) * args.smooth)
                          + bce(D(fake_for_d), torch.zeros(B, 1, device=device)))
            optD.zero_grad(set_to_none=True)
            d_loss.backward(retain_graph=not args.detach)
            optD.step()

            if not args.detach and i == 0 and ep == 1:
                gnorm = sum(p.grad.pow(2).sum().item() for p in G.parameters()
                            if p.grad is not None) ** 0.5
                print(f"  ★ 沒有 detach 時，訓練 D 之後 G 的梯度範數 = {gnorm:.4e}"
                      f"（應該是 0，這就是污染）")

            # ---------- 訓練 G ----------
            if i % args.n_critic == 0:
                z = torch.randn(B, args.nz, 1, 1, device=device)
                fake2 = G(z)
                # ★ 這裡「不能」detach，梯度必須經過 D 流回 G
                if is_wgan:
                    g_loss = -D(fake2).mean()
                else:
                    # ★ non-saturating loss：把假圖標成「真的」(ones)，
                    #   要求 D 認為它是真的。等價於最大化 log D(G(z))。
                    #   為什麼不用原論文的 log(1-D(G(z)))？因為訓練初期 G 很爛、
                    #   D(G(z))≈0，那個式子的梯度接近 0，G 完全學不動。
                    # ★ 這裡「不能」detach —— 梯度必須穿過 D 才能回到 G。
                    g_loss = bce(D(fake2), torch.ones(B, 1, device=device))
                optG.zero_grad(set_to_none=True)
                g_loss.backward()
                optG.step()
                gl += g_loss.item()

            with torch.no_grad():
                dr += torch.sigmoid(D(real)).mean().item()
                df += torch.sigmoid(D(fake.detach())).mean().item()
            dl += d_loss.item(); steps += 1

        n_g = max(1, steps // args.n_critic)
        hist["d_loss"].append(dl / steps); hist["g_loss"].append(gl / n_g)
        hist["d_real"].append(dr / steps); hist["d_fake"].append(df / steps)

        # ★ mode collapse 診斷：生成樣本之間的平均距離
        with torch.no_grad():
            # mode collapse 診斷：生成 64 張圖，算兩兩之間的平均距離。
            # torch.cdist(a, b) 算「a 的每一列」對「b 的每一列」的歐氏距離。
            # 距離持續下降 = 生成的東西越來越像 = mode collapse。
            s = G(torch.randn(64, args.nz, 1, 1, device=device)).flatten(1)
            div = torch.cdist(s, s).mean().item()
        hist["diversity"].append(div)

        msg = (f"  ep {ep:3d}/{args.epochs} | D {dl/steps:+.4f} | G {gl/n_g:+.4f} | "
               f"D(real) {dr/steps:.3f} D(fake) {df/steps:.3f} | 多樣性 {div:.3f}")
        if dr / steps > 0.98 and df / steps < 0.02:
            msg += "  <-- ★ D 太強，G 拿不到梯度"
        if div < 1.0:
            msg += "  <-- ★ 可能 mode collapse"
        print(msg)

        if ep % max(1, args.epochs // 5) == 0 or ep == args.epochs:
            with torch.no_grad():
                # ★ 反正規化：訓練時把資料從 [0,1] 映到 [-1,1]（Normalize(0.5,0.5)），
                #   要顯示就得反過來：x * 0.5 + 0.5
                imgs = G(fixed_z).cpu() * 0.5 + 0.5
            show_batch(imgs, n=64, save_path=out_dir / f"ep{ep:03d}.png")

    save_json(out_dir / "history.json", hist)
    print(f"\n生成結果已存 -> {out_dir}")
    print("""
診斷指南：
  D(real)~0.6-0.8, D(fake)~0.2-0.4  健康
  D(real)~1.0,     D(fake)~0.0      D 太強，降 D 的 lr 或加 label smoothing
  兩者都~0.5 但圖很爛                D 太弱
  多樣性持續下降                     mode collapse，試 WGAN-GP 或加大 batch
  ★ GAN 的 loss 值本身沒有意義，一定要看上面這些指標 + 用眼睛看圖
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--nz", type=int, default=100)             # 雜訊向量 z 的維度
    ap.add_argument("--ngf", type=int, default=64)             # Generator 的基準寬度
    ap.add_argument("--ndf", type=int, default=64)             # Discriminator 的基準寬度
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loss", choices=["bce", "wgangp"], default="bce")
    ap.add_argument("--n-critic", type=int, default=1, help="WGAN-GP 建議 5")
    ap.add_argument("--gp-weight", type=float, default=10.0)   # gradient penalty 的權重
    ap.add_argument("--smooth", type=float, default=0.9, help="label smoothing")
    # 這個參數是刻意設計來「示範錯誤」的：
    #   平常不加 --no-detach，args.detach=True（正確流程）
    #   加了 --no-detach，args.detach=False（故意犯錯，讓你看到梯度污染）
    ap.add_argument("--no-detach", dest="detach", action="store_false",
                    help="★ 故意不 detach，觀察 G 的梯度被污染")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.loss == "wgangp" and args.n_critic == 1:
        args.n_critic = 5
        print("[info] WGAN-GP 自動設定 n_critic=5")
    train(args)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()