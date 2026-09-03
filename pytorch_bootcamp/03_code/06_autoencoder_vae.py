"""AE / Denoising AE / VAE 三合一（對應 02_topics/04_AutoEncoder.md）

用法：
    python 06_autoencoder_vae.py --model ae   --epochs 20
    python 06_autoencoder_vae.py --model dae  --epochs 20 --noise 0.3
    python 06_autoencoder_vae.py --model vae  --epochs 20
    python 06_autoencoder_vae.py --model vae  --beta 10     # ★ 觀察 posterior collapse
    python 06_autoencoder_vae.py --anomaly              # 用 AE 對 2330 做異常偵測
    python 06_autoencoder_vae.py --synthetic --epochs 2  # 沒網路時快速驗證

===========================================================================
三個模型的差別（同一份程式碼用 --model 切換）
---------------------------------------------------------------------------
  ae   輸入 x -> 壓縮成 z -> 還原成 x。學「怎麼用少量數字描述一張圖」
  dae  輸入「加了雜訊的 x」，但目標仍是「乾淨的 x」。逼模型學結構而非複製
  vae  encoder 輸出的是「機率分布」而非一個點，所以 latent space 是連續的，
       可以隨機採樣生成新圖（AE 做不到這件事）
===========================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from common import (set_seed, get_device, print_model_summary, show_batch, save_json)

OUT = Path(__file__).resolve().parent / "outputs"
ROOT = Path(__file__).resolve().parents[2]


# ============================================================
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=784, hidden=(256, 64), latent=32):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, latent),                  # ★ 瓶頸層不加激活
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, in_dim), nn.Sigmoid(),    # ★ 對應 [0,1] 的資料
        )

    def forward(self, x):
        # flatten(1)：(B,1,28,28) -> (B,784)，因為 Linear 只吃二維
        z = self.encoder(x.flatten(1))       # z 是壓縮後的「潛在向量」(B, latent)
        return self.decoder(z), z            # 回傳兩個值：重建結果、以及 z 本身


class VAE(nn.Module):
    def __init__(self, in_dim=784, hidden=400, latent=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)   # ★ 輸出 log(sigma^2) 而非 sigma
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)          # encoder 輸出「分布的參數」而非一個點
        # logvar = log(sigma^2)，所以 sigma = exp(0.5 * logvar)
        # ★ 為什麼輸出 logvar 而不是 sigma？因為 sigma 必須為正，
        #   而 logvar 的值域是整個實數軸，exp 自然保證為正，數值也穩定。
        std = torch.exp(0.5 * logvar)
        # ★★ Reparameterization trick：z = mu + sigma * eps，eps ~ N(0,1)
        #   「採樣」本身不可微，梯度無法傳回 mu 和 sigma。
        #   把隨機性抽到外面的 eps 之後，z 對 mu 的導數是 1、對 std 的導數是 eps，
        #   梯度就能正常流回 encoder。這是整個 VAE 最關鍵的一行。
        z = mu + torch.randn_like(std) * std
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    x = x.flatten(1)
    B = x.size(0)
    # ★ 用 sum：ELBO 對每個維度加總，最後才除以 batch
    # reduction="sum"：ELBO 的定義是對每個維度加總，最後才手動除以 batch size
    # （用 "mean" 會讓 recon 和 KL 的相對權重跑掉）
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    # KL(N(mu,sigma^2) || N(0,1)) 的解析解，直接套公式：
    #   KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 這一項在逼 encoder 的輸出分布靠近標準常態，latent space 才會「填滿沒有洞」
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl) / B, recon_loss.item() / B, kl.item() / B


# ============================================================
def get_data(batch_size, synthetic=False, seed=42, root="./data"):
    from torch.utils.data import DataLoader, TensorDataset, random_split
    if not synthetic:
        try:
            from torchvision import datasets, transforms as T
            tfm = T.ToTensor()                       # ★ 只要 [0,1]，不做 Normalize
            tr = datasets.MNIST(root, True, download=True, transform=tfm)
            te = datasets.MNIST(root, False, download=True, transform=tfm)
            print("[data] MNIST 載入成功")
            return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                    DataLoader(te, batch_size, shuffle=False))
        except Exception as e:
            print(f"[data] MNIST 載入失敗（{type(e).__name__}），改用合成資料")

    g = torch.Generator().manual_seed(seed)
    n = 4000
    proto = torch.rand(10, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    X = (proto[y] + 0.1 * torch.randn(n, 1, 28, 28, generator=g)).clamp(0, 1)
    ds = TensorDataset(X, y)
    tr, te = random_split(ds, [3000, 1000], generator=g)
    print("[data] 使用合成資料")
    return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
            DataLoader(te, batch_size, shuffle=False))


# ============================================================
def train(args):
    set_seed(args.seed)
    device = get_device()
    tr, te = get_data(args.batch_size, args.synthetic, args.seed)

    is_vae = args.model == "vae"
    model = (VAE(784, 400, args.latent) if is_vae
             else AutoEncoder(784, (256, 64), args.latent)).to(device)
    print_model_summary(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    hist = {"loss": [], "recon": [], "kl": []}
    out_dir = OUT / f"ae_{args.model}_beta{args.beta}"

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = rec_s = kl_s = 0.0
        n = 0
        # ★ KL annealing：beta 慢慢升上去，避免 posterior collapse
        beta_ep = args.beta * min(1.0, ep / max(1, args.kl_warmup)) if args.anneal else args.beta

        for x, _ in tr:
            x = x.to(device)
            x_in = x
            if args.model == "dae":
                # ★ DAE：輸入加雜訊，但下面算 loss 時目標仍是「乾淨的 x」。
                #   randn_like(x) 產生跟 x 同形狀同裝置的常態亂數
                #   clamp(0,1) 把值壓回合法的像素範圍
                x_in = (x + args.noise * torch.randn_like(x)).clamp(0, 1)

            if is_vae:
                recon, mu, logvar = model(x_in)
                loss, rl, kl = vae_loss(recon, x, mu, logvar, beta_ep)
            else:
                recon, _ = model(x_in)
                loss = F.mse_loss(recon, x.flatten(1))    # ★ 目標永遠是「乾淨的 x」
                rl, kl = loss.item(), 0.0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            b = x.size(0)
            tot += loss.item() * b; rec_s += rl * b; kl_s += kl * b; n += b

        hist["loss"].append(tot / n); hist["recon"].append(rec_s / n)
        hist["kl"].append(kl_s / n)
        msg = f"  ep {ep:3d}/{args.epochs} | loss {tot/n:.4f} | recon {rec_s/n:.4f}"
        if is_vae:
            msg += f" | KL {kl_s/n:.4f} | beta {beta_ep:.2f}"
            if kl_s / n < 0.05:
                msg += "   <-- ★ KL 趨近 0，posterior collapse！"
        print(msg)

    # ---- 存重建結果 ----
    model.eval()
    x, _ = next(iter(te))
    x = x[:16].to(device)
    with torch.no_grad():
        out = model(x)
        recon = out[0]
    both = torch.cat([x.cpu(), recon.view(-1, 1, 28, 28).cpu()], 0)
    show_batch(both, n=32, save_path=out_dir / "recon.png")
    print(f"  重建圖已存（上半原圖、下半重建）-> {out_dir/'recon.png'}")

    # ---- VAE 額外：從先驗採樣 + latent 內插 ----
    if is_vae:
        with torch.no_grad():
            # ★ VAE 才能做的事：直接從先驗 N(0,I) 採樣，解碼出全新的圖。
            #   AE 做不到，因為它的 latent space 有「洞」，隨便取一點會解出雜訊。
            z = torch.randn(16, args.latent, device=device)
            # view(-1, 1, 28, 28)：-1 代表「這一維你自己算」（這裡會算出 16）
            samples = model.dec(z).view(-1, 1, 28, 28).cpu()
        show_batch(samples, n=16, save_path=out_dir / "sample.png")

        with torch.no_grad():
            mu, _ = model.encode(x[:2])
            # linspace(0,1,10) 產生 0, 0.111, ..., 1 共 10 個數
            # [:, None] 把 (10,) 變成 (10,1)，才能跟 (1,latent) 廣播成 (10,latent)
            alphas = torch.linspace(0, 1, 10, device=device)[:, None]
            # ★ latent 內插：在兩張圖的 latent 之間走一條直線，看解碼結果怎麼變。
            #   VAE 的過渡是平滑的（因為 latent space 連續），AE 則會出現亂碼。
            z_mix = (1 - alphas) * mu[0:1] + alphas * mu[1:2]
            interp = model.dec(z_mix).view(-1, 1, 28, 28).cpu()
        show_batch(interp, n=10, save_path=out_dir / "interpolate.png")
        print(f"  取樣圖與內插圖已存 -> {out_dir}")

    save_json(out_dir / "history.json", hist)
    return model, hist


# ============================================================
def anomaly_detection(args):
    """用 AE 的重建誤差偵測 2330 的異常波動日。"""
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    path = ROOT / "2330.csv"
    if not path.exists():
        print(f"找不到 {path}"); return
    set_seed(args.seed)
    device = get_device()

    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("Date", "label")]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feats).reset_index(drop=True)

    W = 20
    X = df[feats].to_numpy(np.float32)
    n_train = int(len(X) * 0.7)
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-8    # ★ 只用訓練期算
    X = (X - mu) / sd

    from numpy.lib.stride_tricks import sliding_window_view
    Xw = np.ascontiguousarray(sliding_window_view(X, W, axis=0).transpose(0, 2, 1))
    Xw = Xw.reshape(len(Xw), -1)                                # 攤平成向量
    dim = Xw.shape[1]
    n_tr = int(len(Xw) * 0.7)

    model = AutoEncoder(dim, (128, 32), latent=8).to(device)
    # AE 的輸出用 Sigmoid 不適合這裡（資料是 z-score），換成無激活
    model.decoder[-1] = nn.Identity()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xw[:n_tr])),
                           batch_size=128, shuffle=True, drop_last=True)
    print(f"訓練 AE 做異常偵測: 視窗={W} 維度={dim} 訓練樣本={n_tr}")
    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for (xb,) in tr_loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        if ep % max(1, args.epochs // 5) == 0:
            print(f"  ep {ep:3d} | recon MSE {tot/n:.5f}")

    model.eval()
    with torch.no_grad():
        allX = torch.from_numpy(Xw).to(device)
        err = ((model(allX)[0] - allX) ** 2).mean(1).cpu().numpy()
    # ★ 門檻只能用「訓練期」的誤差分布算，用到全部資料就是洩漏
    thr = np.quantile(err[:n_tr], 0.99)               # 99 分位數
    dates = df["Date"].to_numpy()[W - 1:]             # 視窗對齊：第一個視窗結束在第 W-1 天
    # argsort 由小到大排序後的「索引」；[::-1] 反轉變成由大到小；[:15] 取前 15
    idx = np.argsort(err)[::-1][:15]
    print(f"\n異常門檻（訓練期 99 分位）= {thr:.4f}")
    print(f"重建誤差最大的 15 天（可去查那幾天發生什麼事）：")
    for i in sorted(idx):
        print(f"  {str(dates[i])[:10]}  err={err[i]:.4f}"
              f"{'  <-- 超過門檻' if err[i] > thr else ''}")
    print(f"\n全期超過門檻的天數 = {(err > thr).sum()} / {len(err)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ae", "dae", "vae"], default="vae")   # 選要跑哪一種
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--latent", type=int, default=20)         # 瓶頸層的維度
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.3)       # DAE 加噪的強度
    ap.add_argument("--beta", type=float, default=1.0)        # VAE 的 KL 權重
    ap.add_argument("--anneal", action="store_true", help="KL annealing")     # help 會顯示在 -h 裡
    ap.add_argument("--kl-warmup", type=int, default=10)
    ap.add_argument("--anomaly", action="store_true")         # 開關：切換成異常偵測模式
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.anomaly:
        anomaly_detection(args)
    else:
        train(args)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()