"""Self-supervised：SimCLR + SimSiam + linear probe / k-NN 評估
（對應 02_topics/05_SelfSupervised.md）

用法：
    python 07_ssl_simclr.py --method simclr --epochs 100
    python 07_ssl_simclr.py --method simsiam --epochs 100
    python 07_ssl_simclr.py --method simsiam --no-stopgrad   # ★ 觀察 collapse
    python 07_ssl_simclr.py --synthetic --epochs 3           # 快速驗證流程
    python 07_ssl_simclr.py --loss-test                      # 只驗證 InfoNCE 實作
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from common import set_seed, get_device, speed_setup, print_model_summary, save_json

OUT = Path(__file__).resolve().parent / "outputs"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# 1. InfoNCE / NT-Xent  ★ 本檔最重要的函式
# ============================================================
def nt_xent_loss(z1, z2, temperature=0.5):
    """InfoNCE / NT-Xent —— 對比學習的核心 loss。

    輸入：z1, z2 都是 (N, D)。z1[i] 和 z2[i] 是「同一張圖的兩個不同增強」，互為正樣本。
    目標：讓每個 z1[i] 跟 z2[i] 很像，跟 batch 內其他 2N-2 個都不像。

    關鍵洞見：這其實就是一個「2N 類的分類問題」——
             對第 i 個向量，從 2N-1 個候選中挑出它的正樣本。
             所以最後可以直接用 cross_entropy，不用自己寫 log 和除法。
    """
    N = z1.size(0)
    # ① 把兩組疊成一個 (2N, D)：前 N 列是 view1，後 N 列是 view2
    #    F.normalize(dim=1) 把每一列變成單位長度，這樣內積就等於餘弦相似度
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)         # (2N, D)
    # ② z @ z.t() 得到所有兩兩之間的相似度矩陣。t() 是轉置（只適用二維）
    #    除以 temperature：越小越「尖銳」，對難負樣本越敏感
    sim = z @ z.t() / temperature                          # (2N, 2N)
    # ③ 對角線是「自己跟自己」，相似度必定最高，要排除掉
    eye = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))              # -inf 經過 softmax 會變 0
    # ④ 每一列的「正確答案」在第幾欄？
    #    第 i 列（i<N）的正樣本在第 i+N 欄；第 i+N 列的正樣本在第 i 欄
    targets = torch.cat([torch.arange(N, 2 * N),           # [N, N+1, ..., 2N-1]
                         torch.arange(0, N)]).to(z.device) # [0, 1, ..., N-1]
    # ⑤ 把 sim 當 logits、targets 當標籤，直接丟 cross_entropy
    return F.cross_entropy(sim, targets)


def simsiam_loss(p1, p2, z1, z2):
    return -(F.cosine_similarity(p1, z2).mean()
             + F.cosine_similarity(p2, z1).mean()) * 0.5


# ============================================================
# 2. 模型
# ============================================================
def make_encoder(width=64):
    """小型 CNN encoder（8GB 顯卡跑得動）。"""
    def blk(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    return nn.Sequential(
        blk(3, width), blk(width, width),
        blk(width, width * 2, 2), blk(width * 2, width * 2),      # 16x16
        blk(width * 2, width * 4, 2), blk(width * 4, width * 4),  # 8x8
        nn.AdaptiveAvgPool2d(1), nn.Flatten(1),                   # (B, 4w)
    )


class SimCLR(nn.Module):
    def __init__(self, feat_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = make_encoder(feat_dim // 4)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True), nn.Linear(feat_dim, proj_dim))

    def forward(self, x):
        h = self.encoder(x)          # ★ 下游用 h
        return h, self.projector(h)  # ★ loss 算在 z 上


class SimSiam(nn.Module):
    def __init__(self, feat_dim=256, pred_dim=64, stopgrad=True):
        super().__init__()
        self.encoder = make_encoder(feat_dim // 4)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim))
        self.predictor = nn.Sequential(      # ★ 只有一邊有 predictor（不對稱）
            nn.Linear(feat_dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), nn.Linear(pred_dim, feat_dim))
        self.stopgrad = stopgrad

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        # ★★ 下面這個 detach 就是 SimSiam 整篇論文的核心。
        #   它切斷梯度往 z 那條路回流，讓 predictor 追著一個「暫時固定的目標」跑
        #   （類似 EM 演算法的交替最佳化）。
        #   拿掉它 -> 兩邊同時往「常數解」坍縮 -> loss 迅速降到 -1 但什麼都沒學到。
        #   用 --no-stopgrad 可以親眼看到這個現象。
        if self.stopgrad:
            z1, z2 = z1.detach(), z2.detach()    # ★★ 論文的核心就是這一行
        return p1, p2, z1, z2


# ============================================================
# 3. 資料：TwoCropTransform
# ============================================================
class TwoCrop:
    """把同一張圖做「兩次不同的隨機增強」，產生一對正樣本。

    語法說明：定義 __call__ 之後，這個物件就能像函式一樣被呼叫，
              所以可以直接塞給 torchvision 的 transform 參數。
    為什麼兩次結果不同？因為 transform 內部有隨機性（隨機裁切、隨機顏色），
    呼叫兩次自然得到兩個不同的視角。
    """
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, x): return self.tfm(x), self.tfm(x)


class SyntheticTwoCrop(torch.utils.data.Dataset):
    """沒網路時的替代品：同一個原型加不同雜訊 = 兩個 view。"""
    def __init__(self, n=2000, n_cls=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.proto = torch.randn(n_cls, 3, 32, 32, generator=g)
        self.y = torch.randint(0, n_cls, (n,), generator=g)
        self.g = g
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        base = self.proto[self.y[i]]
        return (base + 0.5 * torch.randn_like(base),
                base + 0.5 * torch.randn_like(base)), self.y[i]


def get_data(args):
    from torch.utils.data import DataLoader
    if not args.synthetic:
        try:
            from torchvision import datasets, transforms as T
            ssl_tfm = T.Compose([
                T.RandomResizedCrop(32, scale=(0.2, 1.0)),          # ★ 最重要
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # ★ 第二重要
                T.RandomGrayscale(p=0.2),
                T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
            eval_tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
            root = "./data"
            pre = datasets.CIFAR10(root, True, download=True, transform=TwoCrop(ssl_tfm))
            tr = datasets.CIFAR10(root, True, download=True, transform=eval_tfm)
            te = datasets.CIFAR10(root, False, download=True, transform=eval_tfm)
            print("[data] CIFAR-10 載入成功")
            mk = lambda d, sh, dl: DataLoader(d, args.batch_size, shuffle=sh,
                                              num_workers=0, drop_last=dl,
                                              pin_memory=torch.cuda.is_available())
            return mk(pre, True, True), mk(tr, True, False), mk(te, False, False)
        except Exception as e:
            print(f"[data] CIFAR-10 載入失敗（{type(e).__name__}），改用合成資料")

    from torch.utils.data import TensorDataset
    pre_ds = SyntheticTwoCrop(2000, seed=args.seed)
    X = torch.stack([pre_ds.proto[y] + 0.5 * torch.randn(3, 32, 32) for y in pre_ds.y])
    lin_ds = TensorDataset(X, pre_ds.y)
    mk = lambda d, sh, dl: DataLoader(d, args.batch_size, shuffle=sh, drop_last=dl)
    print("[data] 使用合成資料")
    return mk(pre_ds, True, True), mk(lin_ds, True, False), mk(lin_ds, False, False)


# ============================================================
# 4. 評估
# ============================================================
@torch.no_grad()
def knn_eval(encoder, train_loader, test_loader, device, k=20, T=0.1):
    encoder.eval()
    feats, labels = [], []
    for x, y in train_loader:
        feats.append(F.normalize(encoder(x.to(device)), dim=1).cpu()); labels.append(y)
    Ftr, ytr = torch.cat(feats), torch.cat(labels)
    n_cls = int(ytr.max()) + 1
    correct = total = 0
    for x, y in test_loader:
        f = F.normalize(encoder(x.to(device)), dim=1).cpu()
        sim = f @ Ftr.t()
        k_eff = min(k, sim.size(1))
        tv, ti = sim.topk(k_eff, dim=1)
        scores = torch.zeros(len(f), n_cls).scatter_add_(1, ytr[ti], (tv / T).exp())
        correct += (scores.argmax(1) == y).sum().item(); total += len(y)
    return correct / total


def linear_probe(encoder, train_loader, test_loader, device, feat_dim,
                 n_cls=10, epochs=20, lr=1e-3):
    # ★ Linear probing：把 encoder 完全凍結，只訓練最後一層 Linear。
    #   這是評估「表徵品質」的標準協定 —— 表徵好的話，一條線就能把類別分開。
    for p in encoder.parameters():
        p.requires_grad_(False)           # 結尾的底線 = 就地修改
    # ★ 除了凍結參數，還要切 eval，否則 BatchNorm 的 running stats
    #   會被 forward 一路更新，等於偷偷用測試資料改了模型
    encoder.eval()                        # ★ 別忘了，否則 BN running stats 被污染
    clf = nn.Linear(feat_dim, n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(x)
            loss = F.cross_entropy(clf(h), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = clf(encoder(x.to(device))).argmax(1).cpu()
            correct += (pred == y).sum().item(); total += len(y)
    for p in encoder.parameters():
        p.requires_grad_(True)
    return correct / total


@torch.no_grad()
def collapse_metric(model, loader, device):
    """輸出向量各維度標準差；接近 0 就是 representation collapse。"""
    model.eval()
    zs = []
    for batch in loader:
        (x1, _), _ = batch if isinstance(batch[0], (list, tuple)) else ((batch[0], None), batch[1])
        z = model.encoder(x1.to(device))
        zs.append(F.normalize(z, dim=1).cpu())
        if len(zs) * z.size(0) > 1024:
            break
    Z = torch.cat(zs)
    return Z.std(0).mean().item(), 1 / Z.size(1) ** 0.5


# ============================================================
# 5. 訓練
# ============================================================
def pretrain(args):
    set_seed(args.seed); speed_setup()
    device = get_device()
    pre_loader, lin_loader, test_loader = get_data(args)

    feat_dim = 256
    if args.method == "simclr":
        model = SimCLR(feat_dim, args.proj_dim).to(device)
    else:
        model = SimSiam(feat_dim, args.pred_dim, stopgrad=args.stopgrad).to(device)
        if not args.stopgrad:
            print("★ 已關閉 stop-gradient —— 預期會看到 collapse（std 掉到接近 0）")
    print_model_summary(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {"loss": [], "std": [], "knn": []}
    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for (x1, x2), _ in pre_loader:
            x1, x2 = x1.to(device), x2.to(device)
            if args.method == "simclr":
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent_loss(z1, z2, args.temperature)
            else:
                p1, p2, z1, z2 = model(x1, x2)
                loss = simsiam_loss(p1, p2, z1, z2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * x1.size(0); n += x1.size(0)
        sched.step()

        std, ideal = collapse_metric(model, pre_loader, device)
        hist["loss"].append(tot / n); hist["std"].append(std)
        msg = f"  ep {ep:3d}/{args.epochs} | loss {tot/n:.4f} | z_std {std:.4f} (理想~{ideal:.4f})"
        if std < ideal * 0.2:
            msg += "  <-- ★ collapse！"
        if ep % max(1, args.epochs // 5) == 0 or ep == args.epochs:
            acc = knn_eval(model.encoder, lin_loader, test_loader, device)
            hist["knn"].append(acc)
            msg += f" | kNN {acc:.2%}"
        print(msg)

    print("\n[評估] linear probing（凍結 encoder，只訓練一層 Linear）")
    acc_ssl = linear_probe(model.encoder, lin_loader, test_loader, device,
                           feat_dim, epochs=args.probe_epochs)
    print(f"  SSL 預訓練 encoder  -> {acc_ssl:.2%}")

    set_seed(args.seed + 1)
    rand_enc = make_encoder(feat_dim // 4).to(device)
    acc_rand = linear_probe(rand_enc, lin_loader, test_loader, device,
                            feat_dim, epochs=args.probe_epochs)
    print(f"  隨機初始化 encoder  -> {acc_rand:.2%}   （對照組）")
    print(f"  ★ 差距 = {acc_ssl - acc_rand:+.2%}  <-- 這就是 SSL 學到的東西")

    tag = f"{args.method}{'' if args.stopgrad else '_nostopgrad'}"
    save_json(OUT / f"ssl_{tag}" / "history.json",
              {**hist, "linear_probe_ssl": acc_ssl, "linear_probe_random": acc_rand})
    torch.save({"encoder": model.encoder.state_dict()},
               OUT / f"ssl_{tag}" / "encoder.pt")
    print(f"\nencoder 已存 -> {OUT / f'ssl_{tag}' / 'encoder.pt'}")


def loss_test():
    """驗證 InfoNCE 實作正確。"""
    print("=" * 66)
    torch.manual_seed(0)
    N, D = 8, 16
    # 完全相同的兩個 view -> loss 應該很小
    z = torch.randn(N, D)
    print(f"完全相同的兩個 view       loss = {nt_xent_loss(z, z.clone(), 0.5):.4f}   (應很小)")
    # 完全隨機 -> loss 應接近 log(2N-1)
    import math
    l = nt_xent_loss(torch.randn(N, D), torch.randn(N, D), 0.5)
    print(f"完全隨機的兩個 view       loss = {l:.4f}   (應接近 log(2N-1)={math.log(2*N-1):.4f})")
    # temperature 的影響
    for t in (0.05, 0.5, 5.0):
        print(f"  temperature={t:<5}  loss = {nt_xent_loss(z, z + 0.5*torch.randn_like(z), t):.4f}")
    print("★ temperature 越小，對「難負樣本」越敏感（梯度越集中）")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["simclr", "simsiam"], default="simclr")   # 選對比學習方法
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--pred-dim", type=int, default=64)
    ap.add_argument("--probe-epochs", type=int, default=20)
    # dest="stopgrad" + store_false：預設 args.stopgrad=True（正常訓練）；
    #   命令列寫 --no-stopgrad 時變 False，用來刻意示範 collapse
    ap.add_argument("--no-stopgrad", dest="stopgrad", action="store_false")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--loss-test", action="store_true")       # 只測 InfoNCE 公式本身
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.loss_test:
        loss_test()
    else:
        pretrain(args)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()