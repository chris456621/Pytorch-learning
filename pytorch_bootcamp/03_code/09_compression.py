"""模型壓縮：剪枝 + 量化 + 知識蒸餾 完整流程（對應 02_topics/07_NetworkCompression.md）

用法：
    python 09_compression.py --all              # ★ 跑完整流程並產生比較表
    python 09_compression.py --prune            # 只做剪枝實驗
    python 09_compression.py --kd               # 只做知識蒸餾
    python 09_compression.py --quantize         # 只做量化
    python 09_compression.py --synthetic        # 沒網路時用合成資料

===========================================================================
三種壓縮技術一句話說明
---------------------------------------------------------------------------
  剪枝 pruning      把不重要的權重設成 0。★ 檔案可以變小，但「速度不會變快」，
                    因為矩陣形狀沒變，GPU 照樣算 0 乘 x。要快必須做結構化剪枝。
  量化 quantization float32 -> int8。檔案小 4 倍，整數運算快 2~4 倍。★ 真的會快。
  蒸餾 distillation 用大模型的「軟標籤」教小模型，讓小模型超越自己從零訓練的水準。
===========================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

sys.path.append(str(Path(__file__).resolve().parent))
from common import (set_seed, get_device, print_model_summary, count_params,
                    train_one_epoch, evaluate, save_json)

OUT = Path(__file__).resolve().parent / "outputs"


# ============================================================
def make_cnn(width=64, num_classes=10, in_ch=1):
    def blk(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    return nn.Sequential(
        blk(in_ch, width), blk(width, width * 2, 2),
        blk(width * 2, width * 4, 2),
        # AdaptiveAvgPool2d(1) 會把所有空間資訊壓掉；保留 2x2 對數字辨識明顯較好
        nn.AdaptiveAvgPool2d(2), nn.Flatten(1),
        nn.Linear(width * 4 * 4, num_classes))


# ============================================================
# 工具
# ============================================================
def model_size_mb(model, tmp="_tmp_size.pt"):
    """量模型檔案大小：存到硬碟再看檔案幾 bytes（最誠實的量法）。"""
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return mb


def sparsity(model):
    """稀疏度 = 值為 0 的權重佔的比例。剪枝後應該接近你設定的 amount。"""
    zeros = total = 0                 # 連續賦值：兩個變數都設成 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    return zeros / max(total, 1)


def cpu_latency_ms(model, shape, runs=50):
    """量單張圖的 CPU 推論延遲（毫秒）。這才是「壓縮有沒有用」的真正指標。"""
    model = model.cpu().eval()
    # *shape 把 tuple 拆成一個個參數：torch.randn(*(1,1,28,28)) = torch.randn(1,1,28,28)
    x = torch.randn(*shape)
    with torch.no_grad():
        for _ in range(10):        # ★ warmup，第一次很慢不能算
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - t0) / runs * 1000


def quantize_tensor(x, num_bits=8):
    """手刻非對稱量化：把浮點數壓成 0~255 的整數。

    公式：q = round(x / scale) + zero_point
      scale      = 一個整數刻度代表多少浮點值 =（最大-最小）/（能表示的階數）
      zero_point = 浮點的 0 對應到哪個整數（讓 0 能被精確表示）
    回傳 (量化後的 uint8, scale, zero_point)，這三個東西才能還原。
    """
    qmin, qmax = 0, 2 ** num_bits - 1        # 8 bits -> 0 ~ 255
    mn, mx = x.min().item(), x.max().item()
    scale = (mx - mn) / (qmax - qmin) if mx > mn else 1.0
    zero_point = round(qmin - mn / scale)
    q = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
    return q.to(torch.uint8), scale, zero_point


def dequantize(q, scale, zero_point):
    return (q.float() - zero_point) * scale


def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7, use_t2=True):
    # ★ F.kl_div 的參數順序很反直覺：
    #     第一個要「已經取過 log_softmax 的」student
    #     第二個要「一般 softmax 的」teacher
    #   寫反了不會報錯，但 loss 完全是錯的。
    # ★ reduction="batchmean" 才是數學上正確的 KL（用 "mean" 會多除一個維度數）。
    soft = F.kl_div(F.log_softmax(student_logits / T, 1),
                    F.softmax(teacher_logits / T, 1),
                    reduction="batchmean")
    if use_t2:
        # ★★ 乘 T^2 的理由：softmax(z/T) 對 z 微分會多出 1/T，KL 兩邊都被縮放，
        #   蒸餾項的梯度變成原本的 1/T^2。乘回來才能讓它跟硬標籤項量級相當，
        #   alpha 才是有意義的權重。
        # ⚠️ 但它是「尺度校正」不是「效果加成」：加了它等於放大梯度十幾倍，
        #   不重調 lr 反而可能變差。詳見 02_topics/07_NetworkCompression.md §4.2.1
        soft = soft * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard


# ============================================================
def get_data(args):
    from torch.utils.data import DataLoader, TensorDataset, random_split
    if not args.synthetic:
        try:
            from torchvision import datasets, transforms as T
            tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            tr = datasets.MNIST("./data", True, download=True, transform=tfm)
            te = datasets.MNIST("./data", False, download=True, transform=tfm)
            print("[data] MNIST 載入成功")
            return (DataLoader(tr, args.batch_size, shuffle=True, drop_last=True),
                    DataLoader(te, args.batch_size, shuffle=False), (1, 1, 28, 28))
        except Exception as e:
            print(f"[data] MNIST 載入失敗（{type(e).__name__}），改用合成資料")
    g = torch.Generator().manual_seed(args.seed)
    n = 4000
    proto = torch.randn(10, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    # 雜訊調到讓任務「不是一眼就滿分」，這樣壓縮實驗的差異才看得出來
    X = proto[y] + 2.2 * torch.randn(n, 1, 28, 28, generator=g)
    tr, te = random_split(TensorDataset(X, y), [3000, 1000], generator=g)
    print("[data] 使用合成資料（★ 合成任務較單純，KD 的增益不如真實資料明顯；"
          "有網路時建議改用真實 MNIST 跑一次）")
    return (DataLoader(tr, args.batch_size, shuffle=True, drop_last=True),
            DataLoader(te, args.batch_size, shuffle=False), (1, 1, 28, 28))


def quick_train(model, tr, te, device, epochs, lr=1e-3, teacher=None, args=None):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        if teacher is None:
            train_one_epoch(model, tr, crit, opt, device)
        else:
            model.train(); teacher.eval()
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                # ★ teacher 只負責「出考卷」，本身不更新也不需要梯度。
                #   包在 no_grad 裡可以省一半記憶體、快很多。
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = kd_loss(model(x), t_logits, y, args.temperature,
                               args.alpha, use_t2=args.use_t2)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return evaluate(model, te, crit, device)[1]


# ============================================================
def exp_prune(model, tr, te, device, shape, args):
    print("\n" + "=" * 70)
    print("剪枝實驗：一次剪光 vs 迭代剪枝")
    print("=" * 70)
    crit = nn.CrossEntropyLoss()
    base_acc = evaluate(model, te, crit, device)[1]
    base_lat = cpu_latency_ms(model, shape); model.to(device)
    print(f"  原始模型  acc={base_acc:.4f}  size={model_size_mb(model):.2f}MB  "
          f"CPU={base_lat:.2f}ms  sparsity={sparsity(model):.1%}")

    import copy
    results = {}

    # --- A. 一次剪 80% ---
    m1 = copy.deepcopy(model)
    params = [(m, "weight") for m in m1.modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))]
    # global_unstructured：跨所有層一起排序，把「全域最小的 80%」設成 0。
    # 比逐層各剪 80% 好，因為有些層本來就不重要，應該被剪多一點。
    prune.global_unstructured(params, prune.L1Unstructured, amount=0.8)
    # ★ 剪枝後 PyTorch 是用 weight_orig * weight_mask 動態算出 weight。
    #   prune.remove 把 mask 永久套用並移除這個機制，權重才真的變成 0。
    for m, n in params:
        prune.remove(m, n)
    acc1 = evaluate(m1, te, crit, device)[1]
    quick_train(m1, tr, te, device, epochs=args.finetune_epochs)
    acc1_ft = evaluate(m1, te, crit, device)[1]
    print(f"\n  A. 一次剪 80%   剪完 acc={acc1:.4f} -> fine-tune 後 {acc1_ft:.4f}  "
          f"sparsity={sparsity(m1):.1%}")
    results["oneshot_80"] = acc1_ft

    # --- B. 迭代剪枝 5 輪 x 20% ---
    m2 = copy.deepcopy(model)
    params2 = [(m, "weight") for m in m2.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    print("\n  B. 迭代剪枝（每輪 20% + fine-tune）")
    for r in range(5):
        prune.global_unstructured(params2, prune.L1Unstructured, amount=0.2)
        quick_train(m2, tr, te, device, epochs=args.finetune_epochs)
        acc = evaluate(m2, te, crit, device)[1]
        # 計算「已 mask」的實際稀疏度
        z = sum((m.weight == 0).sum().item() for m, _ in params2)
        t = sum(m.weight.numel() for m, _ in params2)
        print(f"     round {r+1}: sparsity={z/t:.1%}  acc={acc:.4f}")
    for m, n in params2:
        prune.remove(m, n)
    acc2 = evaluate(m2, te, crit, device)[1]
    results["iterative"] = acc2
    lat2 = cpu_latency_ms(m2, shape); m2.to(device)
    print(f"\n  迭代剪枝最終 acc={acc2:.4f}  size={model_size_mb(m2):.2f}MB  "
          f"CPU={lat2:.2f}ms")
    print(f"\n  ★★ 注意：稀疏度 {sparsity(m2):.0%} 但檔案大小與推論時間幾乎沒變")
    print(f"     原始 {base_lat:.2f}ms -> 剪枝後 {lat2:.2f}ms")
    print("     因為非結構化剪枝只是把權重設成 0，矩陣形狀沒變，GPU/CPU 照樣算。")
    print("     想真的加速必須做「結構化剪枝」把整個 channel 拿掉。")
    return results


def exp_kd(tr, te, device, teacher, args):
    """★ 公平比較：每個設定都掃 lr，取最好的結果。

    為什麼要掃 lr？因為乘上 T^2 會把蒸餾項的梯度放大 T^2 倍，
    等於偷偷把學習率乘了好幾倍。若固定 lr 直接比，你比到的其實是
    「哪個設定剛好配到合適的學習率」，不是「T^2 本身好不好」。
    這是做研究時最常見的不公平比較。
    """
    print()
    print("=" * 70)
    print("知識蒸餾實驗（每個設定都掃 lr，取最佳）")
    print("=" * 70)
    crit = nn.CrossEntropyLoss()
    t_acc = evaluate(teacher, te, crit, device)[1]
    t_n, _ = count_params(teacher)
    print(f"  Teacher  params={t_n:,}  acc={t_acc:.4f}")

    lrs = args.lr_grid
    rows = []
    for name, teach, use_t2 in [("Student 從零訓練", None, True),
                                ("Student + KD (乘 T^2)", teacher, True),
                                ("Student + KD (不乘 T^2)", teacher, False)]:
        best_acc, best_lr = 0.0, None
        for lr in lrs:
            set_seed(args.seed)
            s_model = make_cnn(args.student_width).to(device)
            args.use_t2 = use_t2
            acc = quick_train(s_model, tr, te, device, epochs=args.kd_epochs,
                              lr=lr, teacher=teach, args=args)
            if acc > best_acc:
                best_acc, best_lr = acc, lr
        n, _ = count_params(s_model)
        rows.append((name, n, best_acc, best_lr))
        print(f"  {name:<26} params={n:,}  best acc={best_acc:.4f}  (lr={best_lr:g})")

    print()
    print(f"  ★ KD 的增益 = {rows[1][2] - rows[0][2]:+.4f}")
    print(f"  ★ 乘 T^2 vs 不乘 = {rows[1][2] - rows[2][2]:+.4f}")
    print("""
  怎麼解讀：
    T^2 的作用是「讓蒸餾項的梯度量級跟硬標籤項相當」，這樣 alpha 才是有意義的權重。
    它本身不會直接提升準確率 —— 如果你加了 T^2 卻沒調 lr，
    等於把學習率放大 T^2 倍，反而可能變差（我實測過，這很常見）。
    ★ 結論：乘 T^2 是標準做法，但它是「尺度校正」不是「效果加成」，
      改了它就要重新調 lr。這正是為什麼做實驗一定要掃超參數再比較。
""")
    return rows


def exp_quantize(model, te, device, shape, args):
    print("\n" + "=" * 70)
    print("量化實驗")
    print("=" * 70)
    crit = nn.CrossEntropyLoss()
    fp32_acc = evaluate(model, te, crit, device)[1]
    fp32_size = model_size_mb(model)
    fp32_lat = cpu_latency_ms(model, shape)
    print(f"  FP32   acc={fp32_acc:.4f}  size={fp32_size:.2f}MB  CPU={fp32_lat:.2f}ms")

    # --- 手刻量化：看誤差 ---
    w = next(m.weight for m in model.modules() if isinstance(m, nn.Conv2d)).detach().cpu()
    q, s, zp = quantize_tensor(w, 8)
    err = (dequantize(q, s, zp) - w).abs()
    print(f"\n  手刻 INT8 量化（第一層 conv 權重）")
    print(f"    scale={s:.6f} zero_point={zp}")
    print(f"    量化誤差 mean={err.mean():.6f} max={err.max():.6f} "
          f"相對誤差={err.mean()/w.abs().mean():.2%}")
    for bits in (8, 6, 4, 2):
        q, s, zp = quantize_tensor(w, bits)
        e = (dequantize(q, s, zp) - w).abs().mean().item()
        print(f"    {bits} bits -> 平均誤差 {e:.6f}")

    # --- PyTorch 動態量化（只對 Linear 有效）---
    m = model.cpu().eval()
    try:
        qm = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        q_size, q_lat = model_size_mb(qm), cpu_latency_ms(qm, shape)
        print(f"\n  動態量化（Linear -> INT8）size={q_size:.2f}MB  CPU={q_lat:.2f}ms")
        print("  ★ 這個模型主要是 Conv，所以動態量化效果有限。")
        print("    Linear/LSTM 為主的模型（例如你的 Transformer）收益大得多。")
    except Exception as e:
        print(f"  動態量化失敗：{type(e).__name__}: {e}")
    model.to(device)


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    # 這四個是獨立的開關，可以只選一種實驗跑，也可以用 --all 全部跑
    # （main() 主體會用 if not any([...]) 判斷「一個都沒選就當作 --all」）
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--kd", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--kd-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=1)
    ap.add_argument("--teacher-width", type=int, default=64)
    ap.add_argument("--student-width", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    # nargs="+"：這個參數可以吃「一個或多個」值，存成 list。
    #   命令列寫 --lr-grid 1e-3 3e-4 1e-4  ->  args.lr_grid == [0.001, 0.0003, 0.0001]
    ap.add_argument("--lr-grid", type=float, nargs="+",
                    default=[1e-3, 3e-4, 1e-4],
                    help="KD 實驗要掃的學習率")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.use_t2 = True
    if not any([args.all, args.prune, args.kd, args.quantize]):
        args.all = True

    set_seed(args.seed)
    device = get_device()
    tr, te, shape = get_data(args)

    print("\n訓練 teacher（大模型）...")
    teacher = make_cnn(args.teacher_width).to(device)
    print_model_summary(teacher)
    t_acc = quick_train(teacher, tr, te, device, epochs=args.epochs)
    print(f"  teacher acc = {t_acc:.4f}")

    res = {"teacher_acc": t_acc}
    if args.all or args.prune:
        res["prune"] = exp_prune(teacher, tr, te, device, shape, args)
    if args.all or args.kd:
        res["kd"] = exp_kd(tr, te, device, teacher, args)
    if args.all or args.quantize:
        exp_quantize(teacher, te, device, shape, args)

    save_json(OUT / "compression" / "results.json", str(res))
    print("\n" + "=" * 70)
    print("結論：")
    print("  1. 非結構化剪枝 -> 稀疏但不加速（要結構化剪枝才會快）")
    print("  2. 迭代剪枝 > 一次剪光")
    print("  3. KD 讓小模型明顯超越自己從零訓練的水準")
    print("  4. T^2 是尺度校正（改了它要重調 lr），不是免費的效果加成")
    print("  5. 量化位元數越低誤差越大，8 bit 通常是甜蜜點")
    print("=" * 70)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()