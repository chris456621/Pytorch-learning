"""手刻 Transformer：Attention -> MHA -> Encoder -> 序列任務（對應 02_topics/03_Transformer.md）

用法：
    python 05_transformer_scratch.py --verify     # ★ 先跑這個：驗證每個元件正確
    python 05_transformer_scratch.py --task copy  # 訓練「反轉序列」任務
    python 05_transformer_scratch.py --compare    # 手刻 vs nn.TransformerEncoder

===========================================================================
本檔的 shape 慣例（每個變數旁的註解都照這個寫）
---------------------------------------------------------------------------
  B  batch size（一次處理幾筆）
  L  sequence length（序列長度／幾個時間步）
  D  d_model（每個 token 的特徵維度）
  h  nhead（注意力頭數）
  d  每個頭的維度 = D // h
===========================================================================
"""


import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from common import set_seed, get_device, print_model_summary, timer


# ============================================================
# 1. Scaled Dot-Product Attention
# ============================================================
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """Attention 的核心公式：softmax(Q K^T / sqrt(d)) V

    q,k,v: (B, h, L, d)   mask: 可廣播到 (B, h, Lq, Lk)，True = 遮蔽

    直覺：Q 是「我想找什麼」，K 是「我有什麼特徵」，V 是「我提供的內容」。
         QK^T 算出每個位置對每個位置的相似度，softmax 變成權重，再對 V 加權平均。
    """
    d_k = q.size(-1)                 # size(-1) = 最後一維的長度
    # k.transpose(-2, -1) 交換最後兩維：(B,h,L,d) -> (B,h,d,L)
    # 再 @ 做矩陣乘法：(B,h,Lq,d) @ (B,h,d,Lk) -> (B,h,Lq,Lk)
    # ★ 除以 sqrt(d_k)：不除的話分數變異數是 d_k，softmax 會極度尖銳、梯度消失
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # masked_fill：mask 為 True 的位置填成 -inf，softmax 後就會變成 0
        scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)    # dim=-1：對「每一列」做 softmax，每列加起來=1
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v, attn


# ============================================================
# 2. Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必須能被 nhead({nhead}) 整除")
        self.h, self.d = nhead, d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x):
        """把 (B,L,D) 拆成多頭 (B,h,L,d)。這兩步是 MHA 最容易寫錯的地方。"""
        B, L, _ = x.shape             # 解包，_ 代表「這個值我不需要」
        # ① view(B,L,h,d)：把最後一維 D 切成 h 份，每份 d
        # ② transpose(1,2)：把 head 移到 batch 旁邊，之後 attention 就能平行對每個頭做
        return x.view(B, L, self.h, self.d).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B, Lq, D = query.shape
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        out, attn = scaled_dot_product_attention(
            q, k, v, mask, self.dropout if self.training else 0.0)
        # ★ 合併多頭：先把 head 移回去 (B,h,L,d) -> (B,L,h,d)，再併成 (B,L,D)
        #   transpose 之後記憶體不連續，用 view 會報
        #   "view size is not compatible with input tensor's size and stride"
        #   所以一定要用 reshape（或先 .contiguous().view(...)）
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out), attn


# ============================================================
# 3. 位置編碼
# ============================================================
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        # [:, 0::2] = 所有列、從第 0 欄開始每隔 2 欄 = 偶數欄
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)          # 奇數欄
        # ★ register_buffer：這個 tensor 不是可訓練參數，但要「跟著模型搬到 GPU」
        #   也會被存進 state_dict。直接寫 self.pe = ... 的話 model.cuda() 不會搬它，
        #   forward 時就會噴 "Expected all tensors to be on the same device"
        # pe[None] 在最前面加一維 -> (1, max_len, d_model)，方便跟 (B,L,D) 廣播
        self.register_buffer("pe", pe[None])

    def forward(self, x):                          # (B, L, D)
        return x + self.pe[:, :x.size(1)]


# ============================================================
# 4. Encoder Layer（Pre-LN）
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, norm_first=True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, mask=None):
        # Pre-LN vs Post-LN 的差別（現代一律用 Pre-LN）：
        #   Post-LN（原論文）: x = LayerNorm(x + Sublayer(x))
        #   Pre-LN（現代）  : x = x + Sublayer(LayerNorm(x))
        # Pre-LN 的殘差路徑上沒有 LayerNorm，梯度可以無阻礙直達淺層，
        # 所以不需要 warmup 也能訓練、深層時穩定得多。
        if self.norm_first:
            h = self.norm1(x)
            a, _ = self.attn(h, h, h, mask)
            x = x + self.drop(a)
            x = x + self.drop(self.ff(self.norm2(x)))
        else:                                            # Post-LN（原論文）
            a, _ = self.attn(x, x, x, mask)
            x = self.norm1(x + self.drop(a))
            x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ============================================================
# 5. Mask
# ============================================================
def causal_mask(L, device):
    """上三角（不含對角線）為 True = 遮掉未來。

    L=4 時長這樣（True 代表「看不到」）：
            k0     k1     k2     k3
      q0  False  True   True   True     位置 0 只能看自己
      q1  False  False  True   True
      q2  False  False  False  True
      q3  False  False  False  False    位置 3 可以看全部

    ★ device=device 一定要傳：不傳的話 mask 建在 CPU，跟 GPU 上的 scores 相加會報錯。
    """
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)


def padding_mask(lengths, max_len):
    """lengths: (B,) -> (B, 1, 1, max_len)，True 代表 padding。"""
    idx = torch.arange(max_len, device=lengths.device)
    return (idx[None, :] >= lengths[:, None])[:, None, None, :]


# ============================================================
# 6. 完整模型：序列到序列（用 Encoder + causal mask 做自迴歸）
# ============================================================
class SeqModel(nn.Module):
    """把長度 L 的整數序列反轉。用 encoder-only + causal mask 訓練。"""

    def __init__(self, vocab, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, max_len=64, dropout=0.1, causal=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pe = SinusoidalPE(d_model, max_len)
        self.layers = nn.ModuleList([              # ★ 一定要用 ModuleList
            EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.causal = causal

    def forward(self, x):                          # x: (B, L) int64
        L = x.size(1)
        h = self.pe(self.embed(x))                 # (B, L, D)
        m = causal_mask(L, x.device)[None, None] if self.causal else None
        for layer in self.layers:
            h = layer(h, m)
        return self.head(self.norm(h))             # (B, L, vocab)


# ============================================================
# 7. 驗證（★ 一定要跑）
# ============================================================
def verify():
    print("=" * 68)
    print("元件驗證")
    print("=" * 68)
    set_seed(0)
    B, L, D, h = 2, 6, 32, 4
    d = D // h

    # (1) 手刻 attention vs PyTorch 內建
    q = torch.randn(B, h, L, d)
    mine, _ = scaled_dot_product_attention(q, q, q)
    ref = F.scaled_dot_product_attention(q, q, q)
    print(f"[1] 手刻 attention == F.scaled_dot_product_attention : "
          f"{torch.allclose(mine, ref, atol=1e-5)}")

    # (2) MHA shape 流
    mha = MultiHeadAttention(D, h, dropout=0.0).eval()
    x = torch.randn(B, L, D)
    out, attn = mha(x, x, x)
    print(f"[2] MHA 輸入 {tuple(x.shape)} -> 輸出 {tuple(out.shape)}, "
          f"attn {tuple(attn.shape)}   (應為 (B,h,L,L))")
    assert out.shape == x.shape and attn.shape == (B, h, L, L)

    # (3) causal mask 真的擋住未來了嗎
    # [None, None] 在前面加兩個維度 (L,L) -> (1,1,L,L)，才能跟 (B,h,L,L) 廣播
    m = causal_mask(L, x.device)[None, None]
    _, a = mha(x, x, x, m)             # _ 表示「第一個回傳值我不需要」
    upper = a[0, 0].triu(1)            # 取第 0 筆第 0 個頭的上三角
    print(f"[3] causal mask 後上三角注意力總和 = {upper.sum().item():.2e}  (應為 0)")
    assert upper.sum().item() < 1e-6

    # (4) 端到端因果性驗證：改動未來的 token 不能影響現在的輸出
    model = SeqModel(vocab=20, d_model=D, nhead=h, num_layers=2, causal=True).eval()
    xi = torch.randint(0, 20, (1, L))
    with torch.no_grad():
        o1 = model(xi)
        x2 = xi.clone(); x2[0, -1] = (x2[0, -1] + 7) % 20
        o2 = model(x2)
    diff = (o1[:, :-1] - o2[:, :-1]).abs().max().item()
    print(f"[4] 改最後一個 token，前面位置輸出的最大變化 = {diff:.2e}  (應接近 0)")
    assert diff < 1e-5, "因果性被破壞！mask 寫錯了"

    # (5) 沒有位置編碼會怎樣
    class NoPE(SeqModel):
        def forward(self, x):
            h_ = self.embed(x)                     # ★ 故意不加 PE
            for layer in self.layers:
                h_ = layer(h_, None)
            return self.head(self.norm(h_))

    nope = NoPE(vocab=20, d_model=D, nhead=h, num_layers=1, causal=False).eval()
    with torch.no_grad():
        a1 = nope(xi)
        perm = torch.randperm(L)
        a2 = nope(xi[:, perm])
    same = torch.allclose(a1[:, perm], a2, atol=1e-5)
    print(f"[5] 沒有 PE 時，打亂輸入 = 打亂輸出（置換等變）: {same}")
    print("    ★ 這證明 self-attention 本身不知道順序，位置編碼是必要的")

    # (6) 除以 sqrt(d_k) 的影響
    for scale in (1.0, math.sqrt(d)):
        s = (q @ q.transpose(-2, -1) / scale)
        p = s.softmax(-1)
        ent = -(p * (p + 1e-12).log()).sum(-1).mean()
        print(f"[6] 除以 {scale:5.2f} -> softmax 平均熵 = {ent:.4f}  "
              f"({'太尖銳，梯度會消失' if scale == 1.0 else '分布合理'})")

    print("\n全部元件驗證通過。")


# ============================================================
# 8. 訓練一個真實任務：把序列反轉
# ============================================================
def make_batch(bs, L, vocab, device, g):
    """輸入 x，目標 y = x 反轉。用 causal 模型時我們把它當成
    「讀完整個序列後，每個位置預測對應的反轉字元」的簡化練習。"""
    x = torch.randint(1, vocab, (bs, L), generator=g).to(device)
    y = x.flip(1)
    return x, y


def train_task(args):
    set_seed(args.seed)
    device = get_device()
    g = torch.Generator().manual_seed(args.seed)
    vocab, L = 20, args.seq_len

    model = SeqModel(vocab, d_model=args.d_model, nhead=args.nhead,
                     num_layers=args.layers, dim_ff=args.d_model * 4,
                     max_len=L + 8, causal=False).to(device)   # 反轉需要看到全部
    print_model_summary(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.steps
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))

    print(f"\n訓練「序列反轉」任務  vocab={vocab} L={L} steps={total}")
    model.train()
    for step in range(1, total + 1):
        x, y = make_batch(args.batch_size, L, vocab, device, g)
        logits = model(x)                                  # (B, L, vocab)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ★ Transformer 必備
        opt.step(); sched.step()

        if step % max(1, total // 10) == 0 or step == 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"  step {step:5d}/{total} | loss {loss.item():.4f} | "
                  f"acc {acc:.2%} | lr {opt.param_groups[0]['lr']:.2e}")

    model.eval()
    with torch.no_grad():
        x, y = make_batch(4, L, vocab, device, g)
        pred = model(x).argmax(-1)
    print(f"\n範例（前 3 筆，只顯示前 10 個位置）:")
    for i in range(3):
        print(f"  in   {x[i, :10].tolist()}")
        print(f"  want {y[i, :10].tolist()}")
        print(f"  pred {pred[i, :10].tolist()}   "
              f"{'OK' if torch.equal(pred[i], y[i]) else 'X'}")
    print(f"\n整體正確率 {(pred == y).float().mean().item():.2%}")


# ============================================================
# 9. 手刻 vs 內建：正確性與速度
# ============================================================
def compare(args):
    set_seed(0)
    device = get_device()
    B, L, D, h = 32, 128, 256, 8
    x = torch.randn(B, L, D, device=device)

    mine = EncoderLayer(D, h, D * 4, dropout=0.0).to(device).eval()
    builtin = nn.TransformerEncoderLayer(
        D, h, D * 4, dropout=0.0, activation="gelu",
        batch_first=True,        # ★★ 一定要設，否則輸入是 (L,B,D)
        norm_first=True,         # ★★ Pre-LN
    ).to(device).eval()

    print("=" * 68)
    print("手刻 vs nn.TransformerEncoderLayer")
    print("=" * 68)
    print(f"手刻   參數量 {sum(p.numel() for p in mine.parameters()):,}")
    print(f"內建   參數量 {sum(p.numel() for p in builtin.parameters()):,}")
    print("（參數量應該相同 —— 結構是一樣的，只是實作不同）\n")

    with torch.no_grad():
        with timer("手刻 x100"):
            for _ in range(100):
                mine(x)
        with timer("內建 x100"):
            for _ in range(100):
                builtin(x)
    print("\n★ 內建版本用了 fused kernel 與 SDPA（含 FlashAttention），實務上請用它。")
    print("  手刻的價值在於「你知道它在做什麼」。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")     # 開關：只跑元件驗證
    ap.add_argument("--compare", action="store_true")    # 開關：手刻 vs 內建版本比較
    # choices 裡放 None 是合法的：代表「不指定這個參數」也算一個有效選項
    ap.add_argument("--task", default=None, choices=[None, "copy"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.compare:
        compare(args)
    elif args.task:
        train_task(args)
    else:
        verify()
        if not args.verify:
            print("\n" + "=" * 68)
            train_task(args)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()