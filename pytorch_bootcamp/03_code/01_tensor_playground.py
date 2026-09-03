"""張量語法自我測驗 —— 會自動幫你對答案。

用法：
    python 01_tensor_playground.py            # 看解答與說明
    python 01_tensor_playground.py --quiz     # 只印題目，自己先想

每一題都對應 01_syntax/05_tensor基礎.md 與 06_shape操作大全.md。

===========================================================================
本檔用到的 Python 語法
---------------------------------------------------------------------------
  argparse                 讓程式可以吃命令列參數（--quiz）
  f"..."                   字串格式化                     -> 解碼器 §1
  tuple(x.shape)           把 torch.Size 轉成好讀的 tuple
  [round(v,3) for v in ..] list 推導式                     -> 解碼器 §3
  x.data_ptr()             這個 tensor 的記憶體位址（用來判斷是否共享）
  try / except RuntimeError  預期會出錯時，抓住錯誤而不讓程式中斷
===========================================================================
"""

import argparse
import torch
import torch.nn.functional as F

PASS, FAIL = "  [OK]", "  [XX]"


def check(desc, got, want):
    """比對答案並印出結果。got=你的答案，want=正確答案。"""
    ok = (got == want)
    print(f"{PASS if ok else FAIL} {desc}\n        got={got}  want={want}")
    return ok


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def q1_view_vs_reshape():
    section("Q1  view / reshape / permute --- 什麼時候 view 會失敗？")
    x = torch.arange(24).reshape(2, 3, 4)
    print(f"x.shape={tuple(x.shape)}  stride={x.stride()}  "
          f"contiguous={x.is_contiguous()}")

    y = x.permute(1, 0, 2)
    print(f"permute(1,0,2) 後: shape={tuple(y.shape)} stride={y.stride()} "
          f"contiguous={y.is_contiguous()}   <-- 不連續了")

    # try/except：預期這行會拋 RuntimeError，抓住它而不是讓程式中斷
    try:
        y.view(6, 4)
        print(FAIL + " y.view(6,4) 竟然成功了？")
    except RuntimeError as e:      # as e 把錯誤物件綁到變數 e，才能印訊息
        print(PASS + f" y.view(6,4) 如預期失敗: {str(e)[:58]}...")

    check("y.reshape(6,4).shape", tuple(y.reshape(6, 4).shape), (6, 4))
    check("y.contiguous().view(6,4).shape",
          tuple(y.contiguous().view(6, 4).shape), (6, 4))
    print("\n★ 結論：permute/transpose 之後要改形狀，一律用 reshape。")


def q2_broadcasting():
    section("Q2  Broadcasting --- 不看答案先說出結果 shape")
    cases = [
        ((3, 1), (1, 4), (3, 4)),
        ((3, 1, 5), (4, 1), (3, 4, 5)),
        ((8, 3, 32, 32), (3, 1, 1), (8, 3, 32, 32)),
        ((32, 10), (10,), (32, 10)),
        ((32, 1), (32,), (32, 32)),      # ★ 靜默廣播災難！
    ]
    # cases 裡每個元素是 3 個東西的 tuple，for 迴圈直接解包成三個變數
    for sa, sb, want in cases:
        # 建兩個指定形狀的零張量相加，看廣播後變成什麼形狀
        got = tuple((torch.zeros(sa) + torch.zeros(sb)).shape)
        note = "  <-- ★ 這就是靜默廣播 bug！" if want == (32, 32) else ""
        check(f"{sa} + {sb}{note}", got, want)
    print("\n★ 最後一題是迴歸任務最常見的 bug：pred(32,1) 對 target(32,)")
    print("  防禦寫法： assert pred.shape == target.shape")


def q3_dim_reduction():
    section("Q3  dim 與 keepdim --- dim=k 是「消掉第 k 維」")
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    check("x.sum(dim=0).shape", tuple(x.sum(dim=0).shape), (3, 4))
    check("x.sum(dim=1).shape", tuple(x.sum(dim=1).shape), (2, 4))
    check("x.sum(dim=-1).shape", tuple(x.sum(dim=-1).shape), (2, 3))
    check("x.sum(dim=(0,1)).shape", tuple(x.sum(dim=(0, 1)).shape), (4,))
    check("x.sum(dim=1, keepdim=True).shape",
          tuple(x.sum(dim=1, keepdim=True).shape), (2, 1, 4))

    v, i = x.max(dim=1)
    check("x.max(dim=1) 回傳 (values, indices)",
          (tuple(v.shape), tuple(i.shape)), ((2, 4), (2, 4)))

    def my_softmax(t, dim=-1):
        # 手刻 softmax。dim=-1 代表「最後一維」（負數從尾巴數）
        # .values 是因為 max(dim=...) 回傳 (values, indices) 兩個東西
        # keepdim=True 讓結果保持同樣的維度數，才能跟原張量相減（廣播）
        t = t - t.max(dim=dim, keepdim=True).values      # 減最大值防止 exp 溢位
        e = t.exp()
        return e / e.sum(dim=dim, keepdim=True)

    ok = torch.allclose(my_softmax(x), x.softmax(-1), atol=1e-6)
    print(f"{PASS if ok else FAIL} 手刻 softmax 與 torch.softmax 一致：{ok}")


def q4_indexing():
    section("Q4  索引：view vs copy / gather / masked_fill")
    a = torch.arange(10)
    b = a[2:5]                     # 基本切片 -> view
    b[0] = 999
    check("基本切片是 view（改 b 會影響 a）", a[2].item(), 999)

    a = torch.arange(10)
    c = a[[2, 3, 4]]               # fancy indexing -> copy
    c[0] = -1
    check("fancy indexing 是 copy（改 c 不影響 a）", a[2].item(), 2)

    logits = torch.tensor([[1., 5., 3.], [9., 2., 7.], [4., 4., 8.]])
    labels = torch.tensor([1, 0, 2])
    # gather(dim, index)：沿 dim=1（欄）取值，index 的形狀要跟 logits 維度數相同
    #   labels[:, None] 把 (3,) 變成 (3,1)  -> 解碼器 §13
    #   .squeeze(1) 再把結果從 (3,1) 壓回 (3,)
    got = logits.gather(1, labels[:, None]).squeeze(1)
    check("gather 取出正確類別的分數", got.tolist(), [5.0, 9.0, 8.0])
    got2 = logits[torch.arange(3), labels]
    check("等價的 fancy indexing 寫法", got2.tolist(), [5.0, 9.0, 8.0])

    L = 4
    # torch.triu = upper triangle，取上三角；diagonal=1 代表「不含對角線」
    # 得到的 bool 矩陣 True 的位置就是「未來」，要遮掉
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    attn = torch.zeros(L, L).masked_fill(mask, float("-inf")).softmax(-1)
    check("causal mask 後第 0 列只看得到位置 0",
          [round(v, 3) for v in attn[0].tolist()], [1.0, 0.0, 0.0, 0.0])
    check("最後一列可以看到全部（均勻）",
          [round(v, 3) for v in attn[-1].tolist()], [0.25, 0.25, 0.25, 0.25])


def q5_mha_shape_flow():
    section("Q5  Multi-Head Attention 的完整 shape 流 ★")
    B, L, D, h = 2, 7, 64, 8
    d = D // h
    x = torch.randn(B, L, D)
    print(f"  輸入 (B,L,D)                = {tuple(x.shape)}")

    q = x.view(B, L, h, d);   print(f"  view(B,L,h,d)               = {tuple(q.shape)}")
    q = q.transpose(1, 2);    print(f"  transpose(1,2) -> (B,h,L,d) = {tuple(q.shape)}")

    k = v = q
    scores = q @ k.transpose(-2, -1) / d ** 0.5
    print(f"  scores (B,h,L,L)            = {tuple(scores.shape)}")
    out = scores.softmax(-1) @ v
    print(f"  attn @ v (B,h,L,d)          = {tuple(out.shape)}")
    out = out.transpose(1, 2)
    print(f"  transpose 回來 (B,L,h,d)    = {tuple(out.shape)}")
    out = out.reshape(B, L, D)
    print(f"  reshape (B,L,D)             = {tuple(out.shape)}")
    check("最終 shape", tuple(out.shape), (B, L, D))

    q4 = torch.randn(B, h, L, d)
    ref = F.scaled_dot_product_attention(q4, q4, q4)
    mine = (q4 @ q4.transpose(-2, -1) / d ** 0.5).softmax(-1) @ q4
    ok = torch.allclose(ref, mine, atol=1e-5)
    print(f"{PASS if ok else FAIL} 手刻 attention 與 F.scaled_dot_product_attention 一致：{ok}")


def q6_cat_stack_expand():
    section("Q6  cat / stack / expand / repeat")
    a, b = torch.zeros(2, 3), torch.ones(2, 3)
    check("cat(dim=0)", tuple(torch.cat([a, b], 0).shape), (4, 3))
    check("cat(dim=1)", tuple(torch.cat([a, b], 1).shape), (2, 6))
    check("stack(dim=0)  <- 多一維", tuple(torch.stack([a, b], 0).shape), (2, 2, 3))
    check("stack(dim=-1)", tuple(torch.stack([a, b], -1).shape), (2, 3, 2))

    x = torch.randn(1, 3)
    e, r = x.expand(4, 3), x.repeat(4, 1)
    check("expand 與 repeat 的 shape 相同",
          (tuple(e.shape), tuple(r.shape)), ((4, 3), (4, 3)))
    print(f"  expand 共享記憶體: data_ptr 相同 = {e.data_ptr() == x.data_ptr()}")
    print(f"  repeat 複製記憶體: data_ptr 相同 = {r.data_ptr() == x.data_ptr()}")
    print("  ★ 優先用 expand 省記憶體；需要修改結果時才用 repeat")


def q7_einsum():
    section("Q7  einsum")
    A = torch.randn(3, 4); Bm = torch.randn(4, 5)
    check("矩陣乘法", torch.allclose(torch.einsum("ij,jk->ik", A, Bm), A @ Bm), True)
    check("轉置", torch.allclose(torch.einsum("ij->ji", A), A.t()), True)

    X = torch.randn(8, 3, 4); Y = torch.randn(8, 4, 5)
    check("batch 矩陣乘法",
          torch.allclose(torch.einsum("bij,bjk->bik", X, Y), X @ Y, atol=1e-5), True)

    Q = torch.randn(2, 4, 6, 16); K = torch.randn(2, 4, 6, 16)
    s1 = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    s2 = Q @ K.transpose(-2, -1)
    check("attention 分數兩種寫法一致", torch.allclose(s1, s2, atol=1e-5), True)


def q8_autograd():
    section("Q8  Autograd --- 梯度會流到哪裡？")
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    (x ** 2).sum().backward()
    check("d(sum(x^2))/dx = 2x", x.grad.tolist(), [2.0, 4.0, 6.0])

    x2 = torch.tensor(2.0, requires_grad=True)
    (x2 ** 2).backward()
    g1 = x2.grad.item()
    (x2 ** 3).backward()
    check("梯度是累加的（4 然後 4+12=16）", (g1, x2.grad.item()), (4.0, 16.0))

    a = torch.tensor(2.0, requires_grad=True)
    b = (a * 3).detach() * 4
    check("detach 後 requires_grad", b.requires_grad, False)

    # 混合路徑：SimSiam / 知識蒸餾的核心情境
    enc_w = torch.tensor(1.0, requires_grad=True)
    feat = enc_w * torch.ones(4)
    out1 = (feat * 2).sum()
    out2 = (feat.detach() * 5).sum()
    (out1 + out2).backward()
    check("只有未 detach 的路徑貢獻梯度 (4*2=8)", enc_w.grad.item(), 8.0)

    # straight-through estimator
    z = torch.tensor([1.3, 2.7], requires_grad=True)
    # Straight-through estimator（量化 / VQ-VAE 的核心技巧）：
    #   forward 時：z + (round(z) - z) = round(z)      -> 值是四捨五入後的
    #   backward 時：括號被 detach 視為常數，導數只剩 z 的 1  -> 梯度可以穿透
    ste = z + (torch.round(z) - z).detach()
    check("STE 的 forward 值 = round(z)", ste.tolist(), [1.0, 3.0])
    ste.sum().backward()
    check("STE 的 backward 梯度 = 1", z.grad.tolist(), [1.0, 1.0])


QUESTIONS = [q1_view_vs_reshape, q2_broadcasting, q3_dim_reduction, q4_indexing,
             q5_mha_shape_flow, q6_cat_stack_expand, q7_einsum, q8_autograd]

QUIZ_TEXT = """
自我測驗（先自己想，再跑 python 01_tensor_playground.py 對答案）

Q1  x = torch.arange(24).reshape(2,3,4)
    x.permute(1,0,2).view(6,4) 會成功嗎？為什麼？該怎麼改？
Q2  說出下列結果的 shape：
      (3,1)+(1,4)      (3,1,5)+(4,1)     (32,1)+(32,)
Q3  x 是 (2,3,4)。x.sum(dim=1) 的 shape？加 keepdim 呢？
    x.max(dim=1) 回傳幾個東西？
Q4  a[2:5] 和 a[[2,3,4]] 哪個是 view？
    怎麼用 gather 從 (N,C) logits 取出每筆的正確類別分數？
Q5  寫出 MHA 從 (B,L,D) 到 (B,L,D) 的完整 shape 流（7 個步驟）。
Q6  cat 和 stack 差在哪？expand 和 repeat 差在哪？
Q7  用 einsum 寫出 batch 矩陣乘法與 attention 分數。
Q8  x=[1,2,3] requires_grad，y=(x**2).sum()，x.grad 是多少？
    連續 backward 兩次會發生什麼？
    寫出 straight-through estimator 的一行 trick。
"""


def main():
    # ============================================================
    # argparse：讓這支程式能吃命令列參數（第一次出現，完整解釋一次；
    # 之後每支程式都是同一套模式，只是參數不同）
    # ============================================================
    # ① 建一個「參數解析器」
    ap = argparse.ArgumentParser()
    # ② 定義一個參數：
    #      "--quiz"          命令列上的名字，用兩個減號開頭代表「選填」
    #      action="store_true"  這是「開關」型參數：命令列有寫 --quiz 就是 True，
    #                           沒寫就是預設值 False（不用再打 --quiz=true）
    #      help="..."        執行 `python 01_tensor_playground.py -h` 時顯示的說明
    ap.add_argument("--quiz", action="store_true", help="只印題目，不給答案")
    # ③ 讀取你在終端機打的參數（Python 內部存在 sys.argv 裡），
    #    回傳一個物件，之後用 args.quiz 存取剛剛定義的那個參數
    args = ap.parse_args()
    # 範例：
    #   python 01_tensor_playground.py          -> args.quiz == False
    #   python 01_tensor_playground.py --quiz   -> args.quiz == True

    if args.quiz:
        print(QUIZ_TEXT)
        return

    torch.manual_seed(0)
    for q in QUESTIONS:
        q()
    print("\n" + "=" * 68)
    print("全部跑完。任何一題答錯，回去讀 01_syntax/05 與 06 的對應章節。")
    print("=" * 68)


# ★ 只有「直接執行這個檔案」才會呼叫 main()。
#   之後每支程式最後都是這兩行，意思相同，就不再重複解釋。
if __name__ == "__main__":
    main()
