"""NumPy / pandas 特徵工程實戰 —— 直接用你自己的 2330.csv。

用法：
    python 02_numpy_pandas_drill.py
    python 02_numpy_pandas_drill.py --csv ../../2344.csv

會示範：
  1. EDA 起手式
  2. 向量化 vs for 迴圈的速度差距
  3. rolling / shift 做技術指標（★ 防 data leakage）
  4. 按時間切分 + 只用訓練集算正規化參數
  5. sliding_window_view 切出 (N, W, F) 的訓練樣本
  6. pandas -> numpy -> torch -> DataLoader 的完整鏈路

===========================================================================
本檔用到的語法（看不懂查 01_syntax/00_看不懂時先讀這裡.md）
---------------------------------------------------------------------------
  df.assign(x=lambda d: ...)   加新欄位且可串接，lambda 的 d 就是「當下的 df」
  df[cols] = df[cols].shift(1) 一次對多欄做位移（cols 是欄名的 list）
  df.loc[a:b, cols]            用「標籤」選取，切片含右端點  -> 比較大全 §18
  Path(__file__).parents[2]    往上兩層資料夾               -> 解碼器 §15
  lambda / 推導式 / f-string   -> 解碼器 §2 §3 §1
  np.float32 / np.int64        指定 dtype（模型要 float32，標籤要 int64）
===========================================================================
"""

import argparse
import sys
import time

# Windows 主控台（cp950）遇到不支援的字元時不要直接崩潰
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

# __file__ = 這個 .py 檔的路徑；.resolve() 轉成絕對路徑
# .parents[2] = 往上跳兩層資料夾：03_code -> pytorch_bootcamp -> learning
ROOT = Path(__file__).resolve().parents[2]      # learning/
DEFAULT_CSV = ROOT / "2330.csv"


def banner(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ============================================================
def step0_clean(df):
    """★ 真實資料一定有髒值。你的 2330.csv 的 label 欄就混了 Excel 的 #REF!。

    這正是為什麼 df.dtypes 一定要看：label 若被讀成 str/object 而不是 int，
    就代表裡面混了非數字。
    """
    banner("0. 清理髒資料 --- 真實世界的第一課")
    print(f"  清理前 label dtype = {df['label'].dtype}")
    # pd.to_numeric(..., errors="coerce")：轉得成數字就轉，轉不成的變 NaN（不報錯）
    # .isna() 找出變成 NaN 的；& 是「且」（pandas 要用 & 不能用 and）-> 比較大全附錄
    # 加上 .notna() 排除「本來就是空值」的，只留下「有值但不是數字」的髒資料
    bad = pd.to_numeric(df["label"], errors="coerce").isna() & df["label"].notna()
    if bad.any():
        print(f"  ★ 發現 {bad.sum()} 筆無法轉成數字的 label："
              f"{df.loc[bad, 'label'].unique()[:5].tolist()}")
        print(f"    出現在 {df.loc[bad, 'Date'].min().date()} ~ "
              f"{df.loc[bad, 'Date'].max().date()}")
    # errors="coerce" 把轉不了的變成 NaN，再一次 dropna
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    for c in df.columns:
        if c not in ("Date", "label"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(np.int64)
    print(f"  清理後 label dtype = {df['label'].dtype}，"
          f"列數 {before} -> {len(df)}（丟掉 {before - len(df)} 列）")
    return df


# ============================================================
def step1_eda(df):
    banner("1. EDA 起手式 --- 拿到任何新資料都先跑這段")
    print(f"shape = {df.shape}")
    print(f"\ndtypes:\n{df.dtypes.to_string()}")
    na = df.isna().sum()
    print(f"\n缺失值:\n{na[na > 0].to_string() if (na > 0).any() else '  無'}")
    print(f"\n重複列: {df.duplicated().sum()}")
    print(f"\n數值統計:\n{df.describe().T[['mean', 'std', 'min', 'max']].to_string()}")
    if "label" in df.columns:
        vc = df['label'].value_counts(normalize=True).sort_index()
        print(f"\nlabel 分布（★ 看是否不平衡）:\n{vc.to_string()}")
        print(f"  最大類別佔比 = {vc.max():.2%}  "
              f"<-- 全部猜這一類就有這個 accuracy，你的模型必須贏過它")


# ============================================================
def step2_vectorization(df):
    banner("2. 向量化 vs for 迴圈 --- 感受一下差距")
    x = df["K"].to_numpy(np.float64)
    n = len(x)

    # 寫法 A：Python for 迴圈（慢）
    t0 = time.perf_counter()          # perf_counter 是高精度計時器
    out_loop = np.empty(n)            # 先配置好空間（比一直 append 快）
    for i in range(n):
        out_loop[i] = x[i] ** 2 + 3 * x[i]
    t_loop = time.perf_counter() - t0

    # 寫法 B：向量化（快）—— 對「整個陣列」做運算，迴圈在 C 層跑
    t0 = time.perf_counter()
    out_vec = x ** 2 + 3 * x          # ★ 沒有 for，NumPy 自動對每個元素做
    t_vec = time.perf_counter() - t0

    print(f"  for 迴圈 : {t_loop * 1000:8.2f} ms")
    print(f"  向量化   : {t_vec * 1000:8.2f} ms")
    print(f"  加速倍數 : {t_loop / max(t_vec, 1e-9):8.1f}x")
    assert np.allclose(out_loop, out_vec)

    # 移動平均：cumsum 技巧
    def ma_cumsum(a, w):
        c = np.cumsum(np.insert(a, 0, 0.0))
        return (c[w:] - c[:-w]) / w
    ma_pd = df["K"].rolling(20).mean().to_numpy()[19:]
    assert np.allclose(ma_cumsum(x, 20), ma_pd, atol=1e-8)
    print("  ★ cumsum 版移動平均與 pandas rolling 結果一致")


# ============================================================
def step3_features(df):
    banner("3. 特徵工程 --- rolling / shift / ewm")
    d = df.copy()
    # 用 K 當作「價格代理」示範（你的 csv 沒有原始 Close 欄）
    px = d["K"]

    # df.assign(新欄名=值)：回傳「加了新欄的複本」，不改原 df（符合 pandas 3.0 精神）
    # 值寫成 lambda t: ... 時，t 就是「當下這個 df」，所以可以引用剛剛才建好的欄位
    d = d.assign(
        log_vol=lambda t: np.log1p(t["Volume"]),     # log1p(x)=log(1+x)，避免 log(0)
        kd_diff=lambda t: t["K"] - t["D"],
        k_ma5=lambda t: px.rolling(5).mean(),
        k_ma20=lambda t: px.rolling(20).mean(),
        k_std20=lambda t: px.rolling(20).std(),
        k_ret1=lambda t: px.pct_change(),
        k_ema12=lambda t: px.ewm(span=12, adjust=False).mean(),
    )
    d["ma_gap"] = (d["k_ma5"] - d["k_ma20"]) / (d["k_ma20"].abs() + 1e-8)
    d["rsi14"] = rsi(px, 14)
    dif, dea, hist = macd(px)
    d["macd_dif"], d["macd_dea"], d["macd_hist"] = dif, dea, hist

    new_cols = ["log_vol", "kd_diff", "k_ma5", "k_ma20", "k_std20", "k_ret1",
                "k_ema12", "ma_gap", "rsi14", "macd_dif", "macd_dea", "macd_hist"]
    print(f"  新增 {len(new_cols)} 個特徵：{new_cols}")
    print(f"\n{d[new_cols].tail(3).T.to_string()}")
    return d, new_cols


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-12))


def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, dif - dea


# ============================================================
def step4_leakage_and_split(d, feat_cols, window=60,
                            train_ratio=0.70, val_ratio=0.15):
    banner("4. 防 data leakage + 按時間切分 + 正規化 ★ 這節最重要")

    # (a) 所有特徵 shift(1)：只用「昨天以前」的資訊
    d = d.copy()
    # shift(1)：整欄往下移一格，第 t 列拿到的是第 t-1 天的值 = 「昨天的資訊」
    # ★ 這一行就是防 data leakage 的關鍵：確保特徵只用得到過去的資料
    d[feat_cols] = d[feat_cols].shift(1)
    d = d.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)
    print(f"  (a) 特徵全部 shift(1) 後剩 {len(d)} 列")

    # (b) 按時間切（★ 絕對不能用 random_split）
    n = len(d)
    i_tr, i_va = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    print(f"  (b) 切分  train=[0,{i_tr})  val=[{i_tr},{i_va})  test=[{i_va},{n})")
    print(f"      train 最後日期 = {d.loc[i_tr - 1, 'Date']}")
    print(f"      val   第一日期 = {d.loc[i_tr, 'Date']}   <-- 必須晚於上一行")
    assert pd.Timestamp(d.loc[i_tr - 1, "Date"]) < pd.Timestamp(d.loc[i_tr, "Date"])

    # (c) ★ 正規化參數只能用訓練集算
    # ★ 只用「訓練區間」算平均與標準差。用到驗證/測試資料就是洩漏。
    # .loc[: i_tr-1, cols]：loc 的切片「含右端點」，所以是第 0 ~ i_tr-1 列
    # .replace(0, 1.0)：標準差為 0（整欄都一樣）時除法會爆炸，換成 1
    mu = d.loc[: i_tr - 1, feat_cols].mean()
    sd = d.loc[: i_tr - 1, feat_cols].std().replace(0, 1.0)
    d[feat_cols] = (d[feat_cols] - mu) / sd
    print(f"  (c) 正規化參數只用 train 算："
          f"train 區間 mean~={d.loc[:i_tr-1, feat_cols].to_numpy().mean():+.4f}, "
          f"test 區間 mean~={d.loc[i_va:, feat_cols].to_numpy().mean():+.4f}")
    print("      ★ test 的 mean 不是 0 是正常的 —— 那才代表沒有洩漏")

    return d, i_tr, i_va


# ============================================================
def step5_windows(d, feat_cols, i_tr, i_va, window=60):
    banner("5. sliding_window_view 切出 (N, W, F) 訓練樣本")
    # ★ 兩個 dtype 都不能寫錯：
    #   特徵必須 float32（模型權重是 float32，float64 會報 expected Float but found Double）
    #   分類標籤必須 int64（CrossEntropyLoss 硬性要求）
    X = d[feat_cols].to_numpy(np.float32)            # (N, F)
    y = d["label"].to_numpy(np.int64)                # (N,)
    print(f"  原始      X={X.shape}  y={y.shape}")

    # sliding_window_view：把 (N,F) 切成一堆長度 W 的視窗，而且「零複製」
    #   （它只是換一組 stride 去讀同一塊記憶體，所以超快也不佔額外記憶體）
    Xw = sliding_window_view(X, window, axis=0)      # (N-W+1, F, W)
    # transpose 把維度順序換成 (樣本, 時間, 特徵)，這是模型習慣吃的順序
    # ★ transpose 後記憶體不連續，ascontiguousarray 重排成連續（不然 torch 會抱怨）
    Xw = np.ascontiguousarray(Xw.transpose(0, 2, 1))  # -> (N-W+1, W, F)
    yw = y[window - 1:]
    print(f"  切視窗後  Xw={Xw.shape}  yw={yw.shape}")

    # ★ 視窗會跨越切分點，所以驗證/測試的起點要往後推 window-1，避免重疊洩漏
    splits = {
        "train": (0, i_tr - window + 1),
        "val":   (i_tr, i_va - window + 1),
        "test":  (i_va, len(Xw)),
    }
    out = {}
    for name, (lo, hi) in splits.items():
        lo, hi = max(lo, 0), max(hi, 0)
        out[name] = (torch.from_numpy(Xw[lo:hi].copy()),
                     torch.from_numpy(yw[lo:hi].copy()))
        print(f"  {name:5s}: X={tuple(out[name][0].shape)} y={tuple(out[name][1].shape)}")
    return out


# ============================================================
def step6_to_torch(splits):
    banner("6. pandas -> numpy -> torch -> DataLoader")
    loaders = {}
    for name, (X, y) in splits.items():
        ds = TensorDataset(X, y)
        loaders[name] = DataLoader(ds, batch_size=64,
                                   shuffle=(name == "train"), drop_last=(name == "train"))
    # iter() 取迭代器、next() 拿第一個 batch（只拿一批，不會跑完整個 epoch）
    xb, yb = next(iter(loaders["train"]))
    print(f"  一個 batch: x={tuple(xb.shape)} {xb.dtype}   y={tuple(yb.shape)} {yb.dtype}")
    print(f"  x 範圍 [{xb.min():.3f}, {xb.max():.3f}]  mean={xb.mean():.4f}")
    print(f"  y 類別 {sorted(yb.unique().tolist())}")
    assert xb.dtype == torch.float32, "特徵必須是 float32"
    assert yb.dtype == torch.int64, "分類標籤必須是 int64"
    assert not torch.isnan(xb).any(), "輸入含 NaN"
    print("  ★ 全部檢查通過，這份資料可以直接餵給模型")
    return loaders


# ============================================================
EXERCISES = """
動手練習（做完再往下讀 02_topics/）
  1. 把 rsi / macd 之外再加 3 個技術指標（布林通道、KD 背離、成交量比）。
  2. 把 2330 / 2344 / 8028 用 pd.concat(axis=1, join="inner") 對齊日期，
     做出「跨股票特徵」（例如同日大盤三檔的平均 K 值）。
  3. 用 groupby(每月).transform("mean") 做出「當月平均 K」當特徵。
  4. 找出你原本 utils.py 裡所有 data leakage 的地方，列成清單並修好。
  5. 不用任何 for 迴圈，算出 2330 每一天跟「過去 250 天」的最大回撤。
"""


def main():
    # argparse 用法見 01_tensor_playground.py 開頭的完整解釋。
    # 這裡定義兩個參數，都不是開關型（沒有 action="store_true"），
    # 所以要用 `--csv 路徑` `--window 數字` 這樣「名字 空格 值」的方式傳。
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))     # 字串，預設用你的 2330.csv
    ap.add_argument("--window", type=int, default=60)       # type=int 一定要寫，
    args = ap.parse_args()                                  # 否則吃到的是字串 "60" 不是數字 60

    path = Path(args.csv)
    if not path.exists():
        print(f"找不到 {path}，請用 --csv 指定路徑")
        return

    pd.set_option("display.width", 200)
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"讀入 {path.name}: {len(df)} 列, {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    df = step0_clean(df)
    step1_eda(df)
    step2_vectorization(df)
    d, new_cols = step3_features(df)

    base_cols = [c for c in df.columns if c not in ("Date", "label")]
    feat_cols = base_cols + new_cols

    d, i_tr, i_va = step4_leakage_and_split(d, feat_cols)
    splits = step5_windows(d, feat_cols, i_tr, i_va, window=args.window)
    step6_to_torch(splits)

    print(EXERCISES)


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()