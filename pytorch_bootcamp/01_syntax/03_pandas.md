# 03 · pandas 完整教學

> **定位**：pandas 負責「把原始資料整理成模型吃得下的乾淨數字陣列」。
> 你的 `utils.py` 已經在做這件事，這章把它系統化。

## 【新版注意】你的環境是 pandas 3.0.3，跟網路教學差異很大

我實際在你的環境測試過，以下是**確定的行為**：

| 舊教學會寫 | pandas 3.0 的真實行為 | 正確寫法 |
|---|---|---|
| `df.append(row)` | ❌ **已移除**，`AttributeError` | `pd.concat([df, new_df])` |
| `df.applymap(f)` | ❌ **已移除** | `df.map(f)` |
| `df['a'][0] = 9` | ⚠️ **靜默失效**，拋 `ChainedAssignmentError` 警告但 df 不變 | `df.loc[0, 'a'] = 9` |
| `pd.set_option('mode.copy_on_write', False)` | ❌ 已無法關閉，CoW 永遠開啟 | 不需要設定 |
| 字串欄位 dtype 是 `object` | 現在是 `str`（原生字串型別，更快更省記憶體） | 照常用 |
| `df.iteritems()` | ❌ 已移除 | `df.items()` |
| `inplace=True` | 仍可用但**強烈不建議**，未來會廢棄 | `df = df.method(...)` |

### Copy-on-Write 是什麼，為什麼你一定要懂

```python
df = pd.DataFrame({'a': [1, 2, 3]})

df['a'][0] = 99          # ❌ 鏈式賦值：df 完全沒變！只跳警告
print(df['a'].tolist())  # [1, 2, 3]

df.loc[0, 'a'] = 99      # ✅ 一步到位的賦值才有效
print(df['a'].tolist())  # [99, 2, 3]
```

> 🔥 這是你在舊教學上學到、但**在你的環境會靜默壞掉**的頭號陷阱。
> **原則：所有寫入都用 `.loc` / `.iloc` 一次完成，不要分兩次索引。**

---

## §1 建立與讀取

```python
import pandas as pd
import numpy as np

# 從 dict（最常見）
df = pd.DataFrame({
    "date":  pd.date_range("2024-01-01", periods=5),
    "price": [100, 102, 101, 105, 103],
    "vol":   [1000, 1500, 900, 2000, 1200],
})

# 從 NumPy
df = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))

# 讀 CSV（★ 你最常用的）
df = pd.read_csv("2330.csv")
df = pd.read_csv(
    "2330.csv",
    parse_dates=["Date"],          # ★ 直接把 Date 轉成 datetime
    index_col="Date",              # 用日期當索引
    usecols=["Date", "K", "D"],    # 只讀需要的欄，省記憶體
    dtype={"label": "int8"},       # 指定 dtype，省記憶體
    nrows=1000,                    # 只讀前 1000 列（探索時很好用）
)

df.to_csv("out.csv", index=False)
df.to_parquet("out.parquet")       # ★ 大檔案用 parquet，讀寫快 10 倍且保留 dtype
```

---

## §2 第一件事：看資料（EDA 起手式）

```python
df.head(10)          # 前 10 列
df.tail()            # 後 5 列
df.sample(5)         # 隨機 5 列 ★ 比 head 更能發現問題
df.shape             # (列數, 欄數)
df.columns           # 欄位名
df.dtypes            # 每欄型別 ★ 一定要看，字串混進數值欄是常見災難
df.info()            # 型別 + 非空數量 + 記憶體
df.describe()        # 數值欄的統計摘要 ★
df.describe(include="all")   # 含非數值欄

df.isna().sum()                    # ★ 每欄有幾個缺失值
df.isna().mean().sort_values(ascending=False)   # 缺失比例排序
df.duplicated().sum()              # 重複列數
df["label"].value_counts()         # ★ 看類別分布是否不平衡
df["label"].value_counts(normalize=True)        # 比例
df.nunique()                       # 每欄有幾種不同值
```

**你拿到任何新資料的固定流程：**

```python
def quick_eda(df):
    print(f"shape: {df.shape}")
    print(f"\ndtypes:\n{df.dtypes}")
    print(f"\n缺失值:\n{df.isna().sum()[lambda s: s > 0]}")
    print(f"\n重複列: {df.duplicated().sum()}")
    print(f"\n數值統計:\n{df.describe().T}")
    for c in df.select_dtypes(include=['str', 'object', 'category']).columns:
        print(f"\n{c} 前 5 種值:\n{df[c].value_counts().head()}")
```

---

## §3 選取資料：`[]` / `loc` / `iloc` ★

```python
# 選欄
df["price"]              # Series（一維）
df[["price", "vol"]]     # DataFrame（二維）★ 注意雙層括號

# 選列（用 [] 時是列切片，容易混淆，不建議）
df[0:3]

# ★ 正解：一律用 loc / iloc
df.loc[行標籤, 欄標籤]      # 用「標籤」
df.iloc[行位置, 欄位置]     # 用「整數位置」
```

```python
df = pd.read_csv("2330.csv", parse_dates=["Date"], index_col="Date")

df.loc["2020-01-02"]                       # 該日期整列
df.loc["2020-01-01":"2020-01-31"]          # ★ loc 切片「含右端點」
df.loc["2020", ["K", "D"]]                 # 2020 全年的 K、D 欄
df.loc[df["K"] > 80, ["K", "D", "label"]]  # ★ 條件 + 選欄（最常用的組合）

df.iloc[0]                                 # 第 0 列
df.iloc[0:5]                               # ★ iloc 切片「不含右端點」（跟 Python 一致）
df.iloc[-100:]                             # 最後 100 列
df.iloc[:, 0:3]                            # 前 3 欄
df.iloc[[0, 5, 10], [1, 2]]

df.at["2020-01-02", "K"]                   # 單一值（比 loc 快）
df.iat[0, 1]
```

> ★ **記法**：`loc` 用標籤所以含右端點（因為標籤沒有「下一個」的概念）；
> `iloc` 用位置所以不含右端點（跟 Python 切片一致）。

### 條件篩選

```python
df[df["K"] > 80]
df[(df["K"] > 80) & (df["D"] > 80)]        # ★ 要用 & | ~，且每個條件加括號
df[df["label"].isin([0, 1])]
df[df["K"].between(20, 80)]
df[~df["K"].isna()]                        # 非缺失
df.query("K > 80 and D > 80")              # 另一種寫法，可讀性好
df.query("K > @threshold")                 # @ 引用外部變數
```

---

## §4 清理資料

```python
# 缺失值
df.dropna()                        # 丟掉有任何 NaN 的列
df.dropna(subset=["label"])        # ★ 只看特定欄
df.dropna(axis=1, thresh=100)      # 丟掉非空值少於 100 的欄

df.fillna(0)
df.fillna(df.mean(numeric_only=True))       # 用平均補
df["K"] = df["K"].ffill()                   # 用前一個值補（★ 時序資料常用）
df["K"] = df["K"].bfill()                   # 用後一個值補（⚠️ 時序上會漏未來資訊！）
df["K"] = df["K"].interpolate(method="linear")

# 重複
df.drop_duplicates()
df.drop_duplicates(subset=["Date"], keep="last")

# 型別
df["label"] = df["label"].astype("int8")
df["Date"] = pd.to_datetime(df["Date"])
df["x"] = pd.to_numeric(df["x"], errors="coerce")   # ★ 轉不了的變 NaN，不報錯

# 重新命名 / 刪除欄
df = df.rename(columns={"MA5Close_BR": "ma5"})
df = df.drop(columns=["tmp1", "tmp2"])

# 排序
df = df.sort_values("Date")
df = df.sort_values(["label", "K"], ascending=[True, False])
df = df.sort_index()
df = df.reset_index(drop=True)      # ★ 篩選/排序後一定要 reset，否則索引會跳號
```

### 異常值處理

```python
# 用分位數截斷（winsorize）★ 金融資料常用
lo, hi = df["vol"].quantile([0.01, 0.99])
df["vol"] = df["vol"].clip(lo, hi)

# 用 IQR 找異常
q1, q3 = df["vol"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = (df["vol"] < q1 - 1.5*iqr) | (df["vol"] > q3 + 1.5*iqr)
print(f"異常值 {outliers.sum()} 筆")
```

---

## §5 新增欄位與轉換

```python
# 直接運算（向量化，快）★ 優先用這個
df["range"] = df["High"] - df["Low"]
df["log_vol"] = np.log1p(df["Volume"])          # log(1+x)，避免 log(0)

# assign：可以串接，不改原 df（★ 推薦，符合 CoW 精神）
df = df.assign(
    ret   = lambda d: d["Close"].pct_change(),
    ma5   = lambda d: d["Close"].rolling(5).mean(),
    above = lambda d: (d["Close"] > d["ma5"]).astype(int),   # 可以引用前面剛建的欄
)

# apply（慢，能向量化就不要用）
df["x"] = df["K"].apply(lambda v: v / 100)          # ❌ 應該寫 df["K"] / 100
df["y"] = df.apply(lambda r: r["K"] - r["D"], axis=1)   # ❌ 應該寫 df["K"] - df["D"]

# map：Series 逐元素對映（適合類別編碼）
df["label_name"] = df["label"].map({0: "跌", 1: "漲"})

# 分箱（把連續值切成類別）★ 你的 8 類漲跌幅就是這樣做的
df["bin"] = pd.cut(df["ret"], bins=[-1, -0.03, -0.01, 0, 0.01, 0.03, 1],
                   labels=[0, 1, 2, 3, 4, 5])
df["qbin"] = pd.qcut(df["ret"], q=8, labels=False)   # ★ 等頻分箱，每箱樣本數相同

# one-hot
pd.get_dummies(df["label_name"], prefix="lbl")
pd.get_dummies(df, columns=["sector"], drop_first=True)

# 條件賦值
df["signal"] = np.where(df["K"] > 80, -1, np.where(df["K"] < 20, 1, 0))
df["grade"] = np.select(
    [df["ret"] > 0.03, df["ret"] > 0, df["ret"] > -0.03],
    ["大漲", "小漲", "小跌"],
    default="大跌",
)
```

---

## §6 groupby ★（做特徵工程的主力）

```python
# 基本：分組 → 聚合
df.groupby("label")["K"].mean()
df.groupby("label").agg({"K": "mean", "D": ["mean", "std"], "Volume": "sum"})
df.groupby("label").size()                    # 每組筆數
df.groupby(["year", "label"])["ret"].mean()   # 多層分組

# 自訂聚合（推薦這個寫法，欄名乾淨）
df.groupby("label").agg(
    k_mean = ("K", "mean"),
    k_std  = ("K", "std"),
    n      = ("K", "size"),
)

# ★ transform：結果 shape 跟原 df 一樣，可以直接當新欄位
df["k_month_mean"] = df.groupby(df.index.to_period("M"))["K"].transform("mean")
df["k_zscore"] = df.groupby("sector")["K"].transform(lambda s: (s - s.mean()) / s.std())
# ↑ 「每個產業內部做標準化」——這是很常見的特徵工程

# filter：保留符合條件的整組
df.groupby("stock").filter(lambda g: len(g) >= 100)   # 只留資料超過 100 天的股票

# apply：最彈性但最慢
df.groupby("stock").apply(lambda g: g.nlargest(3, "ret"), include_groups=False)
```

---

## §7 合併資料

```python
# concat：上下疊 or 左右接
pd.concat([df1, df2], axis=0, ignore_index=True)   # ★ 疊列（取代已移除的 append）
pd.concat([df1, df2], axis=1)                      # 接欄

# merge：像 SQL 的 JOIN ★
pd.merge(left, right, on="Date", how="inner")      # inner/left/right/outer
pd.merge(left, right, left_on="date", right_on="dt", how="left")
pd.merge(left, right, on=["Date", "stock"], suffixes=("_a", "_b"))

# join：用索引合併（比較簡潔）
df1.join(df2, how="left", rsuffix="_r")

# ★ 實戰：把 2330 / 2344 / 8028 三檔對齊日期
dfs = {}
for code in ["2330", "2344", "8028"]:
    d = pd.read_csv(f"{code}.csv", parse_dates=["Date"], index_col="Date")
    dfs[code] = d[["K", "D"]].add_prefix(f"{code}_")
merged = pd.concat(dfs.values(), axis=1, join="inner")   # 只留三檔都有的日期
```

---

## §8 時間序列 ★★（你的資料就是時序，這節最重要）

```python
df = pd.read_csv("2330.csv", parse_dates=["Date"], index_col="Date").sort_index()

# datetime 屬性（用 .dt，索引則直接用）
df.index.year, df.index.month, df.index.dayofweek, df.index.quarter
df["month"] = df.index.month
df["dow"]   = df.index.dayofweek            # 0=週一
df["is_month_end"] = df.index.is_month_end

# 時間切片（★ 超方便）
df.loc["2020"]                # 2020 全年
df.loc["2020-03"]             # 2020 年 3 月
df.loc["2020-01":"2020-06"]

# 重採樣（改變時間頻率）
df["Close"].resample("W").last()     # 週線
df["Close"].resample("ME").mean()    # 月均（ME = month end）
df.resample("W").agg({"Close": "last", "Volume": "sum"})
```

### 8.1 shift / rolling / diff ★★★（防 data leakage 的關鍵）

```python
close = df["Close"]

close.shift(1)             # ★ 往下移一格 = 「昨天的值」
close.shift(-1)            # ★ 往上移 = 「明天的值」→ 只能當 label，絕不能當特徵！
close.diff()               # 一階差分 = close - close.shift(1)
close.pct_change()         # 報酬率
close.pct_change(5)        # 5 日報酬率

close.rolling(20).mean()               # 20 日移動平均
close.rolling(20).std()                # 20 日波動率
close.rolling(20, min_periods=5).mean()  # 前面不足 20 筆也算（用至少 5 筆）
close.rolling(20).apply(lambda w: w.max() - w.min())   # 自訂視窗函式
close.ewm(span=12).mean()              # 指數加權移動平均（EMA）★ MACD 用這個
close.expanding().mean()               # 從頭到當前的累積平均
```

> 🔥 **Data leakage 是時序模型最致命的錯誤。**
>
> ```python
> # ❌ 用到未來資訊：rolling(20) 的第 t 筆包含了第 t 天當天的收盤價
> df["ma20"] = df["Close"].rolling(20).mean()
> df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
> # 上面 ma20 包含今天收盤，但你要預測「明天」→ 嚴格說今天收盤是可以用的（收盤後才決策）
>
> # ✅ 更保險的做法：所有特徵都 shift(1)，確保只用「昨天以前」的資訊
> feat_cols = ["ma20", "K", "D", "rsi"]
> df[feat_cols] = df[feat_cols].shift(1)
> df = df.dropna()
> ```
>
> **檢查方法**：如果你的模型準確率高得離譜（>85% 預測漲跌），
> 99% 是 leakage，不是你的模型天才。

### 8.2 時序資料切分 ★

```python
# ❌ 絕對不能對時序資料用 train_test_split(shuffle=True)
from sklearn.model_selection import train_test_split
X_tr, X_te = train_test_split(X, test_size=0.2)     # ❌ 打亂 = 用未來預測過去

# ✅ 按時間切
n = len(df)
train = df.iloc[: int(n * 0.7)]
val   = df.iloc[int(n * 0.7) : int(n * 0.85)]
test  = df.iloc[int(n * 0.85) :]

# ✅ 更嚴謹：加 gap 避免視窗重疊造成的洩漏
W = 60          # 你的 WINDOW_SIZE
val   = df.iloc[int(n*0.7) + W : int(n*0.85)]
test  = df.iloc[int(n*0.85) + W :]

# ✅ 標準化參數只能用訓練集算！
mu, sd = train[feat_cols].mean(), train[feat_cols].std()
for part in (train, val, test):
    part[feat_cols] = (part[feat_cols] - mu) / (sd + 1e-8)   # 用同一組 mu/sd
```

> 🔥 **這是你 `utils.py` 目前的問題**：
> `df['Volume'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()`
> 用了**全部資料**（含測試集）的平均和標準差 → 測試集資訊洩漏到訓練。
> 正確做法是只用訓練集的統計量。這在論文審查會被直接抓出來。

---

## §9 pandas → NumPy → PyTorch 的完整橋接 ★

```python
# 1. pandas → NumPy
X = df[feat_cols].to_numpy(dtype=np.float32)     # ★ 用 to_numpy，不要用已過時的 .values
y = df["label"].to_numpy(dtype=np.int64)         # ★ 分類標籤必須 int64

# 2. NumPy → PyTorch
import torch
X_t = torch.from_numpy(X)                        # float32
y_t = torch.from_numpy(y)                        # int64 (long)

# 3. 包成 Dataset
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(X_t, y_t)
loader = DataLoader(ds, batch_size=64, shuffle=True)
```

**一個乾淨的完整前處理函式（可以直接改你的 `utils.py`）：**

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

FEATURES = ["K", "D", "MA5Close_BR", "MA10Close_BR", "MA5High_BR",
            "MA10High_BR", "MA5Low_BR", "MA10Low_BR"]

def load_stock(path, window=60, train_ratio=0.7, val_ratio=0.15):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")

    # --- 特徵工程 ---
    df["log_vol"] = np.log1p(df["Volume"])
    df["kd_diff"] = df["K"] - df["D"]
    feats = FEATURES + ["log_vol", "kd_diff"]

    # --- 只用過去資訊 ---
    df[feats] = df[feats].shift(1)
    df = df.dropna(subset=feats + ["label"]).reset_index(drop=True)

    # --- 按時間切分 ---
    n = len(df)
    i_tr, i_va = int(n * train_ratio), int(n * (train_ratio + val_ratio))

    # --- 標準化參數只從訓練集算 ★ ---
    mu = df.loc[: i_tr - 1, feats].mean()
    sd = df.loc[: i_tr - 1, feats].std().replace(0, 1.0)
    df[feats] = (df[feats] - mu) / sd

    X = df[feats].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int64)

    # --- 切滑動視窗 ---
    from numpy.lib.stride_tricks import sliding_window_view
    Xw = sliding_window_view(X, window, axis=0)     # (N-W+1, F, W)
    Xw = np.ascontiguousarray(Xw.transpose(0, 2, 1))  # → (N-W+1, W, F)
    yw = y[window - 1:]                              # 對齊視窗最後一天

    splits = {}
    for name, lo, hi in [("train", 0, i_tr - window + 1),
                         ("val",   i_tr, i_va - window + 1),
                         ("test",  i_va, len(Xw))]:
        splits[name] = (torch.from_numpy(Xw[lo:hi].copy()),
                        torch.from_numpy(yw[lo:hi].copy()))
    return splits, feats
```

---

## §10 效能與記憶體

```python
# 看記憶體
df.memory_usage(deep=True).sum() / 1e6      # MB

# 降型別（常常能省 50~75%）
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
for c in df.select_dtypes("int64").columns:
    df[c] = pd.to_numeric(df[c], downcast="integer")

# 低基數的字串欄 → category（省超多）
df["stock"] = df["stock"].astype("category")

# 大檔案分塊讀
for chunk in pd.read_csv("huge.csv", chunksize=100_000):
    process(chunk)
```

**速度排序（由快到慢）**：
向量化運算 `>` `df.map` / `Series.map` `>` `df.apply(axis=0)` `>` `df.apply(axis=1)` `>` `for i, row in df.iterrows()`

> ⚠️ **永遠不要用 `iterrows()` 做數值運算**，它比向量化慢 1000 倍。
> 真的需要逐列時用 `df.itertuples()`（快 10 倍）。

---

## §11 動手練習（用你的 `2330.csv`）

1. 讀入 `2330.csv`，印出完整 EDA（缺失、分布、label 比例）。
2. 用 `rolling` 做出：MA5/MA20 差值、20 日波動率、RSI(14)、MACD。
3. 用 `groupby` 算出「每個月的平均 K 值」，並把它 `transform` 回原 df 當特徵。
4. 把 2330、2344、8028 三檔按日期 inner join，做出跨股票特徵。
5. 找出你 `utils.py` 裡所有**可能造成 data leakage** 的地方，列成清單並修好。
6. 寫一個 `assert` 驗證：訓練集的最後一個日期 < 驗證集的第一個日期。

<details>
<summary>第 2 題參考（RSI 與 MACD）</summary>

```python
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, dif - dea      # DIF, DEA, 柱狀圖
```
</details>

---

## ✅ 自我檢核

- [ ] 解釋為何 `df['a'][0] = 1` 在 pandas 3.0 沒有作用，正確寫法是什麼
- [ ] 說出 `loc` 與 `iloc` 在切片端點上的差別，以及為何設計成這樣
- [ ] 說出 `groupby().agg()` 與 `groupby().transform()` 回傳的 shape 差在哪
- [ ] 解釋 `shift(1)` 為何是防 leakage 的關鍵，`shift(-1)` 為何只能當 label
- [ ] 說出「標準化參數只能用訓練集算」的理由
- [ ] 寫出 `df → np.float32 → torch tensor → DataLoader` 的完整鏈路
- [ ] 說出三種比 `iterrows()` 快的替代方案
