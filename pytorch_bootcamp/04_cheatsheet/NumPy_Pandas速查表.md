# NumPy / pandas 速查表

> 環境：NumPy 2.4.4、pandas 3.0.3。標 **【新版】** 的是跟舊教學不同的地方。

---

# NumPy

## 建立

```python
np.array([1,2,3], dtype=np.float32)
np.zeros((2,3))  np.ones((2,3))  np.full((2,3),7.)  np.empty((2,3))  np.eye(3)
np.arange(0,10,2)        np.linspace(0,1,5)      np.logspace(-4,-1,4)
np.zeros_like(a)  np.ones_like(a)

rng = np.random.default_rng(42)          # ★【新版】不要再用 np.random.seed
rng.random((2,3))   rng.normal(0,1,(2,3))   rng.integers(0,10,(2,3))
rng.permutation(10)  rng.choice(100, 10, replace=False)
```

## 【新版】已移除的舊寫法

| 舊 | 新 |
|---|---|
| `np.float_` | `np.float64` |
| `np.NaN` / `np.Inf` | `np.nan` / `np.inf` |
| `np.object` `np.bool` `np.str` | `object` `bool` `str` |
| `np.alltrue` | `np.all` |
| `np.in1d` | `np.isin` |
| `arr.ptp()` | `np.ptp(arr)` |

## 屬性

```python
a.shape  a.ndim  a.size  a.dtype  a.nbytes  a.T
a.astype(np.float32)
```

## 索引

```python
a[0,1,2]        a[:,1,:]      # 該維消失
a[:,1:2,:]      # 保留該維     a[...,0]      a[...,None]
a[2:5]          # ★ view      a[[2,3,4]]    # ★ copy
a[a>0]          a[(a>0)&(a<5)]     # ★ 用 & | ~ 且要加括號
np.where(a>0, a, 0)
a.argmax()  np.argsort(a)  a[np.arange(N), labels]   # ★ 取每列指定索引
```

## 形狀

```python
a.reshape(3,-1)   a.ravel()   a.flatten()
a.transpose(1,0,2)   a.swapaxes(0,1)   np.moveaxis(a,0,-1)
a[None,:]   a[:,None]   np.expand_dims(a,0)   a.squeeze(axis=0)
np.concatenate([a,b], axis=0)     # 不增維
np.stack([a,b], axis=0)           # ★ 增一維
np.split(a,3)   np.array_split(a,3)
```

## Broadcasting

從右往左對齊，每維要相等或其中一個是 1。

```
(3,1) + (1,4)          -> (3,4)
(256,256,3) + (3,)     -> (256,256,3)
(32,1) + (32,)         -> (32,32)   ★ 靜默 bug！
```

```python
X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
d = np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(-1))     # 距離矩陣
onehot = (labels[:,None] == np.arange(C)[None,:]).astype(np.float32)
```

## 軸與歸約

**`axis=k` = 消掉第 k 維。**

```python
a.sum(axis=0)   a.sum(axis=(0,1))   a.sum(axis=0, keepdims=True)
a.mean() a.std() a.min() a.max() a.argmax() a.cumsum()
np.nanmean(a)  np.nansum(a)         # 忽略 NaN
np.median(a)   np.percentile(a,[25,50,75])
```

## 數學

```python
a @ b                np.linalg.norm(a)      np.linalg.solve(A,b)
np.linalg.inv(M)     np.linalg.svd(M)       np.linalg.eig(M)
np.maximum(a,0)      np.clip(a,0,1)         np.log1p(a)   np.expm1(a)
np.einsum("ij,jk->ik", A, B)
```

## 向量化技巧

```python
from numpy.lib.stride_tricks import sliding_window_view
sliding_window_view(x, 60)                  # ★ 零複製切視窗

np.bincount(labels, minlength=C)            # 每類計數
np.bincount(labels, weights=vals) / counts  # 每類平均
np.bincount(y_true*C + y_pred, minlength=C*C).reshape(C,C)   # 混淆矩陣

def moving_average(x, w):                   # cumsum 版移動平均
    c = np.cumsum(np.insert(x, 0, 0)); return (c[w:] - c[:-w]) / w
```

---

# pandas

## 【新版】pandas 3.0 的重大差異 ★

| 舊教學 | pandas 3.0 |
|---|---|
| `df.append(row)` | ❌ 已移除 → `pd.concat([df, new])` |
| `df.applymap(f)` | ❌ 已移除 → `df.map(f)` |
| `df['a'][0] = 9` | ⚠️ **靜默失效** → `df.loc[0,'a'] = 9` |
| `df.iteritems()` | ❌ 已移除 → `df.items()` |
| 字串欄 dtype 是 `object` | 現在是 `str` |
| `inplace=True` | 仍可用但不建議 |

**Copy-on-Write 永遠開啟**：所有寫入都要用 `.loc` / `.iloc` 一步完成。

## 讀寫

```python
pd.read_csv(p, parse_dates=["Date"], index_col="Date",
            usecols=[...], dtype={"label":"int8"}, nrows=1000)
df.to_csv(p, index=False)      df.to_parquet(p)      # ★ 大檔用 parquet
for chunk in pd.read_csv(p, chunksize=100_000): ...
```

## EDA 起手式

```python
df.head()  df.sample(5)  df.shape  df.dtypes  df.info()  df.describe()
df.isna().sum()          df.duplicated().sum()
df["label"].value_counts(normalize=True)
df.nunique()             df.memory_usage(deep=True).sum()/1e6
```

## 選取

```python
df["a"]            df[["a","b"]]
df.loc[row_label, col_label]     # ★ 切片含右端點
df.iloc[row_pos, col_pos]        # ★ 切片不含右端點
df.loc[df["K"]>80, ["K","D"]]
df.query("K > 80 and D > 80")    df.query("K > @thr")
df.at[r,c]   df.iat[i,j]
```

## 清理

```python
df.dropna(subset=["label"])      df.fillna(0)     df["x"].ffill()
df["x"] = pd.to_numeric(df["x"], errors="coerce")     # ★ 轉不了的變 NaN
df.drop_duplicates(subset=["Date"], keep="last")
df = df.rename(columns={...}).drop(columns=[...])
df = df.sort_values("Date").reset_index(drop=True)
lo,hi = df["v"].quantile([0.01,0.99]); df["v"] = df["v"].clip(lo,hi)
```

## 新增欄位

```python
df = df.assign(ret=lambda d: d["Close"].pct_change(),
               ma5=lambda d: d["Close"].rolling(5).mean())
df["b"] = pd.cut(df["ret"], bins=[...], labels=[...])
df["q"] = pd.qcut(df["ret"], q=8, labels=False)       # ★ 等頻分箱
pd.get_dummies(df, columns=["sector"], drop_first=True)
np.where(cond, a, b)     np.select([c1,c2],[v1,v2], default=v3)
```

## groupby

```python
df.groupby("k")["v"].mean()
df.groupby("k").agg(v_mean=("v","mean"), n=("v","size"))
df.groupby("k")["v"].transform("mean")      # ★ shape 跟原 df 一樣
df.groupby("k").filter(lambda g: len(g)>=100)
```

## 合併

```python
pd.concat([a,b], axis=0, ignore_index=True)      # 疊列
pd.concat([a,b], axis=1, join="inner")           # 接欄（對齊索引）
pd.merge(l, r, on="Date", how="left", suffixes=("_a","_b"))
```

## 時間序列 ★

```python
df.index.year  df.index.month  df.index.dayofweek
df.loc["2020"]        df.loc["2020-01":"2020-06"]
df["v"].resample("W").last()      df.resample("ME").mean()

s.shift(1)          # ★ 昨天（特徵要用這個防 leakage）
s.shift(-1)         # ★ 明天（只能當 label）
s.diff()  s.pct_change()
s.rolling(20).mean()   s.rolling(20, min_periods=5).std()
s.ewm(span=12, adjust=False).mean()     s.expanding().mean()
```

## 防 data leakage 檢查清單

```
[ ] 特徵有沒有 shift(1)？
[ ] 有沒有用 random_split 切時序資料？（★ 絕對不行）
[ ] 正規化的 mean/std 是不是只用訓練集算的？
[ ] 驗證/測試起點有沒有避開視窗重疊（推 window-1）？
[ ] 準確率高得不合理嗎？（漲跌預測 > 85% 幾乎必然是 leakage）
```

## 效能

```python
df[c] = df[c].astype("float32")
df["cat"] = df["cat"].astype("category")
# 速度：向量化 > .map > .apply(axis=0) > .apply(axis=1) > itertuples >> iterrows
```

## 轉去 PyTorch

```python
X = df[feats].to_numpy(np.float32)      # ★ 用 to_numpy，不要用 .values
y = df["label"].to_numpy(np.int64)      # ★ 分類標籤必須 int64
X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
```
