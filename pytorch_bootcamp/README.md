# PyTorch 完整學習手冊（輔大資工 · 升大三專用）

> 這個資料夾的目標：讓你在**寫 PyTorch 時不再靠複製貼上**，而是知道每一行在做什麼、
> 為什麼要這樣寫、shape 是什麼、出錯時往哪裡查。最終能獨立讀論文 → 實作 → 做研究。

---

## 0. 你目前的狀態（我看過你的 `learning/` 資料夾）

| 你已經有的 | 檔案 | 評語 |
|---|---|---|
| CNN 圖片分類 | `CNN.py` | 標準 `nn.Sequential` 寫法，觀念正確 |
| CNN 應用在金融 | `CNN_finance.py` | 會把 CNN 遷移到非影像資料，很好 |
| CNN+Transformer 混合模型 | `Transformer.py` | 手刻 PositionalEncoding，程度不錯 |
| 自訂 Dataset | `utils.py` | 已懂 `Dataset` / `__getitem__` |
| 多目標演化演算法 | `NSGA-II.py` | 純 Python 實作 ZDT1，EC 有底子 |

**缺口診斷（這份教材就是來補這些的）：**

1. **語法是零散的**：會用 `view`，但不知道什麼時候該用 `reshape`/`permute`/`einsum`。
2. **不熟 autograd 機制**：不清楚 `detach()`、`no_grad()`、`retain_graph` 的差別 → 寫 GAN/RL 一定會卡。
3. **訓練迴圈每次重寫**：沒有一套自己的 template，導致每個專案品質不一。
4. **缺少驗證/實驗紀律**：沒有 seed 固定、沒有 early stopping、沒有實驗紀錄 → 做研究會很痛苦。
5. **主題只碰了 CNN/Transformer**：AE、SSL、GAN、壓縮、RL 還沒開始。

---

## 1. 資料夾導覽

```
pytorch_bootcamp/
├── 00_plan/         ← 先讀這裡！學習計畫、環境設定、資源清單
│   ├── 學習路線圖.md          24 週逐週計畫 + 每週檢核標準
│   ├── 環境與工具.md          venv / CUDA / VSCode / 實驗紀律
│   └── 資源清單.md            每個主題的必讀論文與課程
│
├── 01_syntax/       ← 語法系統課（本教材的核心價值）
│   ├── 00_看不懂時先讀這裡.md      ★★ Python 語法解碼器（看到怪符號就來查）
│   ├── 01_python進階語法.md        decorator / 生成器 / typing / dataclass
│   ├── 02_numpy.md                  ndarray / broadcasting / 向量化
│   ├── 03_pandas.md                 DataFrame / groupby / 時間序列
│   ├── 04_matplotlib.md             畫訓練曲線、混淆矩陣、特徵圖
│   ├── 05_tensor基礎.md             建立 / dtype / device / 運算
│   ├── 06_shape操作大全.md          view/reshape/permute/broadcast/einsum ★最重要
│   ├── 07_autograd.md               計算圖 / backward / detach / no_grad ★最重要
│   ├── 08_nn_Module.md              Module / Parameter / buffer / hook / init
│   ├── 09_Dataset與DataLoader.md    Dataset / collate_fn / sampler / transform
│   ├── 10_損失_優化器_排程器.md      各種 loss 的坑 / Adam vs AdamW / scheduler
│   ├── 11_訓練迴圈範本.md            可以直接拿去用的標準 template
│   ├── 12_GPU_AMP_效能.md           device / AMP / torch.compile / 記憶體
│   ├── 13_儲存載入與checkpoint.md    state_dict / weights_only / 續訓
│   ├── 14_除錯與常見錯誤.md          30 個你一定會遇到的錯誤與解法 ★最實用
│   └── 15_同性質語法差異比較.md      ★★ 27 組「這兩個差在哪、該用哪個」
│
├── 02_topics/       ← 九大主題教程（理論 + 數學 + PyTorch 實作 + 陷阱）
│   ├── 01_ANN.md
│   ├── 02_CNN.md
│   ├── 03_Transformer.md
│   ├── 04_AutoEncoder.md
│   ├── 05_SelfSupervised.md
│   ├── 06_GAN.md
│   ├── 07_NetworkCompression.md
│   ├── 08_RL.md
│   └── 09_EC演化計算.md
│
├── 03_code/         ← 可直接執行的完整範例（★ 全部在你的機器上實測過）
│   ├── README.md                    ← 執行順序 + 實測結果對照表
│   ├── 00_common.py                 共用工具（seed/device/AverageMeter/EarlyStopping）
│   ├── 01_tensor_playground.py      張量語法自我測驗（會自動對答案）
│   ├── 02_numpy_pandas_drill.py     用你自己的 2330.csv 做特徵工程練習
│   ├── 03_ann_mnist.py              MLP 全流程（含 train/val/test 切分）
│   ├── 04_cnn_cifar10.py            CNN + 資料增強 + 學習率排程
│   ├── 05_transformer_scratch.py    手刻 Multi-Head Attention → 完整 Encoder
│   ├── 06_autoencoder_vae.py        AE / Denoising AE / VAE 三合一
│   ├── 07_ssl_simclr.py             SimCLR 對比學習 + linear probing 評估
│   ├── 08_gan_dcgan.py              DCGAN（含訓練穩定技巧）
│   ├── 09_compression.py            剪枝 + 量化 + 知識蒸餾
│   ├── 10_rl_dqn_reinforce.py       DQN 與 REINFORCE（自製環境，免裝 gym）
│   └── 11_ec_ga_nsga2.py            GA / NSGA-II / 用 EC 調神經網路超參數
│
└── 04_cheatsheet/   ← 印出來貼在桌上
    ├── PyTorch速查表.md
    ├── NumPy_Pandas速查表.md
    └── Shape除錯速查表.md
```

### 這些程式碼不是「看起來會動」而已

`03_code/` 的每個腳本我都在你這台機器上實際跑過，部分結果：

| 驗證 | 結果 |
|---|---|
| 手刻 attention vs PyTorch 內建 | 完全一致（誤差 < 1e-5） |
| 手刻 Transformer 學序列反轉 | 800 步後 **100% 正確** |
| AE 異常偵測跑你的 2330.csv | 抓到 **2000 年 10 月網路泡沫**與 **2008-09-30 金融海嘯** |
| GAN 故意不 detach | G 的梯度範數 = 11.3（正確應為 0）—— 污染現場可重現 |
| NumPy 向量化 vs for 迴圈 | 快 **117 倍** |
| GA vs 同預算隨機搜尋 | GA 好 **5.7 倍** |
| REINFORCE 解 CartPole | 約 150 集突破 195 |
| 知識蒸餾的 T² 到底有沒有用 | **兩次實測結論相反** —— 成了教材裡「為何一定要跑多 seed」的活教材（主題 07 §4.2.1） |

完整對照表在 [`03_code/README.md`](03_code/README.md)。

**順帶抓到的一個真問題**：你的 `2330.csv` 的 `label` 欄混有 Excel 的 `#REF!` 字串，
所以 pandas 會把整欄讀成字串而不是整數。
`02_numpy_pandas_drill.py` 會把它揪出來並示範怎麼清理。

---

## 2. 建議的使用方式（很重要，別跳過）

### ⚠️ 完全看不懂程式碼在寫什麼？先做這件事

**不要硬讀。** 先花 40 分鐘把 [`01_syntax/00_看不懂時先讀這裡.md`](01_syntax/00_看不懂時先讀這裡.md)
翻過一遍（不用背，知道「有這個東西、可以來這裡查」就好），
之後看到 `@decorator`、`lambda`、`x[:, None]`、`f"{x:.4f}"` 這種符號就回去查。

程式碼檔案本身也都加了逐行語法註解，例如：

```python
# ★ 星號必須加！nn.Sequential 要的是「一層、一層、一層」，
#   不是「一個裝著層的 list」。少了 * 會報錯。
self.net = nn.Sequential(*layers)
```

猶豫「`view` 和 `reshape` 到底差在哪」這類問題時，
查 [`01_syntax/15_同性質語法差異比較.md`](01_syntax/15_同性質語法差異比較.md)，
裡面有 27 組對照表 + 可執行的驗證程式碼。

### 每天的節奏（平日 1.5～2 小時）

```
20 min  讀 01_syntax/ 的一個章節
40 min  照著打一遍程式碼（★ 用手打，不要複製貼上）
30 min  做該章節最後的「動手練習」，不看答案
10 min  把不懂的寫進自己的 questions.md
```

### 三條鐵律

1. **看到任何 tensor 運算，先在腦中念出 shape。**
   養成寫 `# [B, C, H, W]` 註解的習慣，你 90% 的 bug 都是 shape 錯。
2. **每個範例都要「弄壞它」。**
   把 `ReLU` 拿掉、把 learning rate 調成 10、把 `optimizer.zero_grad()` 註解掉，
   看它怎麼壞。**知道壞掉長什麼樣子，比知道對的長什麼樣子更值錢。**
3. **每個主題結束，寫一份 1 頁的自我筆記。**
   用自己的話解釋「這個模型解決什麼問題、核心公式是什麼、實作上最容易錯的地方」。

### 檢核你是否真的學會了

每章結尾都有「✅ 自我檢核」。如果任一項答不出來，**回頭重讀該節**，不要往下走。

---

## 3. 你的環境（已為你確認）

| 項目 | 版本 | 備註 |
|---|---|---|
| Python | 3.12.8 | OK |
| PyTorch | 2.12.1+cu126 | 支援 CUDA，很新 |
| torchvision | 0.27.1+cu126 | |
| NumPy | 2.4.4 | ★ NumPy 2.x，部分舊教學語法已失效，本教材已更新 |
| pandas | 3.0.3 | ★ pandas 3.0，Copy-on-Write 是預設，本教材已更新 |
| GPU | RTX 4060 Laptop 8GB | 足夠跑本教材所有範例 |

> ⚠️ 網路上多數教學是 NumPy 1.x / pandas 1.x 時代寫的，
> 有些語法在你的環境會直接報錯。本教材會標註 **【新版注意】**。

---

## 4. 從這裡開始

0. 補裝套件（教材裡的繪圖與模型摘要會用到）：

```bash
pip install matplotlib scikit-learn tqdm tensorboard torchinfo
```

1. 讀 [`00_plan/學習路線圖.md`](00_plan/學習路線圖.md) —— 知道整體要花多久、每週做什麼
2. 讀 [`00_plan/環境與工具.md`](00_plan/環境與工具.md) —— 把實驗紀律先建立好
3. 跑一次 `python 03_code/00_common.py` 確認環境沒問題
4. 開始 [`01_syntax/02_numpy.md`](01_syntax/02_numpy.md)

> 沒裝 matplotlib 也能跑，腳本會自動跳過繪圖並提示，不會中斷訓練。

---

## 5. 給「覺得程式碼像天書」的你

這份教材假設你**不熟 Python 的慣用寫法**（你的底子是 C/C++/Java），所以做了三層防護：

| 層級 | 在哪 | 內容 |
|---|---|---|
| **符號查詢** | `01_syntax/00_看不懂時先讀這裡.md` | 22 節，看到 `@`、`lambda`、`*args`、`x[:,None]`、`f"{x:.4f}"` 就來查，每條都有 C/Java 對照 |
| **選擇困難** | `01_syntax/15_同性質語法差異比較.md` | 27 組「這兩個差在哪、我該用哪個」，含可執行的驗證程式碼 |
| **逐行拆解** | `02_topics/` 每章 + `03_code/` 每個檔案 | 最難的那段程式碼會有「逐行拆解」小節，把一行拆成好幾步解釋 |

**建議做法**：讀程式碼時，看到不懂的**不要跳過**。
把那一行單獨抓出來、前後 `print(x.shape)`、拆成好幾行執行一次。
花五分鐘弄懂一行，比往下讀十頁有用得多。

祝學習順利。有問題就回頭查 `04_cheatsheet/`。
