# 03_code 使用說明

## 這些檔案都有逐行語法註解

每支程式的開頭都有一段「本檔用到的語法」導覽，程式碼中間也標了為什麼要這樣寫，例如：

```python
# ③ 清空上一步的梯度。★ PyTorch 的 .grad 是「累加」不是「覆寫」，
#    忘記這行 -> 梯度越滾越大 -> loss 爆炸
#    set_to_none=True 是把 .grad 設成 None（比填 0 快，也省記憶體）
optimizer.zero_grad(set_to_none=True)
```

看到看不懂的符號（`@decorator`、`lambda`、`x[:, None]`、`*layers`）時，
查 [`../01_syntax/00_看不懂時先讀這裡.md`](../01_syntax/00_看不懂時先讀這裡.md)。
猶豫「這兩個寫法差在哪」時，查
[`../01_syntax/15_同性質語法差異比較.md`](../01_syntax/15_同性質語法差異比較.md)。

## 先跑這個

```bash
python 00_common.py
```

會做兩件事：自我測試，並產生 `common.py`（其他檔案靠它 `from common import ...`）。

## 需要的套件

```bash
pip install matplotlib scikit-learn tqdm tensorboard torchinfo
```

沒裝 `matplotlib` 也能跑，只是會跳過繪圖並提示。

## 檔案一覽與建議執行順序

| 檔案 | 對應章節 | 建議指令 |
|---|---|---|
| `00_common.py` | 11 訓練迴圈 | `python 00_common.py` |
| `01_tensor_playground.py` | 05/06 tensor & shape | `python 01_tensor_playground.py --quiz` 先自測，再不加參數對答案 |
| `02_numpy_pandas_drill.py` | 02/03 numpy & pandas | `python 02_numpy_pandas_drill.py`（用你自己的 2330.csv） |
| `03_ann_mnist.py` | 主題 01 ANN | `--overfit` → 一般訓練 → `--experiments` |
| `04_cnn_cifar10.py` | 主題 02 CNN | `--epochs 60`，再跑 `--no-aug` / `--model plain` 對照 |
| `05_transformer_scratch.py` | 主題 03 Transformer | `--verify` → `--task copy` → `--compare` |
| `06_autoencoder_vae.py` | 主題 04 AE/VAE | `--model ae/dae/vae`、`--anomaly` |
| `07_ssl_simclr.py` | 主題 05 SSL | `--loss-test` → `--method simclr` → `--method simsiam --no-stopgrad` |
| `08_gan_dcgan.py` | 主題 06 GAN | 預設 → `--no-detach` → `--loss wgangp` |
| `09_compression.py` | 主題 07 壓縮 | `--all` |
| `10_rl_dqn_reinforce.py` | 主題 08 RL | `--algo reinforce --episodes 500` → `--algo dqn --episodes 1500` |
| `11_ec_ga_nsga2.py` | 主題 09 EC | `--demo ga` → `--demo nsga2 --generations 400` → `--demo pareto` |

## 沒網路時

所有需要下載資料集的腳本都吃 `--synthetic`，會改用合成資料，
流程一模一樣，只是數字沒有意義。用來確認「程式跑得動」很方便。

## 已在你的機器上實測過的結果（RTX 4060 Laptop 8GB, torch 2.12.1+cu126）

| 腳本 | 結果 |
|---|---|
| `01_tensor_playground.py` | 8 組測驗全數通過 |
| `02_numpy_pandas_drill.py` | 向量化比 for 迴圈快 **117 倍**；抓到 `2330.csv` 的 `label` 欄混有 Excel 的 `#REF!` 髒值 |
| `05_transformer_scratch.py --verify` | 手刻 attention 與 `F.scaled_dot_product_attention` 完全一致；因果性驗證誤差 0 |
| `05_transformer_scratch.py --task copy` | 800 步後序列反轉任務 **100% 正確** |
| `06_autoencoder_vae.py --anomaly` | 重建誤差最大的日子集中在 **2000 年 10 月（網路泡沫）與 2008-09-30（金融海嘯）** |
| `07_ssl_simclr.py --loss-test` | 隨機輸入的 InfoNCE loss 2.82，理論值 `log(2N-1)=2.71`，吻合 |
| `08_gan_dcgan.py --no-detach` | 訓練 D 之後 G 的梯度範數 = 11.3（正確做法應為 0）—— 污染現場 |
| `09_compression.py --kd` | **兩次實測結論相反**（弱 teacher 時 T² 勝、強 teacher 時 T² 敗）—— 單 seed 的差異跟效果量同級，是教材裡「為何一定要多 seed」的活教材，見主題 07 §4.2.1 |
| `10_rl_dqn_reinforce.py --algo reinforce` | 約 150 集突破 195，500 集穩定在 300~420 |
| `10_rl_dqn_reinforce.py --algo dqn` | 收斂較慢，600 集約 90，建議 1500 集以上 |
| `11_ec_ga_nsga2.py --demo ga` | GA 比同預算隨機搜尋好 **5.7 倍**（Rastrigin） |
| `11_ec_ga_nsga2.py --demo nsga2` | ZDT1：150 代誤差 1.22、**400 代誤差 0.03**（2.4 秒） |
| `11_ec_ga_nsga2.py --demo pareto` | 產出準確率 0.747~0.838 對應 0.56K~1.41K 參數的 Pareto front |

## 輸出位置

所有圖表、checkpoint、json 都寫到 `outputs/<實驗名>/`。
