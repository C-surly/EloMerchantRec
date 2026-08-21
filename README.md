# Elo Merchant Final Repro

本仓库用于通过代码复现 `Private RMSE 3.59428` 对应方案。

## 复现目标

执行完整链路后会生成:

- `outputs/v39/submission_v39_best.csv`
- `outputs/v39/submission_v39b_union.csv`
- `submission/submission_final_3.59428.csv.gz`

最终结果由代码按:

```text
F1 = 0.6 * U2 + 0.4 * SC5
```

直接生成，并和仓内参考向量 `artifacts/f1_pred.npy` 做逐位校验。

## 仓库定位

这是一个**代码复现仓**，不是只保存提交文件的结果仓。

仓库内已经包含:

- 完整复现链路所需的主流程代码
- `tp` / `ct` / `ssl` / `sk` 补充成员的上游脚本
- `SC5` / `U2` 终局融合代码
- 默认稳定复现口径所需的冻结成员输入
- 最终参考向量，用于校验生成结果是否与 `3.59428` 对齐

仓库外仍需你自己提供:

- Kaggle Elo 原始数据 `data/raw/*.csv`
- 足够的运行时间和硬件资源

## 默认复现口径

- 默认执行 `bash run_final_repro.sh` 走的是 **frozen-first** 口径,目标是稳定复现 `3.59428`
- 这个口径会优先读取仓内 `external/frozen_members/` 中随仓提供的历史冻结成员
- 若设置 `ELO_PREFER_LOCAL_FINAL=1`,则切到 **local-first** 探测模式,优先读取当前仓现场产物
- `local-first` 目前仍是对齐排查模式,还不能保证最终向量继续等于 `3.59428`

## 环境

```bash
python -m pip install -r requirements.txt
```

推荐:

- Python 3.11
- 至少 1 张 GPU
- `data/raw/` 下放置 Kaggle Elo 原始 CSV

## 运行模式

| 模式 | 命令 | 用途 |
| --- | --- | --- |
| 主链路 | `bash run_all.sh` | 从原始数据生成当前仓主链路产物 |
| 完整复现链路 | `bash run_final_repro.sh` | 在主链路基础上继续生成 `nn_clf`、`dq`、`tp/ct/ssl/sk`、`SC5`、`U2` 和最终 `3.59428` |
| 终局校验 | `python src/blending/blend_final.py` | 在 `SC5` 与 `U2` 已存在时重新生成最终提交并校验 |

## 时间预期

- 从零跑完整主链路 + 终局补充链路: 小时级
- 已有主链路缓存、只补 `SC5/U2/F1`: 分钟级到十几分钟级
- 仅重导最终 `F1 = 0.6 * U2 + 0.4 * SC5`: 秒级

## 数据准备

将 Kaggle Elo Merchant Category Recommendation 的原始文件放到:

```text
data/raw/
├── historical_transactions.csv
├── merchants.csv
├── new_merchant_transactions.csv
├── sample_submission.csv
├── test.csv
└── train.csv
```

## 一键运行

```bash
bash run_final_repro.sh
```

可选环境变量:

- `ELO_NN_DEVICE`: `GRU` / `GRU_X` / `NN_CLF` 使用的 GPU，默认 `0`
- `ELO_TRF_DEVICE`: `Transformer` 使用的 GPU，默认与 `ELO_NN_DEVICE` 相同
- `ELO_FROZEN_MEMBERS_DIR`: 历史冻结成员目录，默认使用仓内 `external/frozen_members`
- `ELO_PREFER_LOCAL_FINAL=1`: 让 `SC5/U2` 优先读取当前仓现算成员；默认关闭，以保持冻结口径 `3.59428`

脚本会优先复用已经存在的中间产物，不会重复训练已完成步骤。默认模式下 `SC5/U2` 仍按冻结口径优先读取历史成员；打开 `ELO_PREFER_LOCAL_FINAL=1` 后才切到“当前仓 `outputs/` 优先，`external/frozen_members` 兜底”。

## 目录

```text
EloMerchantRec/
├── README.md
├── COMPONENTS.md
├── requirements.txt
├── run_all.sh
├── run_final_repro.sh
├── artifacts/
│   ├── card_order.csv
│   └── f1_pred.npy
├── data/
│   └── processed/.gitkeep
├── external/
│   └── frozen_members/
├── outputs/
│   └── README.md
├── src/
│   ├── elo_pipeline.py
│   ├── hetero.py
│   ├── target_encoding.py
│   ├── timediff.py
│   ├── formula.py
│   ├── seq_gru.py
│   ├── seq_nn.py
│   ├── nn_runtime.py
│   ├── fusion.py
│   ├── fuse_final.py
│   ├── fuse_opt.py
│   ├── extras/
│   │   ├── nn_clf.py
│   │   ├── dq.py
│   │   ├── tx_seq.py
│   │   ├── tp_lgb.py
│   │   ├── ct_lgb.py
│   │   ├── ct_gen.py
│   │   ├── ct_clf.py
│   │   ├── ssl_clf.py
│   │   ├── ssl_full_clf.py
│   │   ├── ssl_dn_clf.py
│   │   ├── tp_pfn.py
│   │   └── sk_row.py
│   └── blending/
│       ├── pool_sc5.py
│       ├── pool_union.py
│       └── blend_final.py
└── submission/
    └── .gitkeep
```

## 说明

- `run_all.sh` 负责生成当前仓主链路产物。
- `run_final_repro.sh` 在主链路基础上继续生成 `nn_clf`、`dq`、`tp/ct/ssl/sk`、`SC5`、`U2` 和最终提交。
- `src/extras/` 表示主链路之外、终局复现所需的补充组件。
- `external/frozen_members` 是默认复现口径所需的历史冻结成员输入，不是提交文件仓。
- 最终提交文件默认写到 `submission/submission_final_3.59428.csv.gz`。
- `artifacts/` 中的参考文件只用于校验最终结果，不参与训练。
