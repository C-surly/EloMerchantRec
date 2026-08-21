# Elo Merchant Rank6 Repro

本仓库用于通过代码复现 `Private RMSE 3.59428` 对应的第六名方案。

## 复现目标

执行完整链路后会生成:

- `outputs/v39/submission_v39_best.csv`
- `outputs/v39/submission_v39b_union.csv`
- `submission/submission_rank6_3.59428.csv.gz`

最终结果由代码按:

```text
F1 = 0.6 * U2 + 0.4 * SC5
```

直接生成，并和仓内参考向量 `artifacts/f1_pred.npy` 做逐位校验。

## 环境

```bash
python -m pip install -r requirements.txt
```

推荐:

- Python 3.11
- 至少 1 张 GPU
- `data/raw/` 下放置 Kaggle Elo 原始 CSV

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
bash run_rank6.sh
```

可选环境变量:

- `ELO_NN_DEVICE`: `GRU` / `GRU_X` / `NN_CLF` 使用的 GPU，默认 `0`
- `ELO_TRF_DEVICE`: `Transformer` 使用的 GPU，默认与 `ELO_NN_DEVICE` 相同
- `ELO_OLD_OUTPUTS_DIR`: 旧仓辅助输出目录，默认使用仓内 `external/old_outputs`

## 目录

```text
EloMerchantRec-rank6-release/
├── README.md
├── COMPONENTS.md
├── requirements.txt
├── run_all.sh
├── run_rank6.sh
├── run.sh
├── artifacts/
│   ├── card_order.csv
│   └── f1_pred.npy
├── data/
│   └── processed/.gitkeep
├── external/
│   └── old_outputs/
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
│   ├── archive/
│   │   ├── v15_nn_clf.py
│   │   └── v16_dq.py
│   └── blending/
│       ├── pool_sc5.py
│       ├── pool_union.py
│       └── blend_rank6.py
└── submission/
    └── .gitkeep
```

## 说明

- `run_all.sh` 负责生成当前仓主链路产物。
- `run_rank6.sh` 在主链路基础上继续生成 `nn_clf`、`dq`、`SC5`、`U2` 和最终第六名提交。
- `external/old_outputs` 已内置最终融合所需的旧仓辅助输出，因此不再依赖本机绝对路径。
