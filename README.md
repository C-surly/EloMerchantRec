# Elo Merchant Rank6 Release

本仓库用于复现 `Private RMSE 3.59428` 对应的最终提交文件。

## 快速开始

```bash
python -m pip install -r requirements.txt
bash run.sh
```

运行完成后会生成:

- `submission/submission_rank6_3.59428.csv.gz`

校验成功时会输出:

```text
OK 3.59428
```

## 仓库结构

```text
EloMerchantRec-rank6-release/
├── README.md
├── COMPONENTS.md
├── requirements.txt
├── run.sh
├── artifacts/
│   ├── card_order.csv
│   └── f1_pred.npy
├── src/
│   ├── export_submission.py
│   └── verify_submission.py
└── submission/
    └── submission_rank6_3.59428.csv.gz
```

## 复现命令

```bash
python src/export_submission.py
python src/verify_submission.py
```

导出脚本会读取 `artifacts/card_order.csv` 和 `artifacts/f1_pred.npy`，重建最终提交文件。

校验脚本会检查导出结果的 `sha256` 是否等于:

```text
e708c224d1c28c790027d0e3e3b01196885b0f39272040edd0b952d51ad117e1
```

## 当前交付件

- `artifacts/card_order.csv`: 测试集 `card_id` 顺序
- `artifacts/f1_pred.npy`: 最终预测向量
- `src/export_submission.py`: 导出最终提交文件
- `src/verify_submission.py`: 校验导出结果
- `submission/submission_rank6_3.59428.csv.gz`: 已生成的最终提交文件

## 说明

这个发布仓是结果复现版，目标就是稳定导出 `3.59428` 对应的提交文件。
