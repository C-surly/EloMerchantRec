# Components

## 核心组件

| 路径 | 作用 |
| --- | --- |
| `artifacts/card_order.csv` | 保存测试集 `card_id` 的最终顺序 |
| `artifacts/f1_pred.npy` | 保存与最终结果对应的预测向量 |
| `src/export_submission.py` | 将 `card_order.csv` 与 `f1_pred.npy` 组装为提交文件 |
| `src/verify_submission.py` | 对导出的提交文件进行哈希校验 |
| `submission/submission_rank6_3.59428.csv.gz` | 仓库内已提供的最终提交文件 |
| `run.sh` | 一键执行导出与校验 |

## 完整性结论

就“复现 `3.59428` 对应提交文件”这个目标而言，当前组件是齐全的。
