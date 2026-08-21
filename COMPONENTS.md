# Components

## 主链路

| 路径 | 作用 |
| --- | --- |
| `src/elo_pipeline.py` | 生成主特征并训练基础世代模型 |
| `src/hetero.py` | 训练异构基学习器 |
| `src/target_encoding.py` | 生成并训练 TE 世代 |
| `src/timediff.py` | 生成并训练 TD 世代 |
| `src/formula.py` | 生成并训练 FM 世代 |
| `src/seq_gru.py` | 生成月序列缓存并训练 GRU |
| `src/seq_nn.py` | 训练 `gru_x` 与 `trf` |
| `src/fuse_final.py` | 生成主链路融合结果 |

## 第六名补充组件

| 路径 | 作用 |
| --- | --- |
| `src/archive/v15_nn_clf.py` | 训练 `nn_clf_parts` 并合并为 `base_nn_clf/clf.npz` |
| `src/archive/v16_dq.py` | 训练 `base_dq` 系列成员 |
| `src/blending/pool_sc5.py` | 生成 `SC5` |
| `src/blending/pool_union.py` | 生成 `U2` |
| `src/blending/blend_rank6.py` | 按 `0.6 * U2 + 0.4 * SC5` 导出最终结果 |

## 辅助产物

| 路径 | 作用 |
| --- | --- |
| `external/old_outputs/` | `SC5` / `U2` 所需的旧仓辅助成员 |
| `artifacts/card_order.csv` | 最终参考卡序 |
| `artifacts/f1_pred.npy` | 最终参考预测向量 |
| `run_all.sh` | 主链路运行入口 |
| `run_rank6.sh` | 第六名完整运行入口 |

## 完整性结论

当前仓库已经从“结果导出仓”升级为“代码复现仓”。就复现 `3.59428` 这一目标来说，代码组件和辅助产物已经齐全。
