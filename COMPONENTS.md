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
| `src/archive/tx_seq.py` | 生成 `ssl` 系列所需的 `tx_tensor.npz` 交易级序列缓存 |
| `src/archive/tp_lgb.py` | 生成 `base_tp/lgb.npz` |
| `src/archive/ct_lgb.py` | 生成 `base_ct/lgb.npz` 与数字结构特征缓存 |
| `src/archive/ct_gen.py` | 生成 `base_ct/{xgb,cat,hub}.npz` |
| `src/archive/ct_clf.py` | 生成 `base_ct/clf_ct.npz` |
| `src/archive/ssl_clf.py` | 生成 `base_nn_clf/ssl_clf.npz` 与 `ssl_encoder.pt` |
| `src/archive/ssl_full_clf.py` | 生成 `base_nn_clf/ssl_full_clf.npz` |
| `src/archive/ssl_dn_clf.py` | 生成 `base_nn_clf/ssl_dn_clf.npz` |
| `src/archive/tp_pfn.py` | 生成 `base_tp/pfn.npz` |
| `src/archive/sk_row.py` | 生成 `base_sk/new_rowreg.npz` |
| `src/blending/pool_sc5.py` | 生成 `SC5` |
| `src/blending/pool_union.py` | 生成 `U2` |
| `src/blending/blend_rank6.py` | 按 `0.6 * U2 + 0.4 * SC5` 导出最终结果 |

## 辅助产物

| 路径 | 作用 |
| --- | --- |
| `external/old_outputs/` | `SC5` / `U2` 的历史成员兜底输入；当前仓现算成员存在时不会优先读取这里 |
| `artifacts/card_order.csv` | 最终参考卡序 |
| `artifacts/f1_pred.npy` | 最终参考预测向量 |
| `run_all.sh` | 主链路运行入口 |
| `run_rank6.sh` | 第六名完整运行入口，负责调度 `nn_clf` / `dq` / `tp` / `ct` / `ssl` / `sk` / `SC5` / `U2` |

## 完整性结论

当前仓库已经从“结果导出仓”升级为“代码复现仓”。`SC5/U2` 所需的 `tp/ct/ssl/sk`
成员上游脚本现已并入 `src/archive/` 并接入 `run_rank6.sh`；融合层优先使用当前仓
现场产物，仅在历史成员尚未自举时回退到 `external/old_outputs/`。
