# src 目录结构

当前发布仓只保留复现 `3.59428` 所需的代码。

## 主链路

| 脚本 | 职责 |
|---|---|
| `elo_pipeline.py` | 主特征工程与基础世代训练 |
| `hetero.py` | 异构基学习器 |
| `target_encoding.py` | TE 世代 |
| `timediff.py` | TD 世代 |
| `formula.py` | FM 世代 |
| `seq_gru.py` / `seq_nn.py` / `nn_runtime.py` | 序列模型 |
| `fusion.py` | 融合公共库 |
| `fuse_final.py` | 主链路终局融合 |
| `fuse_opt.py` | 融合层扩展工具,供 `SC5` / `U2` 调用 |

## 第六名补充

| 脚本 | 职责 |
|---|---|
| `extras/v15_nn_clf.py` | 训练 `nn_clf_parts` 与 `base_nn_clf/clf.npz` |
| `extras/v16_dq.py` | 训练 `base_dq` 系列成员 |
| `extras/tx_seq.py` | 生成 `ssl` 系列依赖的交易级序列缓存 |
| `extras/tp_lgb.py` | 生成 `tp_lgb` |
| `extras/ct_lgb.py` / `extras/ct_gen.py` | 生成 `ct` 家族回归成员 |
| `extras/ct_clf.py` | 生成 `ct_clf` |
| `extras/ssl_clf.py` / `extras/ssl_full_clf.py` / `extras/ssl_dn_clf.py` | 生成 `ssl` 家族分类成员 |
| `extras/tp_pfn.py` | 生成 `tp_pfn` |
| `extras/sk_row.py` | 生成 `sk_row` |
| `blending/pool_sc5.py` | 生成 `SC5` |
| `blending/pool_union.py` | 生成 `U2` |
| `blending/blend_rank6.py` | 生成最终 `3.59428` 提交 |
| `blending/paths.py` | 统一管理旧仓辅助输出路径 |

## 运行方式

- 主链路: `bash run_all.sh`
- 完整第六名链路: `bash run_rank6.sh`
- `extras/` 表示主链路之外、终局复现所需的补充组件
- 默认模式下 `run_rank6.sh` 按 `frozen-first` 口径运行,先走历史冻结成员
- 若设置 `ELO_PREFER_LOCAL_RANK6=1`,`blending/` 才会优先读取当前仓 `outputs/`
