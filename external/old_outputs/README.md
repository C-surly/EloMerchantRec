# old_outputs

本目录内置 `SC5` 和 `U2` 所需的旧仓辅助输出。

包含:

- 旧仓基础世代输出
- 旧仓 TE / TD / FM / NN 输出
- `ct` / `tp` / `sk` / `ssl` 等终局补充成员

用途:

- 为 `src/blending/pool_sc5.py` 与 `src/blending/pool_union.py` 提供跨仓成员输入
- 让当前仓可以独立完成第六名终局融合,不再依赖本机绝对路径
