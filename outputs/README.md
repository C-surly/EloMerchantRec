# outputs

运行过程中生成的中间结果会写到这里。

常见目录:

- `base/`, `base_te/`, `base_td/`, `base_fm/`, `base_nn/`: 主链路各世代 OOF/test 产物
- `base_nn_clf/`, `base_dq/`: 第六名补充成员
- `nn_clf_parts/`: `v15_nn_clf.py` 的 seed 分片
- `v39/`: `SC5`、`U2` 及其配置结果
- `logs/`: 训练与融合日志

这些内容默认不纳入 git。
