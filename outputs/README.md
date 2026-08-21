# outputs/

运行产生的中间结果与日志都写到这里。

默认会生成:

- `base/`, `base_te/`, `base_td/`, `base_fm/`, `base_nn/`:各世代 OOF/test 产物;
- `logs/`:训练日志;
- 若干缓存文件(`*.npz`, `*.parquet`, `*.pt`)。

这些内容默认**不纳入 git**。仓库里只保留目录说明,保证项目干净。
