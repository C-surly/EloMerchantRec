# -*- coding: utf-8 -*-
"""F31 终局融合:复现 v14 最优提交(Private 3.59764,私榜等效第 7 / 4111)。

31 元特征 = REG 7(基础特征世代)+ TE 4 + TD 4 + FM 4(三代新信息特征各 4 个 GBDT)
+ 6 个世代 clf/clean 头 + 3 个折内派生(p_cal 保序校准概率 / ev 期望值解析 / 交互)
+ 3 个 NN 序列成员(GRU / 被拒交易 10 通道 GRU / Transformer,各 5-seed 平均)。
二层 BayesianRidge,按 777 分层十折折内拟合。
用法:ELO_SEED=777 python src/fuse_final.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

paths.bootstrap()

import elo_pipeline as ep
import fusion as vf


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    bases = vf.load_bases()
    base = pd.read_parquet(paths.FEATURES)
    train = base[base["is_train"] == 1].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)
    REG = ["lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et"]
    T = ["t_lgb", "t_xgb", "t_cat", "t_hub"]
    D = ["d_lgb", "d_xgb", "d_cat", "d_hub"]
    F = ["f_lgb", "f_xgb", "f_cat", "f_hub"]
    N = ["n_gru", "n_gru_x", "n_trf"]
    allf = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
    missing = [k for k in allf if k not in bases and k not in vf.DERIVED]
    assert not missing, f"缺基模型产物:{missing}(按 run_all.sh 顺序补齐)"
    r, _, pred = vf.evaluate(allf, "bayes", bases, y, ybin, folds,
                             p_src="f_clf", clean_src="f_clean")
    print(f"F31 OOF = {r:.5f}(参考值 3.62062;线上 Private 3.59764)")
    sub = pd.read_csv(paths.raw("sample_submission.csv"))
    sub["target"] = pred
    os.makedirs("submission", exist_ok=True)
    sub.to_csv(paths.sub("submission_v14_repro.csv"), index=False)
    print("已生成 submission/submission_v14_repro.csv")


if __name__ == "__main__":
    main()
