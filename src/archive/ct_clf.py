# -*- coding: utf-8 -*-
"""v26:数字结构 + 时间残差进 outlier 分类器。"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import formula as v11
import fusion as vf
from archive.tp_lgb import build_timeres
from archive.ct_lgb import build_ct, load_te

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet(paths.FEATURES)
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    tp = build_timeres(base)
    base = base.merge(build_ct(), on="card_id", how="left").merge(tp, on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv(paths.FEATURE_IMPORTANCE)
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    te_tr, te_te, te_names = load_te()
    td = pd.read_parquet(paths.out("td_features.parquet"))
    extra = [c for c in train.columns if c.startswith(("cth_", "ctn_", "tp_"))]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[extra].reset_index(drop=True).astype(np.float32)], axis=1)

    X, X_test = asm(train, te_tr, fm_tr), asm(test, te_te, fm_te)
    log(f"X={X.shape}(含 extra {len(extra)});训练 clf")
    oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
    os.makedirs(paths.out("base_ct"), exist_ok=True)
    np.savez(paths.out("base_ct", "clf_ct.npz"), oof=oof, pred=pred)
    log(f"clf+ct+tp AUC={auc:.5f} vs f_clf 0.90586")

    bases = vf.load_bases()
    ybin = (y < -30).astype(int).to_numpy()
    reg = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    t_reg = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    d_reg = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    f_reg = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    n_reg = sorted(k for k in bases if k.startswith("n_"))
    allf = (reg + t_reg + d_reg + f_reg + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + n_reg)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    bases["f_clf"] = (oof, pred)
    r1, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v26 ct_clf]:基线={r0:.5f} 替换后={r1:.5f} → Δ={d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
