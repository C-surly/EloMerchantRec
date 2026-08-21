# -*- coding: utf-8 -*-
"""v24:ct 世代整代注入。"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import formula as v11
import fusion as vf
from archive.ct_lgb import build_ct, load_te

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    ct = build_ct()
    base = pd.read_parquet(paths.FEATURES)
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base = base.merge(ct, on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv(paths.FEATURE_IMPORTANCE)
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    te_tr, te_te, te_names = load_te()
    td = pd.read_parquet(paths.out("td_features.parquet"))
    ct_cols = [c for c in ct.columns if c != "card_id"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[ct_cols].reset_index(drop=True).astype(np.float32)], axis=1)

    X, X_test = asm(train, te_tr, fm_tr), asm(test, te_te, fm_te)
    log(f"X={X.shape};训练 ct 世代 xgb/hub/cat")
    os.makedirs(paths.out("base_ct"), exist_ok=True)
    for name in ("xgb", "hub", "cat"):
        out = paths.out("base_ct", f"{name}.npz")
        if os.path.exists(out):
            log(f"[{name}] 已存在,跳过")
            continue
        if name == "xgb":
            oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds)
        elif name == "cat":
            oof, pred, _ = ep.cv_catboost(X, y, X_test, folds)
        else:
            oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "ct_hub")
        np.savez(out, oof=oof, pred=pred)
        log(f"[ct_{name}] OOF={rmse(y, oof):.5f}")

    bases = vf.load_bases()
    ct_keys = []
    for name, key in [("lgb", "ct_lgb"), ("xgb", "ct_xgb"), ("cat", "ct_cat"), ("hub", "ct_hub")]:
        d = np.load(paths.out("base_ct", f"{name}.npz"))
        bases[key] = (d["oof"], d["pred"])
        ct_keys.append(key)
    ybin = (y < -30).astype(int).to_numpy()
    reg = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    t_reg = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    d_reg = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    f_reg = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    n_reg = sorted(k for k in bases if k.startswith("n_"))
    heads = ["t_clf", "t_clean", "d_clf", "d_clean", "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"]
    allf = reg + t_reg + d_reg + f_reg + heads + n_reg
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    log(f"基线(33 成员)={r0:.5f}")
    best = (-1.0, None, None)
    for tag, feats in [("追加 ct 世代", allf + ct_keys),
                       ("f 世代换 ct 世代", reg + t_reg + d_reg + ct_keys + heads + n_reg)]:
        r1, _, pt = vf.evaluate(feats, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
        d = r0 - r1
        log(f"  {tag}: {r1:.5f} → Δ={d:+.5f}")
        if d > best[0]:
            best = (d, tag, pt)
    d, tag, pt = best
    log(f"判据[v24 ct 世代]:最优[{tag}] Δ={d:+.5f} "
        f"{'✅ 通过(定性:新信息✓)' if d > 0.0005 else '❌ 不足'}")
    if d > 0.0005:
        sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv(paths.out("submission_v24_ct.csv"), index=False)
        log(f"已保存 {paths.out('submission_v24_ct.csv')}")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
