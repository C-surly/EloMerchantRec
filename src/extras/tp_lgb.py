# -*- coding: utf-8 -*-
"""v22:时间预测残差成员 tp_lgb。"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import formula as v11

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def load_te():
    for name in ("te_features_v1.npz", "te_features.npz"):
        p = paths.out(name)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            return z["tr"], z["te"], [str(x) for x in z["names"]]
    raise FileNotFoundError("缺少 TE 缓存: te_features_v1.npz / te_features.npz")


def build_timeres(base: pd.DataFrame) -> pd.DataFrame:
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))
    new["card_id"] = new["card_id"].astype(str)
    last = new.groupby("card_id")["purchase_date"].max().rename("last_new")
    ref = pd.Timestamp(ep.CONFIG["REF_DATE"])
    aux_y = ((ref - last).dt.total_seconds() / 86400.0).rename("tp_last")

    df = base.merge(aux_y, on="card_id", how="left")
    drop = {"card_id", "target", "is_train", "tp_last"}
    feats = [c for c in df.columns
             if c not in drop and "new" not in c and not c.startswith("x_")
             and pd.api.types.is_numeric_dtype(df[c])]
    log(f"aux 特征 {len(feats)} 列(hist-only+静态)")

    has = df["tp_last"].notna().to_numpy()
    Xa, ya = df.loc[has, feats], df.loc[has, "tp_last"]
    params = {**ep.LGB_PARAMS, "objective": "regression", "metric": "rmse"}
    pred = np.full(len(df), np.nan)
    idx = np.where(has)[0]
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=777).split(Xa)):
        m = lgb.train(params, lgb.Dataset(Xa.iloc[tr], ya.iloc[tr]), 10000,
                      valid_sets=[lgb.Dataset(Xa.iloc[va], ya.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pred[idx[va]] = m.predict(Xa.iloc[va], num_iteration=m.best_iteration)
        log(f"  [aux] fold{k + 1}: rmse={rmse(ya.iloc[va], pred[idx[va]]):.3f}d iter={m.best_iteration}")
    out = pd.DataFrame({"card_id": df["card_id"],
                        "tp_last": df["tp_last"].astype(np.float32),
                        "tp_pred": pred.astype(np.float32)})
    out["tp_resid"] = out["tp_last"] - out["tp_pred"]
    log(f"aux OOF rmse={rmse(ya, pred[idx]):.3f}d 覆盖率={has.mean():.3f}")
    return out


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet(paths.FEATURES)
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    tp = build_timeres(base)
    base = base.merge(tp, on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv(paths.FEATURE_IMPORTANCE)
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    te_tr, te_te, te_names = load_te()
    td = pd.read_parquet(paths.out("td_features.parquet"))
    tp_cols = ["tp_last", "tp_pred", "tp_resid"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[tp_cols].reset_index(drop=True)], axis=1)

    X, X_test = asm(train, te_tr, fm_tr), asm(test, te_te, fm_te)
    log(f"X={X.shape}(含 tp 3)")
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+tp")
    os.makedirs(paths.out("base_tp"), exist_ok=True)
    np.savez(paths.out("base_tp", "lgb.npz"), oof=oof, pred=pred)
    ref = rmse(y, np.load(paths.out("base_fm", "lgb.npz"))["oof"])
    s = rmse(y, oof)
    d = ref - s
    g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    rk = {c: int((g2["feature"] == c).to_numpy().argmax()) + 1 for c in tp_cols}
    log(f"tp 列 gain 排名:{rk}")
    log(f"判据[v22 tp_lgb]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
