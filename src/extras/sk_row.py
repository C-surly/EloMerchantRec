# -*- coding: utf-8 -*-
"""v33:senkin13 stage-2 血统移植成员 sk_row。"""
import datetime
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

PARAMS = {
    "objective": "regression", "boosting": "gbdt", "metric": "rmse",
    "max_depth": 9, "min_data_in_leaf": 70, "min_gain_to_split": 0.05,
    "reg_alpha": 0.1, "reg_lambda": 20, "num_leaves": 120, "max_bin": 350,
    "learning_rate": 0.005, "bagging_fraction": 1, "bagging_freq": 1,
    "feature_fraction": 0.2, "verbosity": -1, "num_threads": 16, "seed": 777,
}


def build_rows():
    tr = pd.read_csv(paths.raw("train.csv"))
    te = pd.read_csv(paths.raw("test.csv"))
    df = pd.concat([tr, te], axis=0)
    df["first_active_month"] = df["first_active_month"].fillna("2017-09")
    df["yearmonth"] = df["first_active_month"].map(lambda x: str(x).replace("-", "")).astype(int)
    fam = pd.to_datetime(df["first_active_month"])
    df["year"] = fam.dt.year
    df["month"] = fam.dt.month
    df["elapsed_time"] = (datetime.date(2018, 2, 1) - fam.dt.date).map(lambda x: x.days)
    df["elapsed_time_month"] = df["elapsed_time"] // 30
    df = df.drop(columns=["first_active_month"])

    n = pd.read_csv(paths.raw("new_merchant_transactions.csv"))
    n = n.sort_values(by=["card_id", "purchase_date"], ascending=True)
    n["authorized_flag"] = n["authorized_flag"].map({"Y": 1, "N": 0})
    n["category_1"] = n["category_1"].map({"Y": 1, "N": 0})
    n["category_2_ohe"] = n["category_2"]
    n["category_3_ohe"] = n["category_3"]
    n = pd.get_dummies(n, columns=["category_2_ohe", "category_3_ohe"])
    n["purchase_date"] = pd.to_datetime(n["purchase_date"])
    n["p_year"] = n["purchase_date"].dt.year
    n["p_month"] = n["purchase_date"].dt.month
    n["woy"] = n["purchase_date"].dt.isocalendar().week.astype(int)
    n["doy"] = n["purchase_date"].dt.dayofyear
    n["wday"] = n["purchase_date"].dt.dayofweek
    n["weekend"] = (n["purchase_date"].dt.weekday >= 5).astype(int)
    n["day"] = n["purchase_date"].dt.day
    n["hour"] = n["purchase_date"].dt.hour
    n["datehour"] = n["purchase_date"].dt.strftime("%Y%m%d%H").astype(np.int64)
    n["date"] = n["purchase_date"].dt.strftime("%Y%m%d").astype(np.int64)
    n["purchase_year_month"] = n["p_year"].map({2017: 72, 2018: 84}) + n["p_month"]
    dd = (datetime.date(2018, 5, 1) - n["purchase_date"].dt.date).map(lambda x: x.days)
    n["month_diff"] = dd // 30 + n["month_lag"]
    n["week_diff"] = dd // 7
    n["day_diff"] = dd
    n["pre_purchase_diff"] = n.groupby("card_id")["purchase_date"].transform(lambda x: x.diff(1).dt.days)
    n["purchase_amount_new"] = np.round(n["purchase_amount"] / 0.00150265118 + 497.06, 2)
    n["pre_purchase_amount_new_diff"] = n.groupby("card_id")["purchase_amount_new"].transform(lambda x: x.diff())
    n["refer_date"] = n["p_year"].map({2017: 0, 2018: 12}) + n["p_month"] - n["month_lag"]
    n["refer_purchase_amount_new"] = n["purchase_amount_new"] / n["refer_date"]

    m = pd.read_csv(paths.raw("merchants.csv"))
    m = m.drop_duplicates(subset=["merchant_id"], keep="first")
    m["merchant_category_1"] = m["category_1"].map({"Y": 1, "N": 0})
    m["merchant_category_4"] = m["category_4"].map({"Y": 1, "N": 0})
    m["merchant_category_2"] = m["category_2"]
    m = pd.get_dummies(m, columns=["merchant_category_2"])
    m = m.drop(columns=["category_1", "category_2", "category_4"])
    dup = [c for c in m.columns if c != "merchant_id" and c in n.columns]
    m = m.rename(columns={c: f"mer_{c}" for c in dup})
    n = n.merge(m, on="merchant_id", how="left")
    rows = df.merge(n, on="card_id", how="right")
    for c in ["merchant_id", "most_recent_sales_range", "most_recent_purchases_range", "category_3"]:
        if c in rows.columns and rows[c].dtype == "object":
            rows[c] = LabelEncoder().fit_transform(rows[c].astype(str))
    log(f"行表构建完成 {rows.shape}")
    return rows


def main():
    assert os.environ.get("ELO_SEED") == "777"
    rows = build_rows()
    drop = ["card_id", "target", "purchase_date", "is_month_start", "quarter",
            "outliers", "active_months_lag3", "yearmonth", "category_123", "dummy"]
    feats = [f for f in rows.columns if f not in drop]
    for c in feats:
        if rows[c].dtype == "object":
            rows[c] = pd.to_numeric(rows[c], errors="coerce")
        elif rows[c].dtype == "bool":
            rows[c] = rows[c].astype(np.int8)
    cat_features = [c for c in feats if "feature_" in c]
    rows[feats] = rows[feats].replace([np.inf, -np.inf], np.nan)

    base = pd.read_parquet(paths.FEATURES, columns=["card_id", "is_train", "target"])
    train_cards = base[base["is_train"] == 1].reset_index(drop=True)
    y = train_cards["target"]
    folds = ep.make_folds(y)
    tr_rows = rows[rows["target"].notnull()].reset_index(drop=True)
    te_rows = rows[rows["target"].isnull()].reset_index(drop=True)
    card_fold = {}
    for k, (_, va) in enumerate(folds):
        for cid in train_cards["card_id"].iloc[va]:
            card_fold[cid] = k
    tr_rows["fold"] = tr_rows["card_id"].map(card_fold)
    log(f"train 行 {len(tr_rows)} test 行 {len(te_rows)} 特征 {len(feats)}(cat {len(cat_features)})")

    oof_rows = np.zeros(len(tr_rows))
    pred_rows = np.zeros(len(te_rows))
    for k in range(len(folds)):
        tm = (tr_rows["fold"] != k).to_numpy()
        vm = ~tm
        dtr = lgb.Dataset(tr_rows.loc[tm, feats], tr_rows.loc[tm, "target"],
                          categorical_feature=cat_features)
        dva = lgb.Dataset(tr_rows.loc[vm, feats], tr_rows.loc[vm, "target"], reference=dtr,
                          categorical_feature=cat_features)
        bst = lgb.train(PARAMS, dtr, 10000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_rows[vm] = bst.predict(tr_rows.loc[vm, feats], num_iteration=bst.best_iteration)
        pred_rows += bst.predict(te_rows[feats], num_iteration=bst.best_iteration) / len(folds)
        log(f"fold{k} 行RMSE={rmse(tr_rows.loc[vm, 'target'], oof_rows[vm]):.5f} iters={bst.best_iteration}")

    tr_rows["p"] = oof_rows
    te_rows["p"] = pred_rows
    ag_tr = tr_rows.groupby("card_id")["p"].agg(["mean", "min", "max"])
    ag_te = te_rows.groupby("card_id")["p"].agg(["mean", "min", "max"])
    oof = train_cards[["card_id"]].merge(ag_tr, on="card_id", how="left")
    test_cards = base[base["is_train"] == 0].reset_index(drop=True)
    pred = test_cards[["card_id"]].merge(ag_te, on="card_id", how="left")
    fill = float(np.nanmean(oof["mean"]))
    cov = oof["mean"].notna().mean()
    o = oof["mean"].fillna(fill).to_numpy()
    p = pred["mean"].fillna(fill).to_numpy()
    os.makedirs(paths.out("base_sk"), exist_ok=True)
    np.savez(paths.out("base_sk", "new_rowreg.npz"), oof=o, pred=p,
             oof_min=oof["min"].fillna(fill).to_numpy(), pred_min=pred["min"].fillna(fill).to_numpy(),
             oof_max=oof["max"].fillna(fill).to_numpy(), pred_max=pred["max"].fillna(fill).to_numpy())
    zl = np.load(paths.out("base_fm", "lgb.npz"))
    log(f"sk_row 卡级 OOF={rmse(y, o):.5f} 覆盖率={cov:.2%} f_lgb 相关={np.corrcoef(o, zl['oof'])[0, 1]:.4f}")

    bases = vf.load_bases()
    bases["sk_row"] = (o, p)
    reg = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    t_reg = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    d_reg = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    f_reg = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    n_reg = sorted(k for k in bases if k.startswith("n_"))
    allf = (reg + t_reg + d_reg + f_reg + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + n_reg)
    ybin = (y < -30).astype(int).to_numpy()
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    r1, _, pt = vf.evaluate(allf + ["sk_row"], "bayes", bases, y, ybin, folds,
                            p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v33 sk_row 入池,线 0.0007]:基线={r0:.5f} 加入后={r1:.5f} → Δ={d:+.5f} "
        f"{'✅ 通过' if d > 0.0007 else '❌ 不足'}")
    if d > 0.0007:
        sub = pd.read_csv(paths.raw("sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv(paths.out("submission_v33b.csv"), index=False)
        log(f"已保存 {paths.out('submission_v33b.csv')}(待提交)")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
