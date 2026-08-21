# -*- coding: utf-8 -*-
"""v16:dq 版 outlier 强化通道(两天刷分主线)。

思路:
  1. 复用 features.parquet 中已有的 dec/new/hist 统计量;
  2. 追加一组"恶化轨迹"特征(被拒、失活、近月断崖、merchant 收缩);
  3. 训练一组轻量 DQ 成员:
       q_lgb / q_hub / q_clf / q_clean
     输出到 outputs/base_dq/ , 无缝接入 fusion 中的 F36/F37。

用法:
  ELO_SEED=777 python src/archive/v16_dq.py all
  ELO_SEED=777 python src/archive/v16_dq.py clf
  ELO_SEED=777 python src/archive/v16_dq.py fuse
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import elo_pipeline as ep
import formula as v11
import fusion as vf

OUT_DIR = "outputs/base_dq"
SUB_OUT = "outputs/submission_v16_dq.csv"
TOP_BASE = int(os.environ.get("ELO_DQ_BASE_TOP", 160))
TOP_SEL = int(os.environ.get("ELO_DQ_SEL_TOP", 128))
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()

DQ_CLF_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.01, num_leaves=63,
    max_depth=-1, min_data_in_leaf=40, feature_fraction=0.85,
    bagging_fraction=0.9, bagging_freq=1, lambda_l1=0.2, lambda_l2=8.0,
    is_unbalance=True, verbosity=-1,
    num_threads=ep.CONFIG["N_THREADS"], seed=ep.CONFIG["SEED"])


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a / np.where(np.abs(b) > 1e-9, b, 1.0)


def build_dq(df: pd.DataFrame) -> pd.DataFrame:
    g = lambda c: df[c].fillna(0).to_numpy(np.float64) if c in df else np.zeros(len(df), np.float64)

    out = pd.DataFrame(index=df.index)

    # 原始高价值列直接带入,让树模型自行切分。
    raw_cols = [
        "hist_last_to_ref_days", "new_hist_merchant_ratio", "new_hist_count_ratio",
        "new_hist_sum_ratio", "new_count_per_month", "hist_authorized_flag_mean",
        "hist_recent3_count_ratio", "hist_recent3_sum_ratio", "dec_count",
        "dec_purchase_amount_sum", "dec_merchant_id_nunique", "hist_gap_std",
        "hist_monthsum_slope", "hist_to_new_gap_days", "hist_count_per_month",
        "hist_lag0_count", "hist_lag0_sum", "hist_lag-1_count", "hist_lag-1_sum",
        "hist_lag-6_count", "hist_lag-6_sum", "hist_mcat_entropy", "elapsed_days",
    ]
    for c in raw_cols:
        if c in df.columns:
            out[f"dq_raw_{c}"] = df[c].fillna(0).astype(np.float32)

    # 被拒/授权侧:显式放大"恶化"信号。
    out["dq_auth_drop"] = (1.0 - g("hist_authorized_flag_mean")).astype(np.float32)
    out["dq_decline_ratio"] = safe_div(g("dec_count"), np.maximum(g("hist_count"), 1)).astype(np.float32)
    out["dq_decline_amt_ratio"] = safe_div(
        g("dec_purchase_amount_sum"), np.maximum(g("hist_purchase_amount_sum"), 1)
    ).astype(np.float32)
    out["dq_decline_mer_ratio"] = safe_div(
        g("dec_merchant_id_nunique"), np.maximum(g("hist_merchant_id_nunique"), 1)
    ).astype(np.float32)
    out["dq_gap_x_decline"] = (g("hist_last_to_ref_days") * (1.0 - g("hist_authorized_flag_mean"))).astype(np.float32)
    out["dq_decline_x_entropy"] = (
        safe_div(g("dec_count"), np.maximum(g("hist_count"), 1)) * g("hist_mcat_entropy")
    ).astype(np.float32)
    out["dq_decline_x_gap"] = (
        safe_div(g("dec_count"), np.maximum(g("hist_count"), 1)) * g("hist_last_to_ref_days")
    ).astype(np.float32)

    # 近期断崖:最近 3 月/最近 1 月 vs 更早窗口。
    out["dq_recent3_drop_cnt"] = (1.0 - g("hist_recent3_count_ratio")).astype(np.float32)
    out["dq_recent3_drop_sum"] = (1.0 - g("hist_recent3_sum_ratio")).astype(np.float32)
    out["dq_lag0_vs_lag1_cnt"] = safe_div(g("hist_lag0_count"), np.maximum(g("hist_lag-1_count"), 1)).astype(np.float32)
    out["dq_lag0_vs_lag1_sum"] = safe_div(g("hist_lag0_sum"), np.maximum(g("hist_lag-1_sum"), 1)).astype(np.float32)
    out["dq_lag0_vs_lag6_cnt"] = safe_div(g("hist_lag0_count"), np.maximum(g("hist_lag-6_count"), 1)).astype(np.float32)
    out["dq_lag0_vs_lag6_sum"] = safe_div(g("hist_lag0_sum"), np.maximum(g("hist_lag-6_sum"), 1)).astype(np.float32)
    out["dq_lag_cliff_cnt"] = safe_div(g("hist_lag-6_count") - g("hist_lag0_count"),
                                       np.maximum(g("hist_lag-6_count"), 1)).astype(np.float32)
    out["dq_lag_cliff_sum"] = safe_div(g("hist_lag-6_sum") - g("hist_lag0_sum"),
                                       np.maximum(g("hist_lag-6_sum"), 1)).astype(np.float32)
    out["dq_monthsum_neg_slope"] = (-g("hist_monthsum_slope")).astype(np.float32)

    # 新商户/merchant 收缩:流失用户往往在 new/hist 上出现塌缩。
    out["dq_new_shrink_cnt"] = (1.0 - np.clip(g("new_hist_count_ratio"), 0, 5)).astype(np.float32)
    out["dq_new_shrink_sum"] = (1.0 - np.clip(g("new_hist_sum_ratio"), 0, 5)).astype(np.float32)
    out["dq_new_shrink_mer"] = (1.0 - np.clip(g("new_hist_merchant_ratio"), 0, 5)).astype(np.float32)
    out["dq_new_gap_norm"] = safe_div(g("hist_to_new_gap_days"), np.maximum(g("elapsed_days"), 1)).astype(np.float32)

    # 近期失活/非活跃。
    out["dq_recent_inactive"] = safe_div(g("hist_last_to_ref_days"), np.maximum(g("elapsed_days"), 1)).astype(np.float32)
    out["dq_low_activity"] = safe_div(1.0, np.maximum(g("hist_count_per_month"), 0.2)).astype(np.float32)

    return out.replace([np.inf, -np.inf], 0).fillna(0)


def load_te():
    for p in ("outputs/te_features.npz", "outputs/te_features_v1.npz"):
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            return z["tr"], z["te"], [str(x) for x in z["names"]]
    raise FileNotFoundError("缺少 TE 缓存 outputs/te_features.npz 或 outputs/te_features_v1.npz")


def assemble():
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]

    imp = pd.read_csv("outputs/feature_importance.csv")
    top_base = [c for c in imp[imp["gain"] > 0].head(TOP_BASE)["feature"] if c in train.columns]

    dq_tr = build_dq(train)
    dq_te = build_dq(test)

    blocks_tr = [train[top_base].reset_index(drop=True)]
    blocks_te = [test[top_base].reset_index(drop=True)]
    desc = [f"base {len(top_base)}"]

    if os.environ.get("ELO_DQ_USE_TE", "1") == "1":
        te_tr, te_te, te_names = load_te()
        blocks_tr.append(pd.DataFrame(te_tr, columns=te_names))
        blocks_te.append(pd.DataFrame(te_te, columns=te_names))
        desc.append(f"TE {len(te_names)}")

    if os.environ.get("ELO_DQ_USE_TD", "1") == "1":
        td = pd.read_parquet("outputs/td_features.parquet")
        td_tr = train[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        td_te = test[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        blocks_tr.append(td_tr.astype(np.float32).reset_index(drop=True))
        blocks_te.append(td_te.astype(np.float32).reset_index(drop=True))
        desc.append(f"TD {td_tr.shape[1]}")

    if os.environ.get("ELO_DQ_USE_FM", "1") == "1":
        hl = v11.hist_lag_amt()
        aux = base.merge(hl, on="card_id", how="left")
        aux_tr = aux[aux["is_train"] == 1].reset_index(drop=True)
        aux_te = aux[aux["is_train"] == 0].reset_index(drop=True)
        fm_tr = v11.formula_block(aux_tr).astype(np.float32)
        fm_te = v11.formula_block(aux_te).astype(np.float32)
        blocks_tr.append(fm_tr.reset_index(drop=True))
        blocks_te.append(fm_te.reset_index(drop=True))
        desc.append(f"FM {fm_tr.shape[1]}")

    blocks_tr.append(dq_tr.reset_index(drop=True))
    blocks_te.append(dq_te.reset_index(drop=True))
    desc.append(f"DQ {dq_tr.shape[1]}")

    X = pd.concat(blocks_tr, axis=1)
    X_test = pd.concat(blocks_te, axis=1)
    log(f"候选矩阵 X={X.shape}(" + " + ".join(desc) + ")")
    return train, test, y, X.astype(np.float32), X_test.astype(np.float32), dq_tr


def select_features(X: pd.DataFrame, y: pd.Series):
    ybin = (y < -30).astype(int).to_numpy()
    folds3 = StratifiedKFold(3, shuffle=True, random_state=ep.CONFIG["SEED"])
    imp = np.zeros(X.shape[1], np.float64)
    aucs = []
    for tr, va in folds3.split(X, ybin):
        m = lgb.train(
            DQ_CLF_PARAMS,
            lgb.Dataset(X.iloc[tr], ybin[tr]),
            4000,
            valid_sets=[lgb.Dataset(X.iloc[va], ybin[va])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pv = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        aucs.append(roc_auc_score(ybin[va], pv))
        imp += m.feature_importance("gain")
    imp_df = pd.DataFrame({"feature": X.columns, "gain": imp / 3}).sort_values("gain", ascending=False)
    sel = imp_df[imp_df["gain"] > 0].head(TOP_SEL)["feature"].tolist()
    log(f"DQ clf 3折筛列: AUC={np.mean(aucs):.5f} | {X.shape[1]} -> {len(sel)}")
    log("DQ 列 gain 前 12:\n" + imp_df.head(12).to_string(index=False))
    return sel


def cv_dq_clf(X, y, X_test, folds):
    ybin = (y < -30).astype(int).to_numpy()
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        m = lgb.train(
            DQ_CLF_PARAMS,
            lgb.Dataset(X.iloc[tr], ybin[tr]),
            10000,
            valid_sets=[lgb.Dataset(X.iloc[va], ybin[va])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        log(f"  [dq_clf] fold{k + 1}: AUC={roc_auc_score(ybin[va], oof[va]):.5f} (iter={m.best_iteration})")
    auc = roc_auc_score(ybin, oof)
    log(f"[dq] clf AUC={auc:.5f}")
    return oof, pred, auc


def cv_clean(X, y, X_test, folds):
    mask = (y < -30).to_numpy()
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        tr_c, va_c = tr[~mask[tr]], va[~mask[va]]
        m = lgb.train(
            ep.LGB_PARAMS,
            lgb.Dataset(X.iloc[tr_c], y.iloc[tr_c]),
            10000,
            valid_sets=[lgb.Dataset(X.iloc[va_c], y.iloc[va_c])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        log(f"  [dq_clean] fold{k + 1}: iter={m.best_iteration}")
    return oof, pred


def dump(name, y, oof, pred):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"{name}.npz"), oof=oof, pred=pred)
    log(f"[dq] {name:5s} OOF={rmse(y, oof):.5f} -> {OUT_DIR}/{name}.npz")


def focused_fusion():
    bases = vf.load_bases()
    base_tbl = pd.read_parquet("data/processed/features.parquet")
    train = base_tbl[base_tbl["is_train"] == 1].reset_index(drop=True)
    test = base_tbl[base_tbl["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)

    reg = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    treg = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    dreg = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    freg = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    nreg = sorted(k for k in bases if k.startswith("n_"))
    qreg = [k for k in ("q_lgb", "q_xgb", "q_cat", "q_hub") if k in bases]

    assert freg and nreg and "f_clf" in bases and "f_clean" in bases, "基线 F31 成员不完整"
    allf3 = (
        reg + treg + dreg + freg
        + (["t_clf", "t_clean"] if "t_clf" in bases else [])
        + (["d_clf", "d_clean"] if "d_clf" in bases else [])
        + ["f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"]
    )
    allf4 = allf3 + nreg
    base_r, _, base_pred = vf.evaluate(allf4, "bayes", bases, y, ybin, folds,
                                       p_src="f_clf", clean_src="f_clean")
    log(f"[dq_fuse] F31 bayes={base_r:.5f}")

    best = ("F31", base_r, base_pred)
    if qreg and "q_clf" in bases and "q_clean" in bases:
        allf6 = allf4 + qreg + ["q_clf", "q_clean"]
        for tag, mdl in [("F36 全池+DQ", "ridge"), ("F37 全池+DQ bayes", "bayes")]:
            r, _, pred = vf.evaluate(allf6, mdl, bases, y, ybin, folds,
                                     p_src="q_clf", clean_src="q_clean")
            log(f"[dq_fuse] {tag} {mdl} OOF={r:.5f} vs F31 {base_r:.5f} -> {r - base_r:+.5f}")
            if r < best[1]:
                best = (tag, r, pred)

    if best[0] != "F31":
        pd.DataFrame({"card_id": test["card_id"], "target": best[2]}).to_csv(SUB_OUT, index=False)
        log(f"[dq_fuse] 保存 {SUB_OUT} ({best[0]} OOF={best[1]:.5f})")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "fuse":
        focused_fusion()
        return

    train, test, y, X, X_test, dq_tr = assemble()
    folds = ep.make_folds(y)
    sel = select_features(X, y)
    Xs, Xs_test = X[sel], X_test[sel]
    log(f"DQ 训练矩阵 X={Xs.shape}")

    ybin = (y < -30).astype(int).to_numpy()
    # 记录单变量 AUC 最强的 DQ 列,便于快速判断这批特征有没有信息。
    auc_rows = []
    for c in dq_tr.columns:
        v = np.nan_to_num(dq_tr[c].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(v, v[0]):
            continue
        a = max(roc_auc_score(ybin, v), roc_auc_score(ybin, -v))
        auc_rows.append((c, a))
    auc_rows.sort(key=lambda x: x[1], reverse=True)
    log("单列 DQ AUC 前 10:\n" + "\n".join(f"  {k:22s} {a:.5f}" for k, a in auc_rows[:10]))

    os.makedirs(OUT_DIR, exist_ok=True)
    if mode in ("lgb", "all"):
        oof, pred, _, _ = ep.cv_lightgbm(Xs, y, Xs_test, folds, ep.LGB_PARAMS, "lgb+dq")
        dump("lgb", y, oof, pred)
        oof, pred, _, _ = ep.cv_lightgbm(Xs, y, Xs_test, folds, ep.HUB_PARAMS, "hub+dq")
        dump("hub", y, oof, pred)

    if mode in ("clf", "all"):
        oof, pred, auc = cv_dq_clf(Xs, y, Xs_test, folds)
        np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
        log(f"[dq] clf AUC={auc:.5f} -> {OUT_DIR}/clf.npz")

    if mode in ("clean", "all"):
        oof, pred = cv_clean(Xs, y, Xs_test, folds)
        dump("clean", y, oof, pred)

    if mode in ("all", "fusion"):
        focused_fusion()


if __name__ == "__main__":
    main()
