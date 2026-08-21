# -*- coding: utf-8 -*-
"""v7:21st 剩余两个未移植点的验证(A 线特征 / B 线参数)。

A 线 [feat|lgb]:交易粒度时间差分布,移植自 refs/ELO/zxs/pre_0214.ipynb cell 43/50。
    每笔交易算三个天数差,按卡聚合 mean/median/max/min/std,hist/new 双表共 30 列:
      a2p  = 开卡月(first_active_month)→ 购买日
      p2r  = 购买日 → 参考月末(参考月 = purchase 月 - month_lag,与管线 ref_month_id 同口径;
             21st 用首笔交易日推参考日保留了日粒度噪声,这里统一锚定月末,口径更干净)
      p2now= 购买日 → REF_DATE(2018-05-01)
    我们现状只有卡粒度首笔/末笔时间,无交易粒度分布统计 —— 这是"新信息"候选。
    三个量均不含 target,无泄漏风险。

B 线 [weakreg]:21st 弱正则浅树参数(refs/ELO/zxs/reg_single_cv3.63480.ipynb),
    同特征集(sel + TE-v1)只换参数:leaves 31 / min_leaf 30 / depth -1 / L1 0.1 / 无 L2
    / ff 0.9 / bf 0.9,对照我们的 leaves 63 / depth 8 / min_leaf 150 / L1 1 / L2 10。

判据:单模 lgb OOF 相对 outputs/base_te/lgb.npz(sel+TE-v1,3.64170)改善 >0.0005 才继续;
     >0.003 触发 submissions.md 泄漏警报线(B 线为纯参数改动,警报仅对 A 线有意义)。

用法:python src/timediff.py feat|lgb|weakreg|all
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

TD_CACHE = "outputs/td_features.parquet"
OUT_DIR = "outputs/base_td"
REF_LGB = "outputs/base_te/lgb.npz"     # 判据基线:sel + TE-v1 单模
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()

WEAK_PARAMS = dict(   # 21st reg_single_cv3.63480 原参 + 我们的线程/种子约定
    objective="regression", metric="rmse", boosting="gbdt",
    learning_rate=0.01, num_leaves=31, max_depth=-1, min_data_in_leaf=30,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    lambda_l1=0.1, verbosity=-1,
    num_threads=ep.CONFIG["N_THREADS"], seed=ep.CONFIG["SEED"])


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_td() -> pd.DataFrame:
    """交易粒度时间差 → 卡粒度分布统计,缓存 parquet。"""
    fam = pd.concat([
        pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], f), usecols=["card_id", "first_active_month"])
        for f in ("train.csv", "test.csv")])
    fam = pd.Series(pd.to_datetime(fam["first_active_month"].values, format="%Y-%m", errors="coerce"),
                    index=fam["card_id"].values)
    now = pd.Timestamp(ep.CONFIG["REF_DATE"])

    out = []
    for fname, prefix in (("historical_transactions.csv", "hist"),
                          ("new_merchant_transactions.csv", "new")):
        df = ep.clean_transactions(ep.load_transactions(fname))
        # 参考月末:月序号 mid(0 起)- month_lag 即参考月,其下月首日 = 参考月末界
        mid = df["purchase_date"].dt.year * 12 + (df["purchase_date"].dt.month - 1)
        ref = (mid - df["month_lag"] + 1).astype(np.int32)   # 参考月的下一个月(0 起序号)
        ref_end = pd.to_datetime(pd.DataFrame(
            {"year": ref // 12, "month": ref % 12 + 1, "day": 1}))
        td = pd.DataFrame({"card_id": df["card_id"]})
        td["a2p"] = (df["purchase_date"] - df["card_id"].map(fam)).dt.days.astype(np.float32)
        td["p2r"] = (ref_end - df["purchase_date"]).dt.days.astype(np.float32)
        td["p2now"] = (now - df["purchase_date"]).dt.days.astype(np.float32)
        agg = td.groupby("card_id", observed=True).agg(
            {c: ["mean", "median", "max", "min", "std"] for c in ("a2p", "p2r", "p2now")})
        agg.columns = [f"{prefix}_td_{c}_{s}" for c, s in agg.columns]
        agg.index = agg.index.astype(str)
        out.append(agg)
        log(f"{prefix}: {agg.shape[1]} 列,{len(agg)} 卡")
        del df, td
    res = out[0].join(out[1], how="outer").reset_index().rename(columns={"index": "card_id"})
    res.to_parquet(TD_CACHE)
    log(f"时间差特征缓存 {TD_CACHE}: {res.shape}")
    return res


def load_te(n_expect=36):
    """读 v6 TE-v1 缓存(36 列折外 outlier 率编码),口径与折划分和本脚本一致。"""
    for p in ("outputs/te_features_v1.npz", "outputs/te_features.npz"):
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            names = [str(x) for x in z["names"]]
            if len(names) == n_expect:
                log(f"读取 TE-v1 缓存 {p}({len(names)} 列)")
                return z["tr"], z["te"], names
    raise SystemExit("找不到 36 列的 TE-v1 缓存(te_features_v1.npz / te_features.npz)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("feat", "all") or (mode == "lgb" and not os.path.exists(TD_CACHE)):
        build_td()
    if mode == "feat":
        return

    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    te_tr, te_te, te_names = load_te()
    X = pd.concat([train[sel].reset_index(drop=True), pd.DataFrame(te_tr, columns=te_names)], axis=1)
    X_test = pd.concat([test[sel].reset_index(drop=True), pd.DataFrame(te_te, columns=te_names)], axis=1)
    ref = rmse(y, np.load(REF_LGB)["oof"])
    os.makedirs(OUT_DIR, exist_ok=True)

    def report(tag, oof, thresh_note=""):
        s = rmse(y, oof)
        d = ref - s
        verdict = "✅ 通过" if d > 0.0005 else "❌ 不足"
        if d > 0.003:
            verdict += "(⚠️ 超泄漏警报线 0.003,先审计再上线)"
        log(f"判据[{tag}]:OOF={s:.5f} vs 基线 {ref:.5f} → 改善 {d:+.5f} {verdict}{thresh_note}")

    if mode in ("lgb", "rest", "all"):
        td = pd.read_parquet(TD_CACHE)
        td_tr = train[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        td_te = test[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        # 与已有卡粒度首笔/末笔特征同名冲突检查(应为 0)
        dup = [c for c in td_tr.columns if c in X.columns]
        assert not dup, f"列名冲突:{dup}"
        Xa = pd.concat([X, td_tr.astype(np.float32)], axis=1)
        Xa_test = pd.concat([X_test, td_te.astype(np.float32)], axis=1)

    if mode in ("lgb", "all"):
        log(f"[A 线] X={Xa.shape}(sel {len(sel)} + TE {len(te_names)} + td {td_tr.shape[1]})")
        oof, pred, _, gain = ep.cv_lightgbm(Xa, y, Xa_test, folds, ep.LGB_PARAMS, "lgb+td")
        np.savez(os.path.join(OUT_DIR, "lgb.npz"), oof=oof, pred=pred)
        report("A 线 时间差分布", oof)
        g = pd.DataFrame({"feature": Xa.columns, "gain": gain}).sort_values("gain", ascending=False)
        td_cols = g[g["feature"].str.contains("_td_")]
        log("td 列 gain 前 8:\n" + td_cols.head(8).to_string(index=False))
        log(f"td 列进入 gain 前 50 的个数:{int(g.head(50)['feature'].str.contains('_td_').sum())}")

    if mode in ("weakreg", "all"):
        log(f"[B 线] 21st 弱正则参数,X={X.shape}(特征集与基线完全一致)")
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, WEAK_PARAMS, "lgb+weak")
        np.savez(os.path.join(OUT_DIR, "lgb_weakreg.npz"), oof=oof, pred=pred)
        report("B 线 弱正则参数", oof, "(纯参数改动,无泄漏可能;若大幅改善需怀疑过拟合 CV)")

    if mode in ("rest", "all"):
        # D 阶段:td 版全套基模型(镜像 target_encoding.py rest 模式),供 F 池融合
        def dump(name, oof, pred):
            np.savez(os.path.join(OUT_DIR, f"{name}.npz"), oof=oof, pred=pred)
            log(f"[td] {name:6s} OOF={rmse(y, oof):.5f} -> {OUT_DIR}/{name}.npz")

        oof, pred, _ = ep.cv_xgboost(Xa, y, Xa_test, folds);            dump("xgb", oof, pred)
        oof, pred, _ = ep.cv_catboost(Xa, y, Xa_test, folds);           dump("cat", oof, pred)
        oof, pred, _, _ = ep.cv_lightgbm(Xa, y, Xa_test, folds, ep.HUB_PARAMS, "hub+td"); dump("hub", oof, pred)
        oof, pred, auc = ep.cv_outlier_clf(Xa, y, Xa_test, folds)
        np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
        log(f"[td] clf AUC={auc:.5f}(TE 版 0.90414)")
        mask = (y < -30).to_numpy()
        oc, pc = np.zeros(len(Xa)), np.zeros(len(Xa_test))
        import lightgbm as lgb_
        for k, (tr, va) in enumerate(folds):
            tr_c, va_c = tr[~mask[tr]], va[~mask[va]]
            m = lgb_.train(ep.LGB_PARAMS, lgb_.Dataset(Xa.iloc[tr_c], y.iloc[tr_c]), 10000,
                           valid_sets=[lgb_.Dataset(Xa.iloc[va_c], y.iloc[va_c])],
                           callbacks=[lgb_.early_stopping(200, verbose=False)])
            oc[va] = m.predict(Xa.iloc[va], num_iteration=m.best_iteration)
            pc += m.predict(Xa_test, num_iteration=m.best_iteration) / len(folds)
            log(f"  [clean+td] fold{k + 1} iter={m.best_iteration}")
        dump("clean", oc, pc)


if __name__ == "__main__":
    main()
