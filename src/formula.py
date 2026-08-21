# -*- coding: utf-8 -*-
"""v11:target 公式形状特征(log2 比值族,ε=1e-10 与哨兵机制同构)。

依据(docs/202607/31-v11_target逆向.md):
  target = log2(x + 1e-10),x 为"参考月后 2 月窗口行为量 / 历史基线"类比值:
  · 格点实证:重复值最多的 target 全部命中 2^t ∈ {1,2,1/2,2/3,3/2,2/5,3,...} 简单分数;
  · 哨兵 -33.21928 = log2(1e-10),即 x=0(评估窗内无目标行为);
  · 官方口径(#72993):loyalty = future spending + retention。
x 的精确分子不可观测(评估窗的全量/回头交易未给出,只给了 new 商户交易),
但可观测代理(new 窗金额/笔数/商户数 ÷ 历史月均基线)按公式形状 log2(r+1e-10) 特征化,
让树模型在 target 的原生标度上直接工作。全部由 features.parquet 现有列派生,折外无依赖。

判据:单模 lgb(sel+TE+td+fm)OOF vs outputs/base_td/lgb.npz(777,3.63316)改善>0.0005。
用法:ELO_SEED=777 python src/formula.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

import paths

OUT_DIR = paths.out("base_fm")
REF_LGB = paths.out("base_td", "lgb.npz")
HL_CACHE = paths.out("hist_lag_amt.parquet")   # 历史逐月金额(强化分母用)
EPS = 1e-10
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def hist_lag_amt() -> pd.DataFrame:
    """历史(auth=1)逐月金额透视:近 3 月和 0.9^|lag| 衰减加权月均两种基线。"""
    if os.path.exists(HL_CACHE):
        return pd.read_parquet(HL_CACHE)
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1]
    g = hist.groupby(["card_id", "month_lag"], observed=True)["purchase_amount"].sum().unstack()
    g = g.reindex(columns=range(-13, 1)).fillna(0.0)
    r3 = g[[c for c in (0, -1, -2) if c in g.columns]].sum(1) / 3
    w = np.array([0.9 ** abs(c) for c in g.columns])
    wavg = (g.to_numpy() * w).sum(1) / w.sum()
    res = pd.DataFrame({"hl_r3_amt": r3, "hl_wavg_amt": wavg})
    res.index = res.index.astype(str)
    res = res.reset_index().rename(columns={"index": "card_id"})
    res.to_parquet(HL_CACHE)
    log(f"历史逐月金额基线缓存 {HL_CACHE}: {res.shape}")
    return res


def formula_block(df: pd.DataFrame) -> pd.DataFrame:
    """log2((new 窗行为量 / 历史基线) + 1e-10):无 new 活动 → -33.2,与哨兵同构。"""
    g = lambda c: df[c].fillna(0).to_numpy(np.float64) if c in df else np.zeros(len(df))
    months = np.clip(df["hist_month_lag_max"].fillna(0).to_numpy(np.float64)
                     - df["hist_month_lag_min"].fillna(-12).to_numpy(np.float64) + 1, 1, None)
    h_amt_m = g("hist_purchase_amount_sum") / months          # 历史月均消费
    h_cnt_m = np.clip(g("hist_count") / months, 1e-9, None)   # 历史月均笔数
    h_mer_m = np.clip(g("hist_merchant_id_nunique") / months, 1e-9, None)
    r3_cnt_m = np.clip(g("hist_recent3_count") / 3, 1e-9, None)
    lg = lambda r: np.log2(np.clip(r, 0, None) + EPS).astype(np.float32)
    div = lambda a, b: np.where(b > 1e-9, a / np.where(b > 1e-9, b, 1), 0.0)

    f = pd.DataFrame(index=df.index)
    # 主假设族:new 2 月窗 vs 历史月均
    f["fm_amt"] = lg(div(g("new_purchase_amount_sum") / 2, h_amt_m))
    f["fm_amt_l1"] = lg(div(g("newlag1_purchase_amount_sum"), h_amt_m))
    f["fm_amt_l2"] = lg(div(g("newlag2_purchase_amount_sum"), h_amt_m))
    f["fm_amt_trend"] = lg(div(g("newlag2_purchase_amount_sum"), np.clip(g("newlag1_purchase_amount_sum"), 1e-9, None)))
    # 笔数(retention 频次)与商户广度(retention 宽度)
    f["fm_cnt"] = lg(div(g("new_count") / 2, h_cnt_m))
    f["fm_cnt_l1"] = lg(div(g("newlag1_purchase_amount_count"), h_cnt_m))
    f["fm_cnt_l2"] = lg(div(g("newlag2_purchase_amount_count"), h_cnt_m))
    f["fm_mer"] = lg(div(g("new_merchant_id_nunique") / 2, h_mer_m))
    f["fm_mer_l1"] = lg(div(g("newlag1_merchant_id_nunique"), h_mer_m))
    # 近端基线版(历史近 3 月)
    f["fm_cnt_r3"] = lg(div(g("new_count") / 2, r3_cnt_m))
    # 单笔强度
    f["fm_ticket"] = lg(div(g("new_purchase_amount_mean"), np.clip(g("hist_purchase_amount_mean"), 1e-9, None)))
    # 组合:金额 × 笔数几何中项(spending+retention 双组件的最简合成)
    f["fm_combo"] = (f["fm_amt"] + f["fm_cnt"]) / 2
    # v2 强化:更好的分母 —— 历史近 3 月金额基线与 0.9^|lag| 衰减加权基线
    if "hl_r3_amt" in df:
        r3a = np.clip(df["hl_r3_amt"].fillna(0).to_numpy(np.float64), 0, None)
        wva = np.clip(df["hl_wavg_amt"].fillna(0).to_numpy(np.float64), 0, None)
        f["fm_amt_r3"] = lg(div(g("new_purchase_amount_sum") / 2, np.clip(r3a, 1e-9, None)))
        f["fm_amt_w"] = lg(div(g("new_purchase_amount_sum") / 2, np.clip(wva, 1e-9, None)))
        f["fm_amt_l1_r3"] = lg(div(g("newlag1_purchase_amount_sum"), np.clip(r3a, 1e-9, None)))
        f["fm_base_shift"] = lg(div(np.clip(r3a, 1e-9, None), np.clip(h_amt_m, 1e-9, None)))  # 基线自身的近远期漂移
    return f


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1] if len(sys.argv) > 1 else "lgb"
    base = pd.read_parquet(paths.FEATURES)
    hl = hist_lag_amt()
    base = base.merge(hl, on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = formula_block(train), formula_block(test)
    ok = (y > -30).to_numpy()
    log("公式特征与 target 的 spearman(全量 | 仅非 outlier):")
    for c in fm_tr.columns:
        s_all = spearmanr(fm_tr[c], y).statistic
        s_ok = spearmanr(fm_tr[c][ok], y[ok]).statistic
        log(f"  {c:14s} {s_all:+.4f} | {s_ok:+.4f}")
    imp = pd.read_csv(paths.FEATURE_IMPORTANCE)
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load(paths.out("te_features_v1.npz"), allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet(paths.out("td_features.parquet"))

    def assemble(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True)], axis=1)

    X = assemble(train, z["tr"], fm_tr)
    X_test = assemble(test, z["te"], fm_te)
    log(f"X={X.shape}(sel {len(sel)} + TE {len(te_names)} + td {td.shape[1] - 1} + fm {fm_tr.shape[1]})")
    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, oof, pred):
        np.savez(os.path.join(OUT_DIR, f"{name}.npz"), oof=oof, pred=pred)
        log(f"[fm] {name:6s} OOF={rmse(y, oof):.5f} -> {OUT_DIR}/{name}.npz")

    if mode in ("lgb", "all"):
        oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+fm")
        dump("lgb", oof, pred)
        ref = rmse(y, np.load(REF_LGB)["oof"])
        s = rmse(y, oof)
        d = ref - s
        alarm = "(⚠️ 超警报线,先审计)" if d > 0.003 else ""
        log(f"判据[v11 公式特征]:OOF={s:.5f} vs 基线 {ref:.5f} → 改善 {d:+.5f} "
            f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}{alarm}")
        g = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
        log("fm 列 gain:\n" + g[g["feature"].str.startswith("fm_")].to_string(index=False))
        log(f"fm 列进入 gain 前 50 的个数:{int(g.head(50)['feature'].str.startswith('fm_').sum())}")

    if mode in ("rest", "all"):
        oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds);            dump("xgb", oof, pred)
        oof, pred, _ = ep.cv_catboost(X, y, X_test, folds);           dump("cat", oof, pred)
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "hub+fm"); dump("hub", oof, pred)
        oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
        np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
        log(f"[fm] clf AUC={auc:.5f}(td 版 0.90507)")
        mask = (y < -30).to_numpy()
        oc, pc = np.zeros(len(X)), np.zeros(len(X_test))
        import lightgbm as lgb_
        for k, (tr, va) in enumerate(folds):
            tr_c, va_c = tr[~mask[tr]], va[~mask[va]]
            m = lgb_.train(ep.LGB_PARAMS, lgb_.Dataset(X.iloc[tr_c], y.iloc[tr_c]), 10000,
                           valid_sets=[lgb_.Dataset(X.iloc[va_c], y.iloc[va_c])],
                           callbacks=[lgb_.early_stopping(200, verbose=False)])
            oc[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            pc += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
            log(f"  [clean+fm] fold{k + 1} iter={m.best_iteration}")
        dump("clean", oc, pc)


if __name__ == "__main__":
    main()
