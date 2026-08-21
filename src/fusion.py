# -*- coding: utf-8 -*-
"""v5 阶段 A-2:融合层重构 —— 元特征扩展 + 二层模型选型 + 期望值解析融合。

修正 v4 的结构缺陷:clean 模型游离在 stacking 之外,只通过手工单参数公式
    pred = (1-p^α)·clean + p^α·stack        (α 一维网格搜索)
进入最终预测。本脚本把 clean 与 outlier 概率一并交给二层,让二层自己学融合函数。

新增的两个想法:
  1) 概率校准:clf 用 is_unbalance=True 训练,输出并非真实后验概率;折内 isotonic
     回归校准后才能当概率用(未校准的 p 直接进公式是 v4 α 网格的隐含补偿项)。
  2) 期望值解析融合(ev):target 是混合分布 —— 以概率 p 取哨兵值 -33.219,否则
     落在正常区间。MSE 的最优预测即条件期望
         ev = p·(-33.219) + (1-p)·clean
     这是解析解,不需要网格搜索 α。把 ev 作为元特征交给二层。

所有方案在同一分层十折上折内 fit/predict 评估,与一层 OOF 口径一致,可直接横向比较。
用法:ELO_SEED=777 python src/fusion.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import elo_pipeline as ep
import paths

BASE_DIR = paths.out("base")
OUTLIER = ep.CONFIG["OUTLIER_TARGET"]
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def load_bases():
    """读取 outputs/base/*.npz(v4 特征集)与 outputs/base_ext/*.npz(扩展特征集)。

    扩展特征集的模型以 x_ 前缀区分。两套用的是同一 seed、同一 make_folds,行序一致,
    因此可以混进同一个二层 —— 特征集差异是另一种多样性来源。
    """
    bases = {}
    for d, pre in [(BASE_DIR, ""), (BASE_DIR + "_ext", "x_"), (BASE_DIR + "_te", "t_"),
                   (BASE_DIR + "_td", "d_"), (BASE_DIR + "_fm", "f_"), (BASE_DIR + "_nn", "n_"),
                   (BASE_DIR + "_dq", "q_"), (BASE_DIR + "_fma", "a_")]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                z = np.load(os.path.join(d, f))
                bases[pre + f[:-4]] = (z["oof"], z["pred"])
    return bases


def derive(name, tr, va, bases, ybin, p_src="clf", clean_src="clean"):
    """折内派生元特征:返回 (tr 侧, va 侧, test 侧) 三段列向量。

    p_src:outlier 概率的来源基模型(clf = 原 LGB 分类器,clf_ens = 多模型 rank 融合)。
    折内计算是必须的 —— isotonic 校准是有参数的拟合,若在全量 OOF 上 fit 会把
    验证折的标签信息漏进元特征,二层 RMSE 会虚低。
    """
    p_oof, p_test = bases[p_src]
    if name in ("p_cal", "ev", "p_cal_x_clean"):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_oof[tr], ybin[tr])
        p_tr, p_va, p_te = iso.predict(p_oof[tr]), iso.predict(p_oof[va]), iso.predict(p_test)
        if name == "p_cal":
            return p_tr, p_va, p_te
        c_oof, c_test = bases[clean_src]
        if name == "ev":  # 期望值解析融合
            return (p_tr * OUTLIER + (1 - p_tr) * c_oof[tr],
                    p_va * OUTLIER + (1 - p_va) * c_oof[va],
                    p_te * OUTLIER + (1 - p_te) * c_test)
        return p_tr * c_oof[tr], p_va * c_oof[va], p_te * c_test
    raise KeyError(name)


DERIVED = {"p_cal", "ev", "p_cal_x_clean"}
# 二层可选的少量原始强特征:让元模型学"什么样的卡该更信哪个基模型"(条件融合)。
# 只放 4 个业务含义最强、量纲稳定的列,维数远小于样本量,过拟合风险可控。
COND_COLS = ["hist_ref_month_id", "hist_count", "new_count", "elapsed_days"]


def fit_meta(model, Mtr, ytr, Mva, Mte):
    """二层模型:线性族用闭式解,LGB 用浅树,SLSQP 为非负凸组合(仅纯预测列可用)。"""
    if model == "ridge":
        m = Ridge(alpha=1.0, random_state=ep.CONFIG["SEED"])
    elif model == "ridge10":
        m = Ridge(alpha=10.0, random_state=ep.CONFIG["SEED"])
    elif model == "bayes":
        m = BayesianRidge()
    elif model == "lgb":
        import lightgbm as lgb
        d = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.02,
                           num_leaves=7, min_data_in_leaf=500, feature_fraction=0.9,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                           verbosity=-1, num_threads=ep.CONFIG["N_THREADS"],
                           seed=ep.CONFIG["SEED"]),
                      lgb.Dataset(Mtr, ytr), 400)
        return d.predict(Mva), d.predict(Mte)
    elif model == "xgb":  # 浅 XGB 头:与浅 LGB 同量级容量,分裂策略异构(E16/E17 三头用)
        import xgboost as xgb
        m = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.02, max_depth=3, min_child_weight=500,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=10.0, tree_method="hist",
            n_jobs=ep.CONFIG["N_THREADS"], random_state=ep.CONFIG["SEED"], verbosity=0)
        m.fit(Mtr, ytr)
        return m.predict(Mva), m.predict(Mte)
    elif model == "lgbbag":  # 浅 LGB 头 5-seed bagging:纯降方差,与 E5 双头同机制
        import lightgbm as lgb
        pvs, pts = [], []
        for s in (777, 1777, 2777, 3777, 4777):
            d = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.02,
                               num_leaves=7, min_data_in_leaf=500, feature_fraction=0.9,
                               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                               verbosity=-1, num_threads=ep.CONFIG["N_THREADS"],
                               seed=s, bagging_seed=s, feature_fraction_seed=s),
                          lgb.Dataset(Mtr, ytr), 400)
            pvs.append(d.predict(Mva))
            pts.append(d.predict(Mte))
        return np.mean(pvs, 0), np.mean(pts, 0)
    elif model == "nnls":  # 非负权重、和为 1
        def loss(w):
            return rmse(ytr, Mtr @ w)
        w0 = np.ones(Mtr.shape[1]) / Mtr.shape[1]
        r = minimize(loss, w0, method="SLSQP", bounds=[(0, 1)] * Mtr.shape[1],
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        return Mva @ r.x, Mte @ r.x
    else:
        raise KeyError(model)
    m.fit(Mtr, ytr)
    return m.predict(Mva), m.predict(Mte)


def evaluate(feats, model, bases, y, ybin, folds, cond=None, p_src="clf", clean_src="clean"):
    """按方案(元特征集合 × 二层模型)做折内 fit/predict,返回 OOF RMSE 与 test 预测。

    cond:可选的 (train_arr, test_arr) 原始特征对,附加到元特征做条件融合。
    p_src:派生特征所用的 outlier 概率来源。
    """
    n_tr, n_te = len(y), len(next(iter(bases.values()))[1])
    oof = np.zeros(n_tr)
    pred = np.zeros(n_te)
    for tr, va in folds:
        cols_tr, cols_va, cols_te = [], [], []
        for f in feats:
            if f in DERIVED:
                a, b, c = derive(f, tr, va, bases, ybin, p_src, clean_src)
            else:
                o, t = bases[f]
                a, b, c = o[tr], o[va], t
            cols_tr.append(a)
            cols_va.append(b)
            cols_te.append(c)
        Mtr, Mva, Mte = (np.column_stack(cols_tr), np.column_stack(cols_va),
                         np.column_stack(cols_te))
        if cond is not None:
            Ctr, Cte = cond
            Mtr = np.hstack([Mtr, Ctr[tr]])
            Mva = np.hstack([Mva, Ctr[va]])
            Mte = np.hstack([Mte, Cte])
        pv, pt = fit_meta(model, Mtr, y.to_numpy()[tr], Mva, Mte)
        oof[va] = pv
        pred += pt / len(folds)
    return rmse(y, oof), oof, pred


def alpha_blend_baseline(bases, y, folds, gbdt):
    """v4 现役方案复现:α 网格搜索的概率软融合(作为必须超越的基线)。"""
    rmse_st, oof_st, pred_st = evaluate(gbdt + ["clf"], "ridge", bases, y,
                                        (y < -30).astype(int).to_numpy(), folds)
    c_oof, c_test = bases["clean"]
    p_oof, p_test = bases["clf"]
    best = None
    for a in [0.5, 0.75, 1.0, 1.5, 2.0]:
        w = np.clip(p_oof, 0, 1) ** a
        r = rmse(y, (1 - w) * c_oof + w * oof_st)
        if best is None or r < best[1]:
            wt = np.clip(p_test, 0, 1) ** a
            best = (a, r, (1 - wt) * c_test + wt * pred_st)
    return rmse_st, best


def main():
    bases = load_bases()
    print(f"[fusion] 基模型池:{sorted(bases)}", flush=True)
    base_tbl = pd.read_parquet(paths.FEATURES)
    train = base_tbl[base_tbl["is_train"] == 1].reset_index(drop=True)
    test = base_tbl[base_tbl["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)

    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    GBDT = [k for k in ("lgb", "xgb", "cat", "hub") if k in bases]
    HET = [k for k in REG if k not in GBDT]
    print(f"[fusion] 回归基模型={REG}(异构={HET})", flush=True)
    if "hub" not in bases:
        print("[fusion] 警告:hub 尚未落盘,F0 不等于真正的 v4 基线,结果仅作初步参考", flush=True)

    # 方案矩阵:元特征集合 × 二层模型
    plans = [
        ("F0 v4-stack(基线)", GBDT + ["clf"], "ridge"),
        ("F1 +clean", GBDT + ["clf", "clean"], "ridge"),
        ("F2 +clean+p_cal", GBDT + ["clf", "clean", "p_cal"], "ridge"),
        ("F3 +ev(期望值解析)", GBDT + ["clf", "clean", "p_cal", "ev"], "ridge"),
        ("F4 F3+交互项", GBDT + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"], "ridge"),
        ("F5 F4 ridge10", GBDT + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"], "ridge10"),
        ("F6 F4 bayes", GBDT + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"], "bayes"),
        ("F7 F4 浅LGB", GBDT + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"], "lgb"),
        ("F8 ev+clean nnls", ["ev", "clean"] + GBDT, "nnls"),
    ]
    if len(REG) > len(GBDT):  # 异构模型已就位
        plans += [
            (f"F9 F4+异构{HET}", REG + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"], "ridge"),
            ("F10 F9 bayes", REG + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"], "bayes"),
        ]

    rows = []
    preds = {}
    for name, feats, model in plans:
        r, _, pt = evaluate(feats, model, bases, y, ybin, folds)
        rows.append({"plan": name, "n_meta": len(feats), "model": model, "oof": r})
        preds[name] = pt
        print(f"[fusion] {name:26s} n={len(feats):2d} {model:8s} OOF={r:.5f}", flush=True)

    # ---- 概率源对比:多模型 rank 融合的 outlier 概率(若已就位)----
    if "clf_ens" in bases:
        full = REG + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [("F13 全量+clf_ens概率", "ridge"), ("F14 全量+clf_ens bayes", "bayes")]:
            r, _, pt = evaluate(full, mdl, bases, y, ybin, folds, p_src="clf_ens")
            rows.append({"plan": tag, "n_meta": len(full), "model": mdl + "@ens", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(full):2d} {mdl + '@ens':8s} OOF={r:.5f}", flush=True)

    # ---- TE 版基模型(v6:折外 outlier 率目标编码)----
    TREG = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    if TREG and "t_clean" in bases and "t_clf" in bases:
        tfeats = TREG + ["t_clf", "t_clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [("F17 纯TE版", "ridge"), ("F18 纯TE版 bayes", "bayes")]:
            r, _, pt = evaluate(tfeats, mdl, bases, y, ybin, folds,
                                p_src="t_clf", clean_src="t_clean")
            rows.append({"plan": tag, "n_meta": len(tfeats), "model": mdl + "@te", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(tfeats):2d} {mdl + '@te':8s} OOF={r:.5f}", flush=True)
        # TE 版 + 原版 + 异构:最大融合池
        allf = REG + TREG + ["t_clf", "t_clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [("F19 全池(TE+原版+异构)", "ridge"), ("F20 全池 bayes", "bayes")]:
            r, _, pt = evaluate(allf, mdl, bases, y, ybin, folds,
                                p_src="t_clf", clean_src="t_clean")
            rows.append({"plan": tag, "n_meta": len(allf), "model": mdl + "@all", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(allf):2d} {mdl + '@all':8s} OOF={r:.5f}", flush=True)

    # ---- TD 版基模型(v7:TE + 交易粒度时间差分布特征)----
    DREG = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    if DREG and "d_clean" in bases and "d_clf" in bases:
        dfeats = DREG + ["d_clf", "d_clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [("F21 纯TD版 bayes", "bayes")]:
            r, _, pt = evaluate(dfeats, mdl, bases, y, ybin, folds,
                                p_src="d_clf", clean_src="d_clean")
            rows.append({"plan": tag, "n_meta": len(dfeats), "model": mdl + "@td", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(dfeats):2d} {mdl + '@td':8s} OOF={r:.5f}", flush=True)
        # TD + TE + 原版 + 异构:v7 最大融合池(派生特征用 td 版概率/clean,口径取最强一层)
        TREG2 = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
        allf2 = REG + TREG2 + DREG + (["t_clf", "t_clean"] if "t_clf" in bases else []) \
            + ["d_clf", "d_clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [("F22 全池(TD+TE+原版)", "ridge"), ("F23 全池TD bayes", "bayes")]:
            r, _, pt = evaluate(allf2, mdl, bases, y, ybin, folds,
                                p_src="d_clf", clean_src="d_clean")
            rows.append({"plan": tag, "n_meta": len(allf2), "model": mdl + "@td+", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(allf2):2d} {mdl + '@td+':8s} OOF={r:.5f}", flush=True)

    # ---- FM 版基模型(v11:TD + target 公式形状特征)----
    FREG = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    if FREG and "f_clean" in bases and "f_clf" in bases:
        TREG3 = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
        DREG3 = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
        allf3 = (REG + TREG3 + DREG3 + FREG
                 + (["t_clf", "t_clean"] if "t_clf" in bases else [])
                 + (["d_clf", "d_clean"] if "d_clf" in bases else [])
                 + ["f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"])
        for tag, mdl in [("F28 全池(FM+TD+TE+原版)", "ridge"), ("F29 全池FM bayes", "bayes")]:
            r, _, pt = evaluate(allf3, mdl, bases, y, ybin, folds,
                                p_src="f_clf", clean_src="f_clean")
            rows.append({"plan": tag, "n_meta": len(allf3), "model": mdl + "@fm+", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(allf3):2d} {mdl + '@fm+':8s} OOF={r:.5f}", flush=True)

        # ---- v13/v14:+ 序列 NN 族(异构互补,融合增益判据在此)----
        NREG = sorted(k for k in bases if k.startswith("n_"))
        if NREG:
            allf4 = allf3 + NREG
            for tag, mdl in [("F30 全池+NN族", "ridge"), ("F31 全池+NN族 bayes", "bayes")]:
                r, _, pt = evaluate(allf4, mdl, bases, y, ybin, folds,
                                    p_src="f_clf", clean_src="f_clean")
                rows.append({"plan": tag, "n_meta": len(allf4), "model": mdl + "@nn", "oof": r})
                preds[tag] = pt
                print(f"[fusion] {tag:26s} n={len(allf4):2d} {mdl + '@nn':8s} OOF={r:.5f}"
                      f"  NN成员={NREG}", flush=True)

            # ---- v15:outlier 概率 NN 化(f_clf 与序列 NN 概率 rank 平均为新 p 源)----
            nnc = paths.out("base_nn_clf", "clf.npz")
            if os.path.exists(nnc):
                from scipy.stats import rankdata
                zc = np.load(nnc)
                qr = lambda a: rankdata(a) / len(a)
                bases["pens"] = ((qr(bases["f_clf"][0]) + qr(zc["oof"])) / 2,
                                 (qr(bases["f_clf"][1]) + qr(zc["pred"])) / 2)
                for tag, mdl in [("F32 全池NN p_ens", "ridge"), ("F33 全池NN p_ens bayes", "bayes")]:
                    r, _, pt = evaluate(allf4, mdl, bases, y, ybin, folds,
                                        p_src="pens", clean_src="f_clean")
                    rows.append({"plan": tag, "n_meta": len(allf4), "model": mdl + "@pens", "oof": r})
                    preds[tag] = pt
                    print(f"[fusion] {tag:26s} n={len(allf4):2d} {mdl + '@pens':8s} OOF={r:.5f}", flush=True)

            # ---- v16:dq 版基模型(dec 时间结构 + merchants 增量)----
            QREG = [k for k in ("q_lgb", "q_xgb", "q_cat", "q_hub") if k in bases]
            if QREG and "q_clean" in bases and "q_clf" in bases:
                allf6 = allf4 + QREG + ["q_clf", "q_clean"]
                for tag, mdl in [("F36 全池+DQ", "ridge"), ("F37 全池+DQ bayes", "bayes")]:
                    r, _, pt = evaluate(allf6, mdl, bases, y, ybin, folds,
                                        p_src="q_clf", clean_src="q_clean")
                    rows.append({"plan": tag, "n_meta": len(allf6), "model": mdl + "@dq", "oof": r})
                    preds[tag] = pt
                    print(f"[fusion] {tag:26s} n={len(allf6):2d} {mdl + '@dq':8s} OOF={r:.5f}", flush=True)

            # ---- v20:fm 套三 seed 平均替换(f_* → a_*,同池身份降噪版)----
            AREG = [k for k in ("a_lgb", "a_xgb", "a_cat", "a_hub") if k in bases]
            if AREG and "a_clean" in bases and "a_clf" in bases:
                allf7 = [k for k in allf4 if not k.startswith("f_")] + AREG + ["a_clf", "a_clean"]
                for tag, mdl in [("F38 全池seedavg", "ridge"), ("F39 全池seedavg bayes", "bayes")]:
                    r, _, pt = evaluate(allf7, mdl, bases, y, ybin, folds,
                                        p_src="a_clf", clean_src="a_clean")
                    rows.append({"plan": tag, "n_meta": len(allf7), "model": mdl + "@sa", "oof": r})
                    preds[tag] = pt
                    print(f"[fusion] {tag:26s} n={len(allf7):2d} {mdl + '@sa':8s} OOF={r:.5f}", flush=True)

    # ---- 跨特征集融合:扩展特征训练的模型即使单模更差,误差方向仍可能互补 ----
    XREG = [k for k in bases if k.startswith("x_") and not k.startswith("x_clf")]
    if XREG:
        full = REG + sorted(XREG) + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [(f"F15 跨特征集({len(XREG)}个x_)", "ridge"), ("F16 跨特征集 bayes", "bayes")]:
            r, _, pt = evaluate(full, mdl, bases, y, ybin, folds)
            rows.append({"plan": tag, "n_meta": len(full), "model": mdl + "+x", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(full):2d} {mdl + '+x':8s} OOF={r:.5f}", flush=True)
        print(f"[fusion] 跨特征集成员={sorted(XREG)}", flush=True)

    # ---- 条件融合:元特征 + 少量原始强特征,让二层学"该信谁" ----
    cond_ok = [c for c in COND_COLS if c in train.columns]
    if cond_ok:
        Ctr = train[cond_ok].to_numpy(np.float32)
        Cte = test[cond_ok].to_numpy(np.float32)
        Ctr = np.nan_to_num(Ctr, nan=-999.0)
        Cte = np.nan_to_num(Cte, nan=-999.0)
        base_feats = REG + ["clf", "clean", "p_cal", "ev", "p_cal_x_clean"]
        for tag, mdl in [("F11 条件融合(浅LGB)", "lgb"), ("F12 条件融合(ridge)", "ridge")]:
            r, _, pt = evaluate(base_feats, mdl, bases, y, ybin, folds, cond=(Ctr, Cte))
            rows.append({"plan": tag, "n_meta": len(base_feats) + len(cond_ok),
                         "model": mdl + "+cond", "oof": r})
            preds[tag] = pt
            print(f"[fusion] {tag:26s} n={len(base_feats) + len(cond_ok):2d} "
                  f"{mdl + '+cond':8s} OOF={r:.5f}", flush=True)
        print(f"[fusion] 条件列={cond_ok}", flush=True)

    # v4 现役方案(α 网格软融合)作为参照
    rmse_st, (a_best, r_alpha, pred_alpha) = alpha_blend_baseline(bases, y, folds, GBDT)
    print(f"[fusion] --- 参照:v4 α={a_best} 概率软融合 OOF={r_alpha:.5f}"
          f"(纯 stack {rmse_st:.5f})", flush=True)

    tbl = pd.DataFrame(rows).sort_values("oof").reset_index(drop=True)
    print("\n" + tbl.to_string(index=False), flush=True)
    best = tbl.iloc[0]
    gain = r_alpha - best["oof"]
    print(f"\n[fusion] 最优 {best['plan']} OOF={best['oof']:.5f} "
          f"vs v4 现役 {r_alpha:.5f} → 改善 {gain:+.5f}", flush=True)
    print(f"[fusion] 按 -0.031 偏移推算 Private ≈ {best['oof'] - 0.031:.5f}", flush=True)

    out = os.environ.get("ELO_FUSION_OUT", paths.out("submission_v5_fusion.csv"))
    pd.DataFrame({"card_id": test["card_id"], "target": preds[best["plan"]]}).to_csv(out, index=False)
    desc = f"v5 fusion [{best['plan']}] {best['model']}, OOF {best['oof']:.5f}"
    open(paths.out("v5_fusion_desc.txt"), "w").write(desc)
    json.dump({"table": rows, "alpha_baseline": {"alpha": a_best, "oof": r_alpha},
               "pure_stack_oof": rmse_st, "best": best.to_dict()},
              open(paths.out("v5_fusion_report.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[fusion] 保存 {out}({desc})", flush=True)


if __name__ == "__main__":
    main()
