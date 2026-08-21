# -*- coding: utf-8 -*-
"""v15 融合层优化实验:在 F31(31 元特征 + BayesianRidge)基础上做低成本扩展。

不改动最小复现链路,仅复用 fusion 的 evaluate/derive 协议做对照实验:
  E0  F31 基线(与 fuse_final.py 完全一致,作为必须超越的参照)
  E1  p_src 跨世代 rank 集成:clf/t_clf/d_clf/f_clf 四个 outlier 头 rank 平均后作为
      派生特征(p_cal/ev/交互)的概率源 —— 概率头之间相关但角度不同,rank 平均降方差
  E2  多世代派生列:在 E0 之上追加 (d_clf,d_clean) 与 (t_clf,t_clean) 的 ev 列
      —— 期望值解析融合的"多口径"版本,给二层更多 outlier 通道信息
  E3  E1 + E2 合并
  E4  条件融合:E0 + COND_COLS 4 个原始强特征(bayes 线性可学"该信谁"的一阶近似)
  E5  二层模型对照:E0 元特征 + 浅 LGB 与 bayes 的 OOF 均值(异构二层降方差)
判据:OOF RMSE 低于 E0 才算有效。用法:ELO_SEED=777 python src/fuse_opt.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import elo_pipeline as ep
import fusion as vf

rmse = vf.rmse
import paths

NNCLF_PARTS_DIR = paths.out("nn_clf_parts")


def derive_multi(tr, va, bases, ybin, p_src, clean_src, kind):
    """与 fusion.derive 同协议的多口径派生列(折内 isotonic,防标签泄漏)。"""
    from sklearn.isotonic import IsotonicRegression
    p_oof, p_test = bases[p_src]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_oof[tr], ybin[tr])
    p_tr, p_va, p_te = iso.predict(p_oof[tr]), iso.predict(p_oof[va]), iso.predict(p_test)
    if kind == "p_cal":
        return p_tr, p_va, p_te
    c_oof, c_test = bases[clean_src]
    OUT = vf.OUTLIER
    return (p_tr * OUT + (1 - p_tr) * c_oof[tr],
            p_va * OUT + (1 - p_va) * c_oof[va],
            p_te * OUT + (1 - p_te) * c_test)


def evaluate_ext(feats, model, bases, y, ybin, folds, p_src, clean_src,
                 extra_derived=(), cond=None):
    """fusion.evaluate 的扩展版:支持追加多口径派生列与条件列。

    extra_derived:[(p_src, clean_src, kind), ...],每折折内计算。
    """
    n_te = len(next(iter(bases.values()))[1])
    oof, pred = np.zeros(len(y)), np.zeros(n_te)
    yv = y.to_numpy()
    for tr, va in folds:
        cols_tr, cols_va, cols_te = [], [], []
        for f in feats:
            if f in vf.DERIVED:
                a, b, c = vf.derive(f, tr, va, bases, ybin, p_src, clean_src)
            else:
                o, t = bases[f]
                a, b, c = o[tr], o[va], t
            cols_tr.append(a); cols_va.append(b); cols_te.append(c)
        for ps, cs, kind in extra_derived:
            a, b, c = derive_multi(tr, va, bases, ybin, ps, cs, kind)
            cols_tr.append(a); cols_va.append(b); cols_te.append(c)
        Mtr = np.column_stack(cols_tr)
        Mva = np.column_stack(cols_va)
        Mte = np.column_stack(cols_te)
        if cond is not None:
            Ctr, Cte = cond
            Mtr = np.hstack([Mtr, Ctr[tr]])
            Mva = np.hstack([Mva, Ctr[va]])
            Mte = np.hstack([Mte, Cte])
        if model == "bayes+lgb":
            pv1, pt1 = vf.fit_meta("bayes", Mtr, yv[tr], Mva, Mte)
            pv2, pt2 = vf.fit_meta("lgb", Mtr, yv[tr], Mva, Mte)
            pv, pt = (pv1 + pv2) / 2, (pt1 + pt2) / 2
        else:
            pv, pt = vf.fit_meta(model, Mtr, yv[tr], Mva, Mte)
        oof[va] = pv
        pred += pt / len(folds)
    return rmse(y, oof), oof, pred


NNCLF_SEED_PREFIX = "clf_s"   # nn_clf.py 的 seed 分片命名:clf_s<seed>.npz


def load_avg_parts(parts_dir, prefix=NNCLF_SEED_PREFIX):
    """把同协议 seed 分片平均成一个原始弱特征。

    只收 `prefix` 开头的分片:nn_clf_parts/ 历史上还落过别的实验产物
    (如自监督缩放实验的 ssl_scale_clf.npz),若一并平均进来,z_nnc 会静默改口径,
    SC5/U2/F1 全线跟着漂 —— 曾实测把 SC5 OOF 从 3.61996 推到 3.62003。
    """
    if not os.path.isdir(parts_dir):
        return [], None
    files = sorted(f for f in os.listdir(parts_dir)
                   if f.endswith(".npz") and f.startswith(prefix))
    if not files:
        return [], None
    zs = [np.load(os.path.join(parts_dir, f)) for f in files]
    return files, (np.mean([z["oof"] for z in zs], 0),
                   np.mean([z["pred"] for z in zs], 0))


def evaluate_stratified_blend(feats, bases, y, ybin, folds, p_src, clean_src,
                              extra_derived=(), cond=None):
    """分层二层:正常用户单独训练回归器,用现有概率融合outlier预测。

    核心思路: 只优化正常用户侧(98.91%),outlier侧保持原ev公式。
    """
    n_te = len(next(iter(bases.values()))[1])
    oof, pred = np.zeros(len(y)), np.zeros(n_te)
    yv = y.to_numpy()
    for tr, va in folds:
        cols_tr, cols_va, cols_te = [], [], []
        for f in feats:
            if f in vf.DERIVED:
                a, b, c = derive_multi(tr, va, bases, ybin, p_src, clean_src, f.split("_")[-1])
            else:
                o, t = bases[f]
                a, b, c = o[tr], o[va], t
            cols_tr.append(a); cols_va.append(b); cols_te.append(c)
        for ps, cs, kind in extra_derived:
            a, b, c = derive_multi(tr, va, bases, ybin, ps, cs, kind)
            cols_tr.append(a); cols_va.append(b); cols_te.append(c)
        Mtr, Mva, Mte = np.column_stack(cols_tr), np.column_stack(cols_va), np.column_stack(cols_te)
        if cond is not None:
            Ctr, Cte = cond
            Mtr = np.hstack([Mtr, Ctr[tr]])
            Mva = np.hstack([Mva, Ctr[va]])
            Mte = np.hstack([Mte, Cte])

        # 只用正常用户训练回归器
        mask_norm_tr = ybin[tr] == 0
        reg_model = vf.BayesianRidge(max_iter=300, tol=1e-3)
        reg_model.fit(Mtr[mask_norm_tr], yv[tr][mask_norm_tr])

        # 预测正常侧
        pred_norm_va = reg_model.predict(Mva)
        pred_norm_te = reg_model.predict(Mte)

        # 获取概率并融合
        from sklearn.isotonic import IsotonicRegression
        p_oof, p_test = bases[p_src]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_oof[tr], ybin[tr])
        p_va = iso.predict(p_oof[va])
        p_te = iso.predict(p_test)

        # 融合: p * outlier_target + (1-p) * normal_pred
        oof[va] = p_va * vf.OUTLIER + (1 - p_va) * pred_norm_va
        pred += (p_te * vf.OUTLIER + (1 - p_te) * pred_norm_te) / len(folds)

    return rmse(y, oof), oof, pred


def evaluate_blend(feats, bases, y, ybin, folds, p_src, clean_src,
                   extra_derived=(), cond=None, bayes_w=0.5, n_models=2):
    """多模型二层融合; bayes_w=None 时解析求最优权重。

    n_models=2: bayes + lgb
    n_models=3: bayes + lgb + xgb
    """
    _, o_b, p_b = evaluate_ext(feats, "bayes", bases, y, ybin, folds,
                               p_src, clean_src, extra_derived=extra_derived, cond=cond)
    _, o_l, p_l = evaluate_ext(feats, "lgb", bases, y, ybin, folds,
                               p_src, clean_src, extra_derived=extra_derived, cond=cond)

    if n_models == 2:
        if bayes_w is None:
            yv = y.to_numpy()
            d = o_b - o_l
            den = float(np.dot(d, d))
            bayes_w = 0.5 if den < 1e-12 else float(np.clip(np.dot(yv - o_l, d) / den, 0.0, 1.0))
        oof = bayes_w * o_b + (1 - bayes_w) * o_l
        pred = bayes_w * p_b + (1 - bayes_w) * p_l
        return rmse(y, oof), oof, pred, bayes_w

    elif n_models == 3:
        _, o_x, p_x = evaluate_ext(feats, "xgb", bases, y, ybin, folds,
                                   p_src, clean_src, extra_derived=extra_derived, cond=cond)
        # 三模型解析权重
        from scipy.optimize import minimize
        yv = y.to_numpy()

        def obj(w):
            oof_blend = w[0] * o_b + w[1] * o_l + w[2] * o_x
            return rmse(y, oof_blend)

        res = minimize(obj, [0.33, 0.33, 0.34],
                      bounds=[(0, 1), (0, 1), (0, 1)],
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        w_opt = res.x
        oof = w_opt[0] * o_b + w_opt[1] * o_l + w_opt[2] * o_x
        pred = w_opt[0] * p_b + w_opt[1] * p_l + w_opt[2] * p_x
        return rmse(y, oof), oof, pred, w_opt


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    bases = vf.load_bases()
    base = pd.read_parquet(paths.FEATURES)
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)

    REG = ["lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et"]
    allf = (REG + ["t_lgb", "t_xgb", "t_cat", "t_hub"]
            + ["d_lgb", "d_xgb", "d_cat", "d_hub"]
            + ["f_lgb", "f_xgb", "f_cat", "f_hub"]
            + ["t_clf", "t_clean", "d_clf", "d_clean", "f_clf", "f_clean",
               "p_cal", "ev", "p_cal_x_clean"]
            + ["n_gru", "n_gru_x", "n_trf"])
    missing = [k for k in allf if k not in bases and k not in vf.DERIVED]
    assert not missing, f"缺基模型产物:{missing}"

    # E1 概率源:四个 outlier 头 rank 平均
    qr = lambda a: rankdata(a) / len(a)
    heads = [k for k in ("clf", "t_clf", "d_clf", "f_clf") if k in bases]
    bases["p_ens4"] = (np.mean([qr(bases[k][0]) for k in heads], 0),
                       np.mean([qr(bases[k][1]) for k in heads], 0))
    print(f"[opt] p_ens4 成员={heads}", flush=True)
    nn_members, nn_raw = load_avg_parts(NNCLF_PARTS_DIR)
    if nn_raw is not None:
        bases["z_nnc"] = nn_raw
        print(f"[opt] z_nnc 成员={nn_members}", flush=True)

    # 交互特征: z_nnc × q_clf
    if "z_nnc" in bases and "q_clf" in bases:
        z_oof, z_test = bases["z_nnc"]
        q_oof, q_test = bases["q_clf"]
        bases["z_q_interact"] = (z_oof * q_oof, z_test * q_test)
        print(f"[opt] z_q_interact 已创建 (z_nnc × q_clf)", flush=True)

    cond_ok = [c for c in vf.COND_COLS if c in train.columns]
    Ctr = np.nan_to_num(train[cond_ok].to_numpy(np.float32), nan=-999.0)
    Cte = np.nan_to_num(test[cond_ok].to_numpy(np.float32), nan=-999.0)

    def run_exp(name, model="bayes", feats=None, p_src="f_clf", clean_src="f_clean",
                extra_derived=(), cond=None, bayes_w=0.5, n_models=2):
        feats = allf if feats is None else feats
        if model == "blend":
            r, _, pt, wt = evaluate_blend(feats, bases, y, ybin, folds, p_src, clean_src,
                                          extra_derived=extra_derived, cond=cond,
                                          bayes_w=bayes_w, n_models=n_models)
            if n_models == 2:
                model_tag = f"bayes*{wt:.3f}+lgb*{1 - wt:.3f}"
            elif n_models == 3:
                model_tag = f"bayes*{wt[0]:.3f}+lgb*{wt[1]:.3f}+xgb*{wt[2]:.3f}"
        elif model == "stratified":
            r, _, pt = evaluate_stratified_blend(feats, bases, y, ybin, folds, p_src, clean_src,
                                                 extra_derived=extra_derived, cond=cond)
            model_tag = "stratified(clf+reg)"
        else:
            r, _, pt = evaluate_ext(feats, model, bases, y, ybin, folds, p_src, clean_src,
                                    extra_derived=extra_derived, cond=cond)
            model_tag = model
        rows.append({"exp": name, "model": model_tag, "oof": r})
        preds[name] = pt
        print(f"[opt] {name:25s} {model_tag:30s} OOF={r:.5f}", flush=True)

    rows, preds = [], {}
    run_exp("E0 F31基线")
    run_exp("E1 p_src=rank集成", p_src="p_ens4")
    run_exp("E2 +多世代ev", extra_derived=[("d_clf", "d_clean", "ev"),
                                         ("t_clf", "t_clean", "ev")])
    run_exp("E3 E1+E2", p_src="p_ens4",
            extra_derived=[("d_clf", "d_clean", "ev"),
                           ("t_clf", "t_clean", "ev")])
    run_exp("E4 +条件列", cond=(Ctr, Cte))
    run_exp("E5 bayes+浅LGB等权", model="blend", bayes_w=0.5)
    if cond_ok:
        run_exp("E6 E5+条件列", model="blend", cond=(Ctr, Cte), bayes_w=0.5)
    if "z_nnc" in bases:
        run_exp("E7 +z_nnc raw", model="blend", feats=allf + ["z_nnc"], bayes_w=0.5)
    if "q_clf" in bases:
        run_exp("E8 +q_clf raw", model="blend", feats=allf + ["q_clf"], bayes_w=0.5)
    if "z_nnc" in bases and "q_clf" in bases:
        combo = allf + ["z_nnc", "q_clf"]
        run_exp("E9 +z_nnc+q_clf", model="blend", feats=combo, bayes_w=0.5)
        run_exp("E10 E9解析权重", model="blend", feats=combo, bayes_w=None)

    # 新增:测试完整DQ成员池
    dq_avail = [m for m in ["q_lgb", "q_hub", "q_clean"] if m in bases]
    if "z_nnc" in bases and dq_avail:
        combo_full = allf + ["z_nnc"] + dq_avail
        run_exp("E11 +z_nnc+DQ全部", model="blend", feats=combo_full, bayes_w=0.5)
        run_exp("E12 E11解析权重", model="blend", feats=combo_full, bayes_w=None)

    # 新增P0优化: 分层二层
    if "z_nnc" in bases and "q_clf" in bases:
        combo = allf + ["z_nnc", "q_clf"]
        run_exp("E13 E9+分层二层", model="stratified", feats=combo)

    # 新增P0优化: 交互项
    if "z_q_interact" in bases:
        combo_interact = allf + ["z_nnc", "q_clf", "z_q_interact"]
        run_exp("E14 E9+交互项", model="blend", feats=combo_interact, bayes_w=0.5)
        run_exp("E15 E14解析权重", model="blend", feats=combo_interact, bayes_w=None)

    # 新增P0优化: 三模型融合
    if "z_nnc" in bases and "q_clf" in bases:
        combo = allf + ["z_nnc", "q_clf"]
        run_exp("E16 E9+三模型融合", model="blend", feats=combo, bayes_w=None, n_models=3)

    # 新增: 交互项+三模型组合
    if "z_q_interact" in bases:
        combo_full = allf + ["z_nnc", "q_clf", "z_q_interact"]
        run_exp("E17 交互+三模型", model="blend", feats=combo_full, bayes_w=None, n_models=3)

    tbl = pd.DataFrame(rows).sort_values("oof").reset_index(drop=True)
    print("\n" + tbl.to_string(index=False), flush=True)
    base_oof = [r["oof"] for r in rows if r["exp"].startswith("E0")][0]
    best = tbl.iloc[0]
    print(f"\n[opt] 最优 {best['exp']} OOF={best['oof']:.5f} vs 基线 {base_oof:.5f} "
          f"→ {best['oof'] - base_oof:+.5f}", flush=True)
    if best["oof"] < base_oof:
        out = paths.out("submission_v15_opt.csv")
        pd.DataFrame({"card_id": test["card_id"], "target": preds[best["exp"]]}
                     ).to_csv(out, index=False)
        print(f"[opt] 保存 {out}", flush=True)


if __name__ == "__main__":
    main()
