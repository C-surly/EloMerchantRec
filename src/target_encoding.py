# -*- coding: utf-8 -*-
"""v6-P1:折外 outlier 率目标编码(target encoding),移植自 21st 方案并收紧泄漏控制。

思路来源:refs/ELO/DY/best_single_model.ipynb 的 `get_data_ctr_fea` —— 对卡片属性与
month_diff 统计量,按 CTR 风格编码**outlier 二值标签**(而非 target 本身)的发生率:
    ctr = 1000 * (sum(outliers) + 0.01) / (count + 0.01)
一阶 + 全部两两交叉。编码 outlier 率等于给树模型一个低维的「这类卡有多容易是哨兵值」先验。

**与原方案的关键差异(泄漏控制)**:21st 用一套独立的 5 折(random_state=4590)算 TE,
再用另一套折训练模型 —— TE 的折外信息会漏进模型的验证折,其 CV 3.6348 存在乐观偏差。
本实现直接在**模型自己的十折训练折内**统计,验证折从未参与过自己的编码;test 侧取
十折统计的平均(与模型 test 预测的折平均口径一致),因此 OOF 与线上口径严格对齐。

平滑用贝叶斯先验而非原方案的 +0.01 —— 高基数列上 (sum+0.01)/(count+0.01) 在
count=1 时直接退化成 0 或 1000,是纯噪声;贝叶斯平滑按样本量向全局先验收缩。

用法:ELO_SEED=777 python src/target_encoding.py [列表基数诊断|train]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

BASE_DIR = "outputs/base"
TE_PREFIX = ("te_", "teor_", "tetm_")   # v1 用 te_,v2 起按编码目标分 teor_/tetm_
KEYSET = os.environ.get("ELO_TE_KEYS", "v1")      # v1=原 8 键;v2=+行为分箱与高基数众数
ENCODE_TM = os.environ.get("ELO_TE_TM", "0") == "1"  # 是否追加 clean-target 均值编码
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def te_keys(df: pd.DataFrame):
    """构造参与编码的键列,按基数分两组返回 (低基数, 高基数)。

    分组的理由是交叉的可行性:低基数键两两交叉后每组仍有几百样本,统计稳定;
    高基数键(众数类目/城市,基数 300+)一旦交叉就是几万组、每组 1-2 样本,
    即使贝叶斯平滑也只是把噪声拉回先验,白白增加维度 —— 故只做一阶。

    连续列先离散化:21st 直接对 float 的 month_diff_mean 做 groupby(基数上万,
    每组 1-2 样本,编码值几乎是标签本身的复制)。这里 round 到整数,基数压到十几。
    """
    low = pd.DataFrame(index=df.index)
    for c in ("feature_1", "feature_2", "feature_3"):
        low[c] = df[c].astype(np.int16)
    if "hist_ref_month_id" in df:          # 观察月:本赛 target 分布与其强相关
        low["ref_month"] = df["hist_ref_month_id"].fillna(-1).astype(np.int16)
    for src, name in [("hist_month_diff_mean", "hmd_mean"), ("new_month_diff_mean", "nmd_mean"),
                      ("hist_month_diff_max", "hmd_max"), ("hist_month_diff_min", "hmd_min")]:
        if src in df:
            low[name] = np.round(df[src].fillna(-999).astype(np.float64)).astype(np.int16)

    high = pd.DataFrame(index=df.index)
    if KEYSET in ("v2", "v3"):   # v3/v2 含行为强度分箱(实测为泄漏源)
        # 行为强度分箱:活跃度、授权率、开卡时长 —— 与卡片静态属性正交
        if "hist_authorized_flag_mean" in df:
            low["auth_bin"] = np.clip((df["hist_authorized_flag_mean"].fillna(-1) * 10), -1, 10).astype(np.int16)
        if "hist_count" in df:
            low["cnt_bin"] = np.log2(df["hist_count"].fillna(0).clip(lower=1)).astype(np.int16)
        if "elapsed_days" in df:
            low["elapsed_bin"] = (df["elapsed_days"].fillna(-1) // 90).astype(np.int16)
        # 高基数众数列:TE 正是处理这类列的手段(one-hot 不可行)
        # 高基数众数键仅 v2 使用。v2 线上实测大幅变差(OOF −0.0043 但 Private +0.0059),
        # v3 保留低基数分箱、剔除高基数键,用于隔离归因。
    if KEYSET in ("v2", "v4"):   # v4 = v1 静态键 + 高基数众数键,隔离众数键的净效应
        for src, name in [("hist_mode_merchant_category_id", "mode_mcat"),
                          ("hist_mode_city_id", "mode_city"),
                          ("hist_mode_state_id", "mode_state")]:
            if src in df:
                high[name] = df[src].fillna(-1).astype(np.int32)
    return low, high


def fit_map(keys_tr: pd.DataFrame, tvals_tr: np.ndarray, cols, alpha: float, prior: float):
    """在训练折上统计一组键的目标均值(贝叶斯平滑),返回可用于 map 的 Series。"""
    g = pd.DataFrame({"_k": _join(keys_tr, cols).values, "_y": tvals_tr}).groupby("_k")["_y"]
    s, n = g.sum(), g.count()
    return (s + prior * alpha) / (n + alpha)


def _join(keys: pd.DataFrame, cols):
    """多列组合成单一键(字符串拼接对 int16 足够快,且避免 groupby 多列的开销)。"""
    if len(cols) == 1:
        return keys[cols[0]]
    out = keys[cols[0]].astype(np.int64)
    for c in cols[1:]:
        out = out * 100003 + keys[c].astype(np.int64)   # 大质数混合,冲突可忽略
    return out


def oof_target_encode(train, test, ybin, y, folds, alpha=20.0):
    """返回 (te_train, te_test, 列名):train 侧折内统计,test 侧十折平均。

    编码两种目标:
      · or_ = outlier 发生率(21st 原方案),给树模型"这类卡多容易是哨兵值"的先验;
      · tm_ = **非 outlier 子集**的 target 均值(v2 新增)。直接编码全量 target 均值会被
        -33.22 支配、与 or_ 高度共线;剔除 outlier 后编码的是"正常卡在这类里的忠诚度水平",
        与 or_ 正交 —— 两者合起来才是 target 混合分布的完整刻画。
    """
    lt, ht = te_keys(train)
    ls, hs = te_keys(test)
    lcols, hcols = list(lt.columns), list(ht.columns)
    combos = ([[c] for c in lcols]
              + [[lcols[i], lcols[j]] for i in range(len(lcols)) for j in range(i + 1, len(lcols))]
              + [[c] for c in hcols])
    kt = pd.concat([lt, ht], axis=1)
    ks = pd.concat([ls, hs], axis=1)

    yv = y.to_numpy(np.float64)
    ok = ~(ybin.astype(bool))              # 非 outlier 掩码
    targets = [("or", ybin.astype(np.float64), None)]
    if ENCODE_TM:
        targets.append(("tm", yv, ok))
    names = [f"te{t}_" + "_x_".join(c) for t, _, _ in targets for c in combos]
    log(f"键:低基数 {len(lcols)} + 高基数 {len(hcols)} → {len(combos)} 组合 × "
        f"{len(targets)} 编码目标 = {len(names)} 列")

    te_tr = np.zeros((len(train), len(names)), np.float32)
    te_te = np.zeros((len(test), len(names)), np.float32)
    for k, (tr, va) in enumerate(folds):
        col = 0
        for _, tvals, mask in targets:
            sub = tr if mask is None else tr[mask[tr]]      # tm 只用非 outlier 行统计
            prior = float(tvals[sub].mean())
            for c in combos:
                m = fit_map(kt.iloc[sub], tvals[sub], c, alpha, prior)
                te_tr[va, col] = _join(kt.iloc[va], c).map(m).fillna(prior).to_numpy(np.float32)
                te_te[:, col] += _join(ks, c).map(m).fillna(prior).to_numpy(np.float32) / len(folds)
                col += 1
        log(f"  fold{k + 1} 完成")
    return te_tr, te_te, names


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)

    if mode == "diag":   # 基数诊断
        lo, hi = te_keys(train)
        for tag, d in (("低", lo), ("高", hi)):
            for c in d.columns:
                print(f"  [{tag}] {c:12s} nunique={d[c].nunique():6d}")
        return

    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]

    # TE 矩阵算一次即缓存:后续每个基模型直接复用同一套编码,保证口径一致
    cache = f"outputs/te_features_{KEYSET}{'_tm' if ENCODE_TM else ''}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        te_tr, te_te, names = z["tr"], z["te"], list(z["names"])
        log(f"读取 TE 缓存 {cache}({len(names)} 列)")
    else:
        te_tr, te_te, names = oof_target_encode(train, test, ybin, y, folds)
        np.savez(cache, tr=te_tr, te=te_te, names=np.array(names, dtype=object))

    X = pd.concat([train[sel].reset_index(drop=True),
                   pd.DataFrame(te_tr, columns=names)], axis=1)
    X_test = pd.concat([test[sel].reset_index(drop=True),
                        pd.DataFrame(te_te, columns=names)], axis=1)
    log(f"stage={mode} X={X.shape}(原 {len(sel)} + TE {len(names)})")
    OUT = BASE_DIR + "_te" + ("" if KEYSET == "v1" and not ENCODE_TM else f"_{KEYSET}{'_tm' if ENCODE_TM else ''}")
    os.makedirs(OUT, exist_ok=True)

    def dump(name, oof, pred):
        np.savez(os.path.join(OUT, f"{name}.npz"), oof=oof, pred=pred)
        log(f"[te] {name:6s} OOF={rmse(y, oof):.5f} -> {OUT}/{name}.npz")

    if mode in ("train", "lgb", "all"):
        oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+te")
        dump("lgb", oof, pred)
        ref = np.load(f"{BASE_DIR}/lgb.npz")["oof"]
        log(f"判据:lgb+TE OOF={rmse(y, oof):.5f} vs 无 TE {rmse(y, ref):.5f} "
            f"→ 改善 {rmse(y, ref) - rmse(y, oof):+.5f}")
        imp_te = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
        log("TE 列中 gain 最高的 8 个:\n" + imp_te[imp_te["feature"].str.startswith(TE_PREFIX)].head(8).to_string(index=False))
        n_top = int(imp_te.head(50)["feature"].str.startswith(TE_PREFIX).sum())
        log(f"TE 列进入 gain 前 50 的个数:{n_top}")

    if mode in ("rest", "all"):
        oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds);            dump("xgb", oof, pred)
        oof, pred, _ = ep.cv_catboost(X, y, X_test, folds);           dump("cat", oof, pred)
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "hub+te"); dump("hub", oof, pred)
        oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
        np.savez(os.path.join(OUT, "clf.npz"), oof=oof, pred=pred)
        log(f"[te] clf AUC={auc:.5f}(无 TE 0.90235)")
        # clean:训练侧剔除 outlier,预测完整折
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
            log(f"  [clean+te] fold{k + 1} iter={m.best_iteration}")
        dump("clean", oc, pc)


if __name__ == "__main__":
    main()
