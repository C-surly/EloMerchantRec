# -*- coding: utf-8 -*-
"""v39b:跨仓并池终验(信息回填假设)。

背景:v39 发现旧仓成员在新池融合 ΔOOF -0.00184,远超吸收天花板。良性解释:
新仓为精简重建(E0 3.62411 vs 旧 F31 3.62062,系统性差 0.0035),旧成员携带
重建丢失的信息。若假设成立,并入旧仓 F31 全部原始成员应回收更多。

  A0   E10 复现(锚点)
  A1   SC5 复现(v39 最优,锚点)
  U1   E10 + 旧F31 全部原始成员(o_ 前缀,28 个左右)
  U2   U1 + SC5 中不在 U1 的成员(ct/tp/sk/ssl 世代)
  U3   U2 换 bagLGB 头(对照)

附:折噪声判别 —— 对 SC5 成员输出 std(oof差分) vs std(test差分):
折噪声在 test 端按 √10 收缩,系统性管线差异则两端同量级。

用法:ELO_SEED=777 python src/blending/pool_union.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# 允许 `python src/<子目录>/xxx.py` 直接执行:先把 src/ 挂进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import fusion as vf
import fuse_opt as v15

rmse = vf.rmse
U2_OUT = paths.U2_CSV
SC5 = ["ct_lgb", "ct_clf", "tp_pfn", "tp_lgb", "sk_row", "ssl_dn"]
SC5_PATH = {
    "ct_lgb": "base_ct/lgb.npz", "ct_clf": "base_ct/clf_ct.npz",
    "tp_pfn": "base_tp/pfn.npz", "tp_lgb": "base_tp/lgb.npz",
    "sk_row": "base_sk/new_rowreg.npz", "ssl_dn": "base_nn_clf/ssl_dn_clf.npz",
}


def load_old_f31(prefer_local: bool):
    """历史 F31 原始成员(o_ 前缀);默认 frozen-first,可切到 local-first。"""
    out = {}
    srcs = {}
    for d, pre in [("base", ""), ("base_te", "t_"), ("base_td", "d_"),
                   ("base_fm", "f_"), ("base_nn", "n_")]:
        cur = paths.OUT_DIR / d
        old = paths.frozen_members_dir() / d
        files = set()
        if prefer_local:
            if cur.is_dir():
                files.update(f.name for f in cur.iterdir() if f.suffix == ".npz")
            if old.is_dir():
                files.update(f.name for f in old.iterdir() if f.suffix == ".npz")
        elif old.is_dir():
            files.update(f.name for f in old.iterdir() if f.suffix == ".npz")
        if not files:
            continue
        for f in sorted(files):
            p = paths.resolve_output(f"{d}/{f}", prefer_frozen=not prefer_local)
            z = np.load(p)
            if "oof" in z and "pred" in z:
                key = "o_" + pre + f[:-4]
                out[key] = (np.asarray(z["oof"], float), np.asarray(z["pred"], float))
                srcs[key] = f"{paths.source_tag(p)}:{p}"
    return out, srcs


def main():
    assert os.environ.get("ELO_SEED") == "777"
    prefer_local = os.environ.get("ELO_PREFER_LOCAL_RANK6", "0") == "1"
    bases = vf.load_bases()
    base = pd.read_parquet(paths.FEATURES)
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    yv = y.to_numpy()
    ybin = (yv < -30).astype(int)
    folds = ep.make_folds(y)
    _, znnc = v15.load_avg_parts(v15.NNCLF_PARTS_DIR)
    bases["z_nnc"] = znnc
    sc_srcs = {}
    for k, rel in SC5_PATH.items():
        p = paths.resolve_output(rel, prefer_frozen=not prefer_local)
        z = np.load(p)
        bases[k] = (np.asarray(z["oof"], float), np.asarray(z["pred"], float))
        sc_srcs[k] = f"{paths.source_tag(p)}:{p}"
    old, old_srcs = load_old_f31(prefer_local)
    for k, v in old.items():
        assert v[0].shape == (len(train),) and v[1].shape == (len(test),), k
        bases[k] = v
    print(f"[v39b] SC5 成员来源:", flush=True)
    for k in SC5:
        print(f"[v39b]   {k:<8} <- {sc_srcs[k]}", flush=True)
    print(
        f"[v39b] 历史 F31 成员 {len(old)} 个; "
        f"取数模式={'local-first' if prefer_local else 'frozen-first'}",
        flush=True,
    )
    src_sum = {"outputs": 0, "frozen_members": 0, "external": 0}
    for p in old_srcs.values():
        tag = p.split(":", 1)[0]
        src_sum[tag] = src_sum.get(tag, 0) + 1
    print(f"[v39b] F31 来源统计: {src_sum}", flush=True)

    # 折噪声判别
    print("\n[v39b] 差分结构(折噪声会使 test 端收缩≈√10):")
    for k in ("ct_lgb", "tp_lgb"):
        do = bases[k][0] - bases["f_lgb"][0]
        dt = bases[k][1] - bases["f_lgb"][1]
        print(f"  {k}-f_lgb: std(oof差)={do.std():.4f}  std(test差)={dt.std():.4f} "
              f" 比值={do.std() / max(dt.std(), 1e-9):.2f}", flush=True)

    REG = ["lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et"]
    allf = (REG + ["t_lgb", "t_xgb", "t_cat", "t_hub"]
            + ["d_lgb", "d_xgb", "d_cat", "d_hub"]
            + ["f_lgb", "f_xgb", "f_cat", "f_hub"]
            + ["t_clf", "t_clean", "d_clf", "d_clean", "f_clf", "f_clean",
               "p_cal", "ev", "p_cal_x_clean"]
            + ["n_gru", "n_gru_x", "n_trf"])
    E10 = allf + ["z_nnc", "q_clf"]
    goodSC = ["ct_lgb", "tp_lgb", "sk_row", "ssl_dn", "ct_clf"]
    oldk = sorted(old)

    cache = {}

    def head(feats, model):
        key = (tuple(feats), model)
        if key not in cache:
            cache[key] = v15.evaluate_ext(feats, model, bases, y, ybin, folds,
                                          "f_clf", "f_clean")
        return cache[key]

    def blend2(feats, m2="lgb"):
        _, ob, pb = head(feats, "bayes")
        _, ol, pl = head(feats, m2)
        d = ob - ol
        den = float(d @ d)
        w = 0.5 if den < 1e-12 else float(np.clip((yv - ol) @ d / den, 0.0, 1.0))
        oof = w * ob + (1 - w) * ol
        return rmse(yv, oof), oof, w * pb + (1 - w) * pl, round(w, 3)

    rows, preds, oofs = [], {}, {}

    def run(name, feats, m2="lgb"):
        r, oof, pred, w = blend2(feats, m2)
        rows.append(dict(exp=name, oof=round(r, 5), n=len(feats), w_bayes=w))
        preds[name], oofs[name] = pred, oof
        print(f"[v39b] {name:<26} OOF={r:.5f}  n={len(feats)}  w_bayes={w}", flush=True)
        return r

    run("A0 E10复现", E10)
    run("A1 SC5复现", E10 + goodSC)
    run("U1 E10+旧F31", E10 + oldk)
    run("U2 U1+SC5", E10 + oldk + goodSC)
    run("U3 U2 bagLGB头", E10 + oldk + goodSC, "lgbbag")

    tbl = pd.DataFrame(rows).sort_values("oof").reset_index(drop=True)
    print("\n" + tbl.to_string(index=False), flush=True)
    best = tbl.iloc[0]
    os.makedirs(paths.V39, exist_ok=True)
    with open(os.path.join(paths.V39, "union_results.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if best["exp"] not in ("A0 E10复现",):
        np.savez(os.path.join(paths.V39, "union_best.npz"),
                 oof=oofs[best["exp"]], pred=preds[best["exp"]])
        pd.DataFrame({"card_id": test["card_id"], "target": preds[best["exp"]]}
                     ).to_csv(U2_OUT, index=False)
        with open(os.path.join(paths.V39, "union_best_config.json"), "w") as f:
            json.dump(dict(exp=str(best["exp"]), oof=float(best["oof"])), f,
                      ensure_ascii=False, indent=2)
        print(f"[v39b] 已保存 {U2_OUT}({best['exp']})", flush=True)


if __name__ == "__main__":
    main()
