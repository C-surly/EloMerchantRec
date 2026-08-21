# -*- coding: utf-8 -*-
"""v39:E10 之上的剩余零成本候选终验(2026-08-10《提分空间独立评估》P0 项)。

  G0    E10 复现(bayes+浅LGB 解析权重,+z_nnc+q_clf)—— 判据锚点,应=3.62180
  G1    E10 换 bagged-LGB 头(5-seed 平均;纯降方差,与 E5 双头同机制)      [P0-2]
  G2    三头解析 bayes+lgb+xgb(补跑原 E16,SLSQP 单纯形权重)
  G3    三头 + bagged-LGB 头(G1×G2 组合)
  S_*   旧仓库低判据真信号成员逐个裸列进 E10                            [P0-3]
        折协议兼容性已验证:两仓 train/test 卡序与 target 完全一致,
        make_folds 逐字相同、同机同环境同 ELO_SEED=777 → 折切分一致,无堆叠污染。
        ct_lgb/ct_clf = v24 数字结构族;tp_pfn/tp_lgb = v33 TabPFN 族;
        sk_row = v33 senkin13 行级回归;ssl_* = v27 自监督预训练 clf 三变体。
  SC    个体 ΔOOF<0 的成员合流(+最优头配置)

判据纪律:提交候选须 ΔOOF ≤ -0.0005 并过「新信息×异构」定性关(v22 教训);
中间对照仅记录。用法:ELO_SEED=777 python src/blending/pool_sc5.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 允许 `python src/<子目录>/xxx.py` 直接执行:先把 src/ 挂进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import fusion as vf
import fuse_opt as v15

rmse = vf.rmse
SC5_OUT = paths.SC5_CSV
CAND = {
    "ct_lgb": "base_ct/lgb.npz",
    "ct_clf": "base_ct/clf_ct.npz",
    "tp_pfn": "base_tp/pfn.npz",
    "tp_lgb": "base_tp/lgb.npz",
    "sk_row": "base_sk/new_rowreg.npz",
    "ssl_dn": "base_nn_clf/ssl_dn_clf.npz",
    "ssl_full": "base_nn_clf/ssl_full_clf.npz",
    "ssl_clf": "base_nn_clf/ssl_clf.npz",
}


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    prefer_local = os.environ.get("ELO_PREFER_LOCAL_RANK6", "0") == "1"
    bases = vf.load_bases()
    base = pd.read_parquet(paths.FEATURES)
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    yv = y.to_numpy()
    ybin = (yv < -30).astype(int)
    folds = ep.make_folds(y)

    members, znnc = v15.load_avg_parts(v15.NNCLF_PARTS_DIR)
    assert znnc is not None, "缺 nn_clf_parts"
    bases["z_nnc"] = znnc
    assert "q_clf" in bases, "缺 base_dq/q_clf.npz"
    srcs = {}
    for k, rel in CAND.items():
        p = paths.resolve_output(rel, prefer_frozen=not prefer_local)
        z = np.load(p)
        assert z["oof"].shape == (len(train),), f"{k} oof 形状不符"
        assert z["pred"].shape == (len(test),), f"{k} pred 形状不符"
        bases[k] = (np.asarray(z["oof"], float), np.asarray(z["pred"], float))
        srcs[k] = f"{paths.source_tag(p)}:{p}"
    print(
        f"[v39] z_nnc 成员={members};候选成员={list(CAND)}; "
        f"取数模式={'local-first' if prefer_local else 'frozen-first'}",
        flush=True,
    )
    for k in CAND:
        print(f"[v39]   {k:<8} <- {srcs[k]}", flush=True)

    REG = ["lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et"]
    allf = (REG + ["t_lgb", "t_xgb", "t_cat", "t_hub"]
            + ["d_lgb", "d_xgb", "d_cat", "d_hub"]
            + ["f_lgb", "f_xgb", "f_cat", "f_hub"]
            + ["t_clf", "t_clean", "d_clf", "d_clean", "f_clf", "f_clean",
               "p_cal", "ev", "p_cal_x_clean"]
            + ["n_gru", "n_gru_x", "n_trf"])
    E10 = allf + ["z_nnc", "q_clf"]

    cache = {}

    def head(feats, model):
        key = (tuple(feats), model)
        if key not in cache:
            cache[key] = v15.evaluate_ext(feats, model, bases, y, ybin, folds,
                                          "f_clf", "f_clean")
        return cache[key]  # (rmse, oof, pred)

    def blend2(feats, m2):
        _, ob, pb = head(feats, "bayes")
        _, ol, pl = head(feats, m2)
        d = ob - ol
        den = float(d @ d)
        w = 0.5 if den < 1e-12 else float(np.clip((yv - ol) @ d / den, 0.0, 1.0))
        oof = w * ob + (1 - w) * ol
        return rmse(yv, oof), oof, w * pb + (1 - w) * pl, (round(w, 3), round(1 - w, 3))

    def blend3(feats, m2):
        _, ob, pb = head(feats, "bayes")
        _, ol, pl = head(feats, m2)
        _, ox, px = head(feats, "xgb")
        O = np.column_stack([ob, ol, ox])
        P = np.column_stack([pb, pl, px])
        r = minimize(lambda w: rmse(yv, O @ w), np.ones(3) / 3, method="SLSQP",
                     bounds=[(0, 1)] * 3,
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        return rmse(yv, O @ r.x), O @ r.x, P @ r.x, tuple(round(v, 3) for v in r.x)

    rows, preds, oofs = [], {}, {}

    def run(name, fn, *args):
        r, oof, pred, wt = fn(*args)
        rows.append(dict(exp=name, oof=round(r, 5), w=str(wt)))
        preds[name], oofs[name] = pred, oof
        base_r = rows[0]["oof"]
        print(f"[v39] {name:<22} OOF={r:.5f}  Δvs G0={r - base_r:+.5f}  w={wt}", flush=True)
        return r

    g0 = run("G0 E10复现", blend2, E10, "lgb")
    run("G1 E10+bagLGB头", blend2, E10, "lgbbag")
    run("G2 三头bayes+lgb+xgb", blend3, E10, "lgb")
    run("G3 三头+bagLGB", blend3, E10, "lgbbag")
    singles = {}
    for k in CAND:
        singles[k] = run(f"S_{k}", blend2, E10 + [k], "lgb")
    good = [k for k, r in singles.items() if r < g0 - 1e-6]
    if good:
        print(f"[v39] 个体改善成员: {good}", flush=True)
        run("SC 合流(lgb头)", blend2, E10 + good, "lgb")
        run("SC 合流(bagLGB头)", blend2, E10 + good, "lgbbag")
        run("SC 合流(三头bag)", blend3, E10 + good, "lgbbag")

    tbl = pd.DataFrame(rows).sort_values("oof").reset_index(drop=True)
    print("\n" + tbl.to_string(index=False), flush=True)
    best = tbl.iloc[0]
    print(f"\n[v39] 最优 {best['exp']} OOF={best['oof']:.5f} vs E10 {g0:.5f} "
          f"→ {best['oof'] - g0:+.5f}(判据线 -0.00050)", flush=True)

    os.makedirs(paths.V39, exist_ok=True)
    with open(os.path.join(paths.V39, "results.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(paths.V39, "G0_e10.npz"), oof=oofs["G0 E10复现"], pred=preds["G0 E10复现"])
    if best["exp"] != "G0 E10复现" and best["oof"] < g0:
        np.savez(os.path.join(paths.V39, "best.npz"), oof=oofs[best["exp"]], pred=preds[best["exp"]])
        pd.DataFrame({"card_id": test["card_id"],
                      "target": preds[best["exp"]]}
                     ).to_csv(SC5_OUT, index=False)
        with open(os.path.join(paths.V39, "best_config.json"), "w") as f:
            json.dump(dict(exp=str(best["exp"]), oof=float(best["oof"]),
                           delta_vs_e10=float(best["oof"] - g0)), f,
                      ensure_ascii=False, indent=2)
        print(f"[v39] 已保存 {SC5_OUT}(提交与否按判据另行决策)", flush=True)


if __name__ == "__main__":
    main()
