# -*- coding: utf-8 -*-
"""v33:TabPFN 成员 tp_pfn。"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import formula as v11
import fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
CTX = 10000
E_VAL = 3
CHUNK = 4096


def load_te():
    for name in ("te_features_v1.npz", "te_features.npz"):
        p = paths.out(name)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            return z["tr"], z["te"], [str(x) for x in z["names"]]
    raise FileNotFoundError("缺少 TE 缓存: te_features_v1.npz / te_features.npz")


def main():
    assert os.environ.get("ELO_SEED") == "777"
    from tabpfn import TabPFNRegressor

    base = pd.read_parquet(paths.FEATURES)
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv(paths.FEATURE_IMPORTANCE)
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    te_tr, te_te, te_names = load_te()
    td = pd.read_parquet(paths.out("td_features.parquet"))

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True)], axis=1)

    x = asm(train, te_tr, fm_tr).to_numpy(np.float32)
    x_test = asm(test, te_te, fm_te).to_numpy(np.float32)
    x = np.nan_to_num(x, nan=-999, posinf=1e9, neginf=-1e9)
    x_test = np.nan_to_num(x_test, nan=-999, posinf=1e9, neginf=-1e9)
    yv = y.to_numpy(np.float32)
    ybin = (yv < -30).astype(int)
    log(f"X={x.shape} test={x_test.shape};CTX={CTX} E_VAL={E_VAL}")

    oof = np.zeros(len(x))
    pred = np.zeros(len(x_test))
    for k, (tr, va) in enumerate(folds):
        rng = np.random.RandomState(777 + k)
        val_acc = np.zeros(len(va))
        for e in range(E_VAL):
            pos = tr[ybin[tr] == 1]
            neg = tr[ybin[tr] == 0]
            n_pos = max(int(CTX * len(pos) / len(tr)), 50)
            ctx = np.r_[rng.choice(pos, n_pos, replace=False),
                        rng.choice(neg, CTX - n_pos, replace=False)]
            model = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True,
                                    random_state=777 + k * 10 + e)
            model.fit(x[ctx], yv[ctx])
            for i in range(0, len(va), CHUNK):
                val_acc[i:i + CHUNK] += model.predict(x[va[i:i + CHUNK]])
            if e == 0:
                for i in range(0, len(x_test), CHUNK):
                    pred[i:i + CHUNK] += model.predict(x_test[i:i + CHUNK]) / len(folds)
        oof[va] = val_acc / E_VAL
        log(f"fold{k} val_rmse={rmse(yv[va], oof[va]):.5f}")
    score = rmse(yv, oof)
    os.makedirs(paths.out("base_tp"), exist_ok=True)
    np.savez(paths.out("base_tp", "pfn.npz"), oof=oof, pred=pred)
    zl = np.load(paths.out("base_fm", "lgb.npz"))
    corr = float(np.corrcoef(oof, zl["oof"])[0, 1])
    log(f"TabPFN 单模 OOF={score:.5f}(f_lgb 相关 {corr:.4f});弱度 vs 融合基线 3.62062:{score - 3.62062:+.4f}")

    bases = vf.load_bases()
    bases["tp_pfn"] = (oof, pred)
    reg = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    t_reg = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    d_reg = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    f_reg = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    n_reg = sorted(k for k in bases if k.startswith("n_"))
    allf = (reg + t_reg + d_reg + f_reg + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + n_reg)
    ybin_s = (y < -30).astype(int).to_numpy()
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin_s, folds, p_src="f_clf", clean_src="f_clean")
    r1, _, pt = vf.evaluate(allf + ["tp_pfn"], "bayes", bases, y, ybin_s, folds,
                            p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v33 tp_pfn 入池,线 0.0007]:基线={r0:.5f} 加入后={r1:.5f} → Δ={d:+.5f} "
        f"{'✅ 通过' if d > 0.0007 else '❌ 不足'}")
    if d > 0.0007:
        sub = pd.read_csv(paths.raw("sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv(paths.out("submission_v33a.csv"), index=False)
        log(f"已保存 {paths.out('submission_v33a.csv')}(待提交)")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
