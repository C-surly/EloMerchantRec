# -*- coding: utf-8 -*-
"""v5 阶段 B:异构基学习器(torch MLP / ExtraTrees),产物同为 outputs/base/{name}.npz。

动机:现有 4 个回归基模型(lgb/xgb/cat/huber-lgb)同族同特征,残差高度相关,
stacking 的去相关收益已饱和(v3→v4 调参+3-seed 仅换来 0.0001 级)。
本脚本引入两个非 GBDT 学习器,单模指标次于 GBDT 也无妨 —— 纳入门槛是
「残差与各 GBDT 的相关性 < 0.95」,即它是否带来新的误差方向。

预处理差异(树模型无需、NN 必需):
  · NaN 中位数填充(NN 无法处理缺失);
  · QuantileTransformer 正态化 —— 聚合特征长尾极重(sum/max 类跨数量级),
    直接标准化会让梯度被极端值支配。

用法:
    python src/hetero.py mlp    # GPU(CUDA_VISIBLE_DEVICES 指定卡)
    python src/hetero.py et     # CPU
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import QuantileTransformer

import elo_pipeline as ep

BASE_DIR = "outputs/base"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def load_data():
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    imp = pd.read_csv("outputs/feature_importance.csv")
    selected = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"]
                if c in train.columns]
    return train[selected], test[selected], y, ep.make_folds(y)


def prep_numeric(X, X_test):
    """NaN 中位数填充 + 分位数正态变换(在 train 上 fit,避免测试集泄漏)。"""
    Xa = X.to_numpy(np.float32)
    Xt = X_test.to_numpy(np.float32)
    Xa[~np.isfinite(Xa)] = np.nan
    Xt[~np.isfinite(Xt)] = np.nan
    med = np.nanmedian(Xa, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa)
    Xt = np.where(np.isnan(Xt), med, Xt)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200000, random_state=ep.CONFIG["SEED"])
    return qt.fit_transform(Xa).astype(np.float32), qt.transform(Xt).astype(np.float32)


# ---------------------------------------------------------------------------
def run_mlp(X, X_test, y, folds, variant="mlp"):
    """十折 MLP:BN+Dropout 的 3 隐层,按验证 RMSE 早停。

    variant='mlp'  → MSE 损失、3 隐层、patience 10(与 GBDT 同目标)
    variant='mlp2' → Huber 损失、更宽更深、训练更充分,对 -33.22 哨兵梯度饱和,
                     误差方向与 MSE 系模型不同,是融合池里更有价值的一员
    """
    import torch
    import torch.nn as nn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} variant={variant}")
    Xa, Xt = prep_numeric(X, X_test)
    ya = y.to_numpy(np.float32)
    Tt = torch.from_numpy(Xt).to(dev)
    huber = variant == "mlp2"
    epochs, patience_max = (200, 25) if huber else (60, 10)

    def build(d_in):
        if huber:
            return nn.Sequential(
                nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 1)).to(dev)
        return nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1)).to(dev)

    oof = np.zeros(len(Xa))
    pred = np.zeros(len(Xt))
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(ep.CONFIG["SEED"] + k)
        net = build(Xa.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.HuberLoss(delta=5.0) if huber else nn.MSELoss()
        Xtr = torch.from_numpy(Xa[tr]).to(dev)
        ytr = torch.from_numpy(ya[tr]).to(dev).unsqueeze(1)
        Xva = torch.from_numpy(Xa[va]).to(dev)
        yva = ya[va]
        best, best_state, patience = np.inf, None, 0
        for ep_i in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(perm), 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad()
                loss = lossf(net(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xva).squeeze(1).cpu().numpy()
            r = rmse(yva, pv)
            if r < best - 1e-5:
                best, patience = r, 0
                best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_max:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            oof[va] = net(torch.from_numpy(Xa[va]).to(dev)).squeeze(1).cpu().numpy()
            pred += net(Tt).squeeze(1).cpu().numpy() / len(folds)
        log(f"  [{variant}] fold{k + 1}: RMSE={best:.5f} (epochs={ep_i + 1})")
    return oof, pred


def run_et(X, X_test, y, folds):
    """十折 ExtraTrees:随机分裂点带来与 GBDT 不同的偏差-方差结构。"""
    from sklearn.ensemble import ExtraTreesRegressor
    Xa = X.to_numpy(np.float32)
    Xt = X_test.to_numpy(np.float32)
    Xa[~np.isfinite(Xa)] = np.nan
    Xt[~np.isfinite(Xt)] = np.nan
    med = np.nanmedian(Xa, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa)
    Xt = np.where(np.isnan(Xt), med, Xt)
    ya = y.to_numpy(np.float32)

    oof = np.zeros(len(Xa))
    pred = np.zeros(len(Xt))
    for k, (tr, va) in enumerate(folds):
        m = ExtraTreesRegressor(n_estimators=300, max_features=0.4, min_samples_leaf=20,
                                n_jobs=ep.CONFIG["N_THREADS"], random_state=ep.CONFIG["SEED"])
        m.fit(Xa[tr], ya[tr])
        oof[va] = m.predict(Xa[va])
        pred += m.predict(Xt) / len(folds)
        log(f"  [et] fold{k + 1}: RMSE={rmse(ya[va], oof[va]):.5f}")
    return oof, pred


# ---------------------------------------------------------------------------
def diversity_report(name, oof, y):
    """残差相关性:与已有基模型逐一对比,决定是否值得纳入融合池。

    口径说明:全行残差相关性在本赛题会被 1.09% 的 -33.22 哨兵样本支配 ——
    所有模型都在同一批 outlier 上大幅错误,残差天然高度相关(常 >0.98),
    据此判断同质会漏掉真正有用的异构成员。因此同时给出**非 outlier 子集**的
    残差相关性,这才反映模型在主体分布上的误差方向差异。
    最终是否纳入以融合层实测 OOF 为准(见 fusion.py 的 F9/F10 方案)。
    """
    yv = y.to_numpy()
    mask = yv > -30
    res_new = yv - oof
    rows = []
    if not os.path.isdir(BASE_DIR):
        log(f"[{name}] {BASE_DIR} 尚不存在,跳过残差相关报告")
        return
    for f in sorted(os.listdir(BASE_DIR)):
        if not f.endswith(".npz") or f[:-4] in ("clf", name):
            continue
        o = np.load(os.path.join(BASE_DIR, f))["oof"]
        res_o = yv - o
        rows.append((f[:-4],
                     float(np.corrcoef(res_new, res_o)[0, 1]),
                     float(np.corrcoef(res_new[mask], res_o[mask])[0, 1])))
    log(f"[{name}] 残差相关(全行):" + "  ".join(f"{k}={a:.4f}" for k, a, _ in rows))
    log(f"[{name}] 残差相关(非 outlier 子集):" + "  ".join(f"{k}={b:.4f}" for k, _, b in rows))
    if rows:
        mx = max(b for _, _, b in rows)
        verdict = "误差方向有差异,值得进融合池" if mx < 0.95 else "与树模型主体误差同质"
        log(f"[{name}] 非 outlier 最大相关={mx:.4f} → {verdict}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    X, X_test, y, folds = load_data()
    log(f"{which}: X={X.shape} X_test={X_test.shape}")
    if which in ("mlp", "mlp2"):
        oof, pred = run_mlp(X, X_test, y, folds, variant=which)
    elif which == "et":
        oof, pred = run_et(X, X_test, y, folds)
    else:
        raise SystemExit(f"未知模型 {which}(可选 mlp / mlp2 / et)")
    log(f"[{which}] OOF RMSE={rmse(y, oof):.5f}")
    diversity_report(which, oof, y)
    os.makedirs(BASE_DIR, exist_ok=True)
    np.savez(os.path.join(BASE_DIR, f"{which}.npz"), oof=oof, pred=pred)
    log(f"[{which}] 保存 {BASE_DIR}/{which}.npz")


if __name__ == "__main__":
    main()
