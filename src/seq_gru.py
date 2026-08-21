# -*- coding: utf-8 -*-
"""v13:序列端到端 GRU(docs/202608/03-v13_序列端到端.md)。

hist(auth=1)+new 合并为逐月序列(month_lag -13..+2,16 步 × 7 通道),GRU 直接回归 target。
静态分支刻意瘦(5 列):要的是树模型看不到的序列形状,保持融合互补性。
判据:正式判据在融合层(fusion F30/F31 vs F29 3.62340,ΔOOF>0.0005);单模 OOF 仅参考。
用法:ELO_SEED=777 python src/seq_gru.py data|train|all
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

import paths

SEQ_CACHE = paths.out("seq_tensor.npz")
OUT_DIR = paths.out("base_nn")
LAGS = list(range(-13, 3))          # 16 步
STATIC = ["feature_1", "feature_2", "feature_3", "elapsed_days", "n_months"]
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_seq():
    """月度序列张量,按 features.parquet 的 train/test card 顺序对齐。"""
    base = pd.read_parquet(paths.FEATURES)
    cols = ["card_id", "month_lag", "purchase_amount", "merchant_id", "installments", "category_1"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1][cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    tx["card_id"] = tx["card_id"].astype(str)
    del hist, new
    log(f"合并交易 {tx.shape}")
    g = tx.groupby(["card_id", "month_lag"], observed=True).agg(
        amt=("purchase_amount", "sum"), amx=("purchase_amount", "max"),
        cnt=("purchase_amount", "size"), mer=("merchant_id", "nunique"),
        inst=("installments", "mean"), c1=("category_1", "mean"))
    log(f"逐月聚合 {g.shape}")

    def pivot(col):
        return g[col].unstack().reindex(columns=LAGS)

    mats = {c: pivot(c) for c in ["amt", "amx", "cnt", "mer", "inst", "c1"]}
    cards = mats["amt"].index
    chan = []
    for c in ["amt", "amx", "cnt", "mer"]:
        chan.append(np.log1p(np.clip(mats[c].fillna(0.0).to_numpy(np.float32), 0, None)))
    chan.append((mats["cnt"].fillna(0.0).to_numpy(np.float32) > 0).astype(np.float32))  # active
    chan.append(mats["inst"].fillna(0.0).to_numpy(np.float32))
    chan.append(mats["c1"].fillna(0.0).to_numpy(np.float32))
    seq = np.stack(chan, axis=2)                     # [N_cards, 16, 7]
    seq_df_index = pd.Index(cards.astype(str))

    base["n_months"] = (base["hist_month_lag_max"].fillna(0)
                        - base["hist_month_lag_min"].fillna(-12) + 1)
    out = {}
    for part, flag in [("tr", 1), ("te", 0)]:
        side = base[base["is_train"] == flag]
        idx = seq_df_index.get_indexer(side["card_id"].astype(str))
        xs = np.zeros((len(side), len(LAGS), seq.shape[2]), np.float32)
        hit = idx >= 0
        xs[hit] = seq[idx[hit]]
        st = side[STATIC].fillna(0).to_numpy(np.float32)
        out[f"xs_{part}"], out[f"st_{part}"] = xs, st
        log(f"{part}: xs={xs.shape} 无交易卡 {int((~hit).sum())}")
    # z-score(按 train):序列 4 个 log 通道 + inst;静态全列
    m = out["xs_tr"].reshape(-1, seq.shape[2])
    mu, sd = m.mean(0), m.std(0) + 1e-6
    keep = np.array([1, 1, 1, 1, 0, 1, 0], np.float32)   # active/c1 不动
    for p in ("tr", "te"):
        out[f"xs_{p}"] = (out[f"xs_{p}"] - mu * keep) / np.where(keep > 0, sd, 1.0)
    smu, ssd = out["st_tr"].mean(0), out["st_tr"].std(0) + 1e-6
    for p in ("tr", "te"):
        out[f"st_{p}"] = (out[f"st_{p}"] - smu) / ssd
    np.savez_compressed(SEQ_CACHE, **out)
    log(f"序列缓存 {SEQ_CACHE}: " + ", ".join(f"{k}={v.shape}" for k, v in out.items()))


def train():
    import torch
    import torch.nn as nn
    import nn_runtime as nnrt
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if nnrt.ensure_gru_backend(dev, 7, 96):
        log("[gru] 检测到 cuDNN RNN 初始化失败,自动切换到原生 CUDA GRU 后端")
    z = np.load(SEQ_CACHE)
    xs_tr, st_tr = z["xs_tr"], z["st_tr"]
    xs_te, st_te = z["xs_te"], z["st_te"]
    base = pd.read_parquet(paths.FEATURES)
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yv = y.to_numpy(np.float32)
    bs = nnrt.resolve_batch_size("gru", "train")
    infer_bs = nnrt.resolve_batch_size("gru", "infer")
    log(f"train {xs_tr.shape} test {xs_te.shape} dev={dev} bs={bs} infer_bs={infer_bs}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(xs_tr.shape[2], 96, batch_first=True)
            self.stat = nn.Linear(st_tr.shape[1], 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st):
            _, h = self.gru(xs)
            return self.head(torch.cat([h[-1], torch.relu(self.stat(st))], 1)).squeeze(1)

    XS = torch.from_numpy(xs_tr).to(dev)
    ST = torch.from_numpy(st_tr).to(dev)
    Y = torch.from_numpy(yv).to(dev)
    XSe = torch.from_numpy(xs_te).to(dev)
    STe = torch.from_numpy(st_te).to(dev)

    oof = np.zeros(len(yv))
    pred = np.zeros(len(xs_te))
    BS, MAX_EP, PAT = bs, 40, 5
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(777 + k)
        np.random.seed(777 + k)
        model = Net().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        lossf = nn.MSELoss()
        tr_t = torch.from_numpy(tr).to(dev)
        best, best_ep, wait, best_state = 1e9, -1, 0, None
        for epn in range(MAX_EP):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(perm), BS):
                b = perm[i:i + BS]
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(XS[b], ST[b]), Y[b])
                loss.backward()
                opt.step()
            vr = rmse(yv[va], nnrt.infer_numpy(model, XS[va], ST[va], infer_bs))
            sch.step(vr)
            if vr < best - 1e-5:
                best, best_ep, wait = vr, epn, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PAT:
                    break
        model.load_state_dict(best_state)
        oof[va] = nnrt.infer_numpy(model, XS[va], ST[va], infer_bs)
        pred += nnrt.infer_numpy(model, XSe, STe, infer_bs) / len(folds)
        log(f"  [gru] fold{k + 1}: RMSE={best:.5f} (ep={best_ep + 1})")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "gru.npz"), oof=oof, pred=pred)
    log(f"[nn] gru OOF={rmse(yv, oof):.5f}(参考值,正式判据在融合层)-> {OUT_DIR}/gru.npz")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("data", "all") and not os.path.exists(SEQ_CACHE):
        build_seq()
    if mode in ("train", "all"):
        train()


if __name__ == "__main__":
    main()
