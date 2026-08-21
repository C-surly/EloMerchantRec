# -*- coding: utf-8 -*-
"""v14:NN 族纵深(docs/202608/03-v14_NN族纵深.md)。

三路:A gru(v13 架构 × 多 seed 平均)  B gru_x(10 通道 ext + 2 层)  C trf(Transformer)。
中间产物 outputs/nn_parts/<arch>_s<seed>.npz;平均后成员 outputs/base_nn/<name>.npz。
判据在融合层(fusion 新 F31 vs 旧 F31 3.62162)。
用法:ELO_SEED=777 python src/seq_nn.py data_ext
     ELO_SEED=777 python src/seq_nn.py train <gru|gru_x|trf> <dev_id> [seeds,逗号分隔]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import seq_gru as v13

PARTS_DIR = "outputs/nn_parts"
OUT_DIR = "outputs/base_nn"
SEQX_CACHE = "outputs/seq_tensor_ext.npz"
SEEDS = [777, 1777, 2777, 3777, 4777]
ARCH_OUT = {"gru": "gru", "gru_x": "gru_x", "trf": "trf"}
LAGS = v13.LAGS
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_seq_ext():
    """10 通道版:v13 的 7 通道 + 被拒逐月笔数/占比 + 城市广度。"""
    base = pd.read_parquet("data/processed/features.parquet")
    cols = ["card_id", "month_lag", "purchase_amount", "merchant_id",
            "installments", "category_1", "city_id", "authorized_flag"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))[cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    tx["card_id"] = tx["card_id"].astype(str)
    del hist, new
    ok = tx[tx["authorized_flag"] == 1]
    g = ok.groupby(["card_id", "month_lag"], observed=True).agg(
        amt=("purchase_amount", "sum"), amx=("purchase_amount", "max"),
        cnt=("purchase_amount", "size"), mer=("merchant_id", "nunique"),
        inst=("installments", "mean"), c1=("category_1", "mean"),
        city=("city_id", "nunique"))
    den = (tx[tx["authorized_flag"] == 0]
           .groupby(["card_id", "month_lag"], observed=True)["purchase_amount"]
           .size().rename("dcnt"))
    g = g.join(den, how="outer")
    log(f"逐月聚合 {g.shape}")

    piv = {c: g[c].unstack().reindex(columns=LAGS) for c in g.columns}
    cards = piv["amt"].index
    fz = lambda c: piv[c].fillna(0.0).to_numpy(np.float32)
    cnt, dcnt = fz("cnt"), fz("dcnt")
    chan = [np.log1p(np.clip(fz(c), 0, None)) for c in ("amt", "amx", "cnt", "mer")]
    chan += [(cnt > 0).astype(np.float32), fz("inst"), fz("c1"),
             np.log1p(dcnt), (dcnt / np.clip(cnt + dcnt, 1, None)).astype(np.float32),
             np.log1p(fz("city"))]
    seq = np.stack(chan, axis=2)                       # [N,16,10]
    keep = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1], np.float32)
    idxmap = pd.Index(cards.astype(str))

    base["n_months"] = (base["hist_month_lag_max"].fillna(0)
                        - base["hist_month_lag_min"].fillna(-12) + 1)
    out = {}
    for part, flag in [("tr", 1), ("te", 0)]:
        side = base[base["is_train"] == flag]
        idx = idxmap.get_indexer(side["card_id"].astype(str))
        xs = np.zeros((len(side), len(LAGS), seq.shape[2]), np.float32)
        hit = idx >= 0
        xs[hit] = seq[idx[hit]]
        out[f"xs_{part}"] = xs
        out[f"st_{part}"] = side[v13.STATIC].fillna(0).to_numpy(np.float32)
        log(f"{part}: xs={xs.shape} 无交易卡 {int((~hit).sum())}")
    m = out["xs_tr"].reshape(-1, seq.shape[2])
    mu, sd = m.mean(0), m.std(0) + 1e-6
    for p in ("tr", "te"):
        out[f"xs_{p}"] = (out[f"xs_{p}"] - mu * keep) / np.where(keep > 0, sd, 1.0)
    smu, ssd = out["st_tr"].mean(0), out["st_tr"].std(0) + 1e-6
    for p in ("tr", "te"):
        out[f"st_{p}"] = (out[f"st_{p}"] - smu) / ssd
    np.savez_compressed(SEQX_CACHE, **out)
    log(f"ext 序列缓存 {SEQX_CACHE}: " + ", ".join(f"{k}={v.shape}" for k, v in out.items()))


def train(arch, dev_id, seeds):
    import torch
    import torch.nn as nn
    import nn_runtime as nnrt
    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    if arch in {"gru", "gru_x"}:
        num_layers = 2 if arch == "gru_x" else 1
        dropout = 0.1 if arch == "gru_x" else 0.0
        input_size = 10 if arch == "gru_x" else 7
        if nnrt.ensure_gru_backend(dev, input_size, 96,
                                   num_layers=num_layers, dropout=dropout):
            log(f"[{arch}] 检测到 cuDNN RNN 初始化失败,自动切换到原生 CUDA GRU 后端")
    z = np.load(SEQX_CACHE if arch == "gru_x" else v13.SEQ_CACHE)
    xs_tr, st_tr, xs_te, st_te = z["xs_tr"], z["st_tr"], z["xs_te"], z["st_te"]
    base = pd.read_parquet("data/processed/features.parquet")
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yv = y.to_numpy(np.float32)
    C, S = xs_tr.shape[2], st_tr.shape[1]
    bs = nnrt.resolve_batch_size(arch, "train")
    infer_bs = nnrt.resolve_batch_size(arch, "infer")
    log(f"[{arch}] train {xs_tr.shape} dev={dev} seeds={seeds} bs={bs} infer_bs={infer_bs}")

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            if arch == "trf":
                self.proj = nn.Linear(C, 64)
                self.pos = nn.Parameter(torch.zeros(1, len(LAGS), 64))
                layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, 2)
                enc_dim = 64
            else:
                nl = 2 if arch == "gru_x" else 1
                self.enc = nn.GRU(C, 96, num_layers=nl, batch_first=True,
                                  dropout=0.1 if nl > 1 else 0.0)
                enc_dim = 96
            self.stat = nn.Linear(S, 32)
            self.head = nn.Sequential(
                nn.Linear(enc_dim + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st):
            if self.arch == "trf":
                h = self.enc(self.proj(xs) + self.pos).mean(1)
            else:
                _, hn = self.enc(xs)
                h = hn[-1]
            return self.head(torch.cat([h, torch.relu(self.stat(st))], 1)).squeeze(1)

    XS, ST = torch.from_numpy(xs_tr).to(dev), torch.from_numpy(st_tr).to(dev)
    Y = torch.from_numpy(yv).to(dev)
    XSe, STe = torch.from_numpy(xs_te).to(dev), torch.from_numpy(st_te).to(dev)

    os.makedirs(PARTS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    BS, MAX_EP, PAT = bs, 40, 5
    accum_oof, accum_pred, done = [], [], []
    for sd in seeds:
        oof, pred = np.zeros(len(yv)), np.zeros(len(xs_te))
        for k, (tr, va) in enumerate(folds):
            torch.manual_seed(sd + k * 101)
            np.random.seed(sd + k * 101)
            model = SeqNet().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
            lossf = nn.MSELoss()
            tr_t = torch.from_numpy(tr).to(dev)
            best, wait, best_state = 1e9, 0, None
            for _ in range(MAX_EP):
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
                    best, wait = vr, 0
                    best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= PAT:
                        break
            model.load_state_dict(best_state)
            oof[va] = nnrt.infer_numpy(model, XS[va], ST[va], infer_bs)
            pred += nnrt.infer_numpy(model, XSe, STe, infer_bs) / len(folds)
        np.savez(os.path.join(PARTS_DIR, f"{arch}_s{sd}.npz"), oof=oof, pred=pred)
        accum_oof.append(oof)
        accum_pred.append(pred)
        done.append(sd)
        avg_o = np.mean(accum_oof, 0)
        log(f"[{arch}] seed={sd} OOF={rmse(yv, oof):.5f} | {len(done)}seed 平均 OOF={rmse(yv, avg_o):.5f}")
    np.savez(os.path.join(OUT_DIR, f"{ARCH_OUT[arch]}.npz"),
             oof=np.mean(accum_oof, 0), pred=np.mean(accum_pred, 0))
    log(f"[{arch}] 完成:{len(done)} seed 平均 OOF={rmse(yv, np.mean(accum_oof, 0)):.5f}"
        f" -> {OUT_DIR}/{ARCH_OUT[arch]}.npz")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1]
    if mode == "data_ext":
        if not os.path.exists(SEQX_CACHE):
            build_seq_ext()
        return
    if mode == "train":
        arch = sys.argv[2]
        dev_id = sys.argv[3] if len(sys.argv) > 3 else "0"
        seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else SEEDS
        train(arch, dev_id, seeds)


if __name__ == "__main__":
    main()
