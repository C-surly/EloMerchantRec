# -*- coding: utf-8 -*-
"""v17:交易级序列模型。

每卡最近 128 笔交易(hist auth=1 + new,时间升序,pad 前置)作为 token 序列,
数值 7 + 类别 emb(品类/行业),GRU 直接回归 target。静态分支复用月序列侧的瘦静态。

用法:
  ELO_SEED=777 python src/archive/tx_seq.py data
  ELO_SEED=777 python src/archive/tx_seq.py train <dev_id> <seeds逗号>
  ELO_SEED=777 python src/archive/tx_seq.py merge
"""
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
import seq_gru as v13

TX_CACHE = paths.out("tx_tensor.npz")
PARTS_DIR = paths.out("nn_parts")
OUT_DIR = paths.out("base_nn")
L = 128
SEEDS_ALL = [777, 1777, 2777, 3777, 4777]
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_tx():
    if not os.path.exists(v13.SEQ_CACHE):
        v13.build_seq()
    base = pd.read_parquet(paths.FEATURES)
    cols = ["card_id", "purchase_date", "purchase_amount", "month_lag",
            "installments", "merchant_category_id", "subsector_id"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1][cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    del hist, new
    tx["card_id"] = tx["card_id"].astype(str)
    tx = tx.sort_values(["card_id", "purchase_date"], kind="mergesort").reset_index(drop=True)
    log(f"交易合并排序 {tx.shape}")
    tx["dg"] = (tx.groupby("card_id", observed=True)["purchase_date"].diff()
                .dt.total_seconds().fillna(0.0) / 86400.0)
    tx["rn"] = tx.groupby("card_id", observed=True).cumcount(ascending=False)
    tx = tx[tx["rn"] < L]
    mcat, _ = pd.factorize(tx["merchant_category_id"], use_na_sentinel=True)
    ssec, _ = pd.factorize(tx["subsector_id"], use_na_sentinel=True)
    tx["mcat"] = (mcat + 2).astype(np.int16)
    tx["ssec"] = (ssec + 2).astype(np.int16)
    n_mcat, n_ssec = int(tx["mcat"].max()) + 1, int(tx["ssec"].max()) + 1
    log(f"截断到近 {L} 笔:{tx.shape},vocab mcat={n_mcat} ssec={n_ssec}")

    cards = pd.Index(base["card_id"].astype(str))
    cidx = pd.Series(np.arange(len(cards)), index=cards)
    tx["ci"] = tx["card_id"].map(cidx)
    tx = tx[tx["ci"].notna()]
    ci = tx["ci"].to_numpy(np.float64).astype(np.int32)
    pos = (L - 1 - tx["rn"]).to_numpy(np.int32)

    dow = tx["purchase_date"].dt.dayofweek.to_numpy(np.float32)
    num = np.zeros((len(cards), L, 7), np.float16)
    num[:, :, 6] = 1.0
    feats = np.stack([
        np.log1p(np.clip(tx["purchase_amount"].to_numpy(np.float64), 0, None)) / 10.0,
        np.log1p(np.clip(tx["dg"].to_numpy(np.float64), 0, None)) / 5.0,
        tx["month_lag"].to_numpy(np.float64) / 13.0,
        dow / 6.0,
        (dow >= 5).astype(np.float32),
        np.nan_to_num(tx["installments"].to_numpy(np.float64), nan=0.0) / 12.0,
        np.zeros(len(tx)),
    ], axis=1).astype(np.float16)
    num[ci, pos] = feats
    cats = np.zeros((len(cards), L, 2), np.int16)
    cats[ci, pos, 0] = tx["mcat"].to_numpy(np.int16)
    cats[ci, pos, 1] = tx["ssec"].to_numpy(np.int16)

    zs = np.load(v13.SEQ_CACHE)
    is_tr = (base["is_train"] == 1).to_numpy()
    np.savez_compressed(
        TX_CACHE,
        num_tr=num[is_tr],
        num_te=num[~is_tr],
        cat_tr=cats[is_tr],
        cat_te=cats[~is_tr],
        st_tr=zs["st_tr"],
        st_te=zs["st_te"],
        vocab=np.array([n_mcat, n_ssec]),
    )
    log(f"交易级张量缓存 {TX_CACHE}: num_tr={num[is_tr].shape} cat_tr={cats[is_tr].shape}")


def train(dev_id, seeds, suffix=""):
    import torch
    import torch.nn as nn

    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    z = np.load(TX_CACHE)
    n_mcat, n_ssec = int(z["vocab"][0]), int(z["vocab"][1])
    base = pd.read_parquet(paths.FEATURES)
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yv = y.to_numpy(np.float32)
    log(f"[gru_t] dev={dev} seeds={seeds} vocab=({n_mcat},{n_ssec})")

    class TxNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.em = nn.Embedding(n_mcat, 8, padding_idx=0)
            self.es = nn.Embedding(n_ssec, 4, padding_idx=0)
            self.proj = nn.Linear(7 + 12, 64)
            self.enc = nn.GRU(64, 96, batch_first=True)
            self.stat = nn.Linear(z["st_tr"].shape[1], 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xn, xc, st):
            e = torch.cat([xn, self.em(xc[:, :, 0]), self.es(xc[:, :, 1])], 2)
            _, hn = self.enc(torch.relu(self.proj(e)))
            return self.head(torch.cat([hn[-1], torch.relu(self.stat(st))], 1)).squeeze(1)

    NUM = torch.from_numpy(z["num_tr"].astype(np.float32)).to(dev)
    CAT = torch.from_numpy(z["cat_tr"].astype(np.int64)).to(dev)
    ST = torch.from_numpy(z["st_tr"]).to(dev)
    Y = torch.from_numpy(yv).to(dev)
    NUMe = torch.from_numpy(z["num_te"].astype(np.float32)).to(dev)
    CATe = torch.from_numpy(z["cat_te"].astype(np.int64)).to(dev)
    STe = torch.from_numpy(z["st_te"]).to(dev)

    def infer(model, xn, xc, st, bs=4096):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xn[i:i + bs], xc[i:i + bs], st[i:i + bs])
                              for i in range(0, len(xn), bs)]).float().cpu().numpy()

    os.makedirs(PARTS_DIR, exist_ok=True)
    BS, MAX_EP, PAT = 512, 30, 4
    for sd in seeds:
        oof, pred = np.zeros(len(yv)), np.zeros(len(NUMe))
        for k, (tr, va) in enumerate(folds):
            torch.manual_seed(sd + k * 101)
            np.random.seed(sd + k * 101)
            model = TxNet().to(dev)
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
                    opt.zero_grad()
                    loss = lossf(model(NUM[b], CAT[b], ST[b]), Y[b])
                    loss.backward()
                    opt.step()
                vr = rmse(yv[va], infer(model, NUM[va], CAT[va], ST[va]))
                sch.step(vr)
                if vr < best - 1e-5:
                    best, wait = vr, 0
                    best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= PAT:
                        break
            model.load_state_dict(best_state)
            oof[va] = infer(model, NUM[va], CAT[va], ST[va])
            pred += infer(model, NUMe, CATe, STe) / len(folds)
            log(f"[gru_t] seed={sd} fold{k + 1}: {best:.5f}")
        np.savez(os.path.join(PARTS_DIR, f"gru_t_s{sd}{suffix}.npz"), oof=oof, pred=pred)
        log(f"[gru_t] seed={sd} OOF={rmse(yv, oof):.5f}")


def merge():
    base = pd.read_parquet(paths.FEATURES)
    y = base[base["is_train"] == 1]["target"].to_numpy()
    oofs, preds = [], []
    for sd in SEEDS_ALL:
        p = os.path.join(PARTS_DIR, f"gru_t_s{sd}.npz")
        if os.path.exists(p):
            zz = np.load(p)
            oofs.append(zz["oof"])
            preds.append(zz["pred"])
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "gru_t.npz"),
             oof=np.mean(oofs, 0), pred=np.mean(preds, 0))
    log(f"[gru_t] merge {len(oofs)} seed 平均 OOF={rmse(y, np.mean(oofs, 0)):.5f} -> {OUT_DIR}/gru_t.npz")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1]
    if mode == "data":
        if not os.path.exists(TX_CACHE):
            build_tx()
    elif mode == "train":
        train(sys.argv[2], [int(s) for s in sys.argv[3].split(",")])
    elif mode == "merge":
        merge()


if __name__ == "__main__":
    main()
