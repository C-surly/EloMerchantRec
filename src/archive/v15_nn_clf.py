# -*- coding: utf-8 -*-
"""v15:outlier 概率 NN 化 —— 10 通道序列 + 静态特征的二分类 GRU。

背景:F31 的派生特征(p_cal/ev/交互)全部依赖 f_clf(LGB 分类器)。E1 实验证明
概率头之间的 rank 集成已饱和,需要**新信息源**;序列 NN 从逐月动态直接判别
outlier,与树模型的误差方向天然异构。fusion 已预留 F32/F33 入口:
outputs/base_nn_clf/clf.npz 存在时自动做 f_clf × NN 概率 rank 集成。
产物:outputs/nn_clf_parts/clf_s<seed>.npz;平均后 outputs/base_nn_clf/clf.npz。
用法:ELO_SEED=777 python src/archive/v15_nn_clf.py train <dev_id> [seeds,逗号分隔]
     ELO_SEED=777 python src/archive/v15_nn_clf.py merge   # 平均 parts 并落盘成员
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 允许 `python src/<子目录>/xxx.py` 直接执行:先把 src/ 挂进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import seq_gru as v13
import seq_nn as v14

PARTS_DIR = paths.out("nn_clf_parts")
OUT_DIR = paths.out("base_nn_clf")
SEEDS = [777, 1777, 2777, 3777, 4777]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def train(dev_id, seeds):
    import torch
    import torch.nn as nn
    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    z = np.load(v14.SEQX_CACHE)
    xs_tr, st_tr, xs_te, st_te = z["xs_tr"], z["st_tr"], z["xs_te"], z["st_te"]
    base = pd.read_parquet(paths.FEATURES)
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    ybin = (y < -30).astype(np.float32).to_numpy()
    C, S = xs_tr.shape[2], st_tr.shape[1]
    pw = float((1 - ybin.mean()) / ybin.mean()) ** 0.5  # sqrt 级正类加权,缓解 1% 失衡
    log(f"[nn_clf] train {xs_tr.shape} dev={dev} seeds={seeds} pos_weight={pw:.2f}")

    class ClfNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.GRU(C, 96, num_layers=2, batch_first=True, dropout=0.1)
            self.stat = nn.Linear(S, 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st):
            _, hn = self.enc(xs)
            return self.head(torch.cat([hn[-1], torch.relu(self.stat(st))], 1)).squeeze(1)

    XS, ST = torch.from_numpy(xs_tr).to(dev), torch.from_numpy(st_tr).to(dev)
    Y = torch.from_numpy(ybin).to(dev)
    XSe, STe = torch.from_numpy(xs_te).to(dev), torch.from_numpy(st_te).to(dev)

    def infer(model, xs, st, bs=8192):
        model.eval()
        with torch.no_grad():
            logit = torch.cat([model(xs[i:i + bs], st[i:i + bs])
                               for i in range(0, len(xs), bs)])
            return torch.sigmoid(logit).float().cpu().numpy()

    os.makedirs(PARTS_DIR, exist_ok=True)
    BS, MAX_EP, PAT = 1024, 40, 5
    for sd in seeds:
        oof, pred = np.zeros(len(ybin)), np.zeros(len(xs_te))
        for k, (tr, va) in enumerate(folds):
            torch.manual_seed(sd + k * 101)
            np.random.seed(sd + k * 101)
            model = ClfNet().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
            tr_t = torch.from_numpy(tr).to(dev)
            best, wait, best_state = -1.0, 0, None
            for _ in range(MAX_EP):
                model.train()
                perm = tr_t[torch.randperm(len(tr_t), device=dev)]
                for i in range(0, len(perm), BS):
                    b = perm[i:i + BS]
                    opt.zero_grad()
                    loss = lossf(model(XS[b], ST[b]), Y[b])
                    loss.backward()
                    opt.step()
                auc = roc_auc_score(ybin[va], infer(model, XS[va], ST[va]))
                if auc > best + 1e-5:
                    best, wait = auc, 0
                    best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= PAT:
                        break
            model.load_state_dict(best_state)
            oof[va] = infer(model, XS[va], ST[va])
            pred += infer(model, XSe, STe) / len(folds)
        np.savez(os.path.join(PARTS_DIR, f"clf_s{sd}.npz"), oof=oof, pred=pred)
        log(f"[nn_clf] seed={sd} OOF AUC={roc_auc_score(ybin, oof):.5f}")


def merge():
    base = pd.read_parquet(paths.FEATURES)
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    ybin = (y < -30).astype(int).to_numpy()
    zs = [np.load(os.path.join(PARTS_DIR, f)) for f in sorted(os.listdir(PARTS_DIR))
          if f.endswith(".npz")]
    oof = np.mean([z["oof"] for z in zs], 0)
    pred = np.mean([z["pred"] for z in zs], 0)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
    log(f"[nn_clf] {len(zs)} seed 平均 AUC={roc_auc_score(ybin, oof):.5f} -> {OUT_DIR}/clf.npz")


if __name__ == "__main__":
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    if sys.argv[1] == "train":
        dev_id = sys.argv[2] if len(sys.argv) > 2 else "0"
        seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else SEEDS
        train(dev_id, seeds)
    elif sys.argv[1] == "merge":
        merge()
