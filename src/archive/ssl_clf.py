# -*- coding: utf-8 -*-
"""v27:自监督预训练 outlier 分类成员 ssl_clf。"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

import elo_pipeline as ep
import fusion as vf
from archive.tx_seq import TX_CACHE, build_tx

t0 = time.time()
L = 128
SSL_ENCODER = paths.out("nn_parts", "ssl_encoder.pt")


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    import torch
    import torch.nn as nn

    dev_id = os.environ.get("ELO_NN_DEVICE", "0")
    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(TX_CACHE):
        build_tx()
    z = np.load(TX_CACHE)
    n_mcat, n_ssec = int(z["vocab"][0]), int(z["vocab"][1])
    mc_mask, ss_mask = n_mcat, n_ssec
    num_all = np.concatenate([z["num_tr"], z["num_te"]])
    cat_all = np.concatenate([z["cat_tr"], z["cat_te"]])
    st_tr, st_te = z["st_tr"], z["st_te"]
    n_tr = len(z["num_tr"])
    base = pd.read_parquet(paths.FEATURES)
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yb = (y < -30).astype(int).to_numpy()
    log(f"张量 {num_all.shape} vocab=({n_mcat},{n_ssec}) dev={dev}")

    num = torch.from_numpy(num_all).to(dev)
    cat = torch.from_numpy(cat_all.astype(np.int32)).to(dev)
    st = torch.from_numpy(np.concatenate([st_tr, st_te])).to(dev)
    d_model = 96

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.em = nn.Embedding(n_mcat + 1, 16, padding_idx=0)
            self.es = nn.Embedding(n_ssec + 1, 8, padding_idx=0)
            self.proj = nn.Linear(8 + 24, d_model)
            self.pos = nn.Embedding(L, d_model)
            layer = nn.TransformerEncoderLayer(d_model, 4, 192, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 4)

        def forward(self, xn, xc, msk_flag):
            e = torch.cat([xn, msk_flag.unsqueeze(2),
                           self.em(xc[:, :, 0].long()), self.es(xc[:, :, 1].long())], 2)
            h = self.proj(e) + self.pos.weight.unsqueeze(0)
            pad = xn[:, :, 6] > 0.5
            return self.enc(h, src_key_padding_mask=pad), pad

    torch.manual_seed(777)
    np.random.seed(777)
    encoder = Encoder().to(dev)
    head_mc = nn.Linear(d_model, n_mcat).to(dev)
    head_amt = nn.Linear(d_model, 1).to(dev)

    pre_ep, bs = 3, 256
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head_mc.parameters())
                            + list(head_amt.parameters()), lr=1e-3, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    n_all = len(num)
    gen = torch.Generator(device=dev).manual_seed(777)
    for epn in range(pre_ep):
        perm = torch.randperm(n_all, device=dev)
        tot, tot_acc, tot_n = 0.0, 0.0, 0
        for i in range(0, n_all, bs):
            b = perm[i:i + bs]
            xn = num[b].float()
            xc = cat[b].clone()
            real = xn[:, :, 6] < 0.5
            mrand = torch.rand(real.shape, device=dev, generator=gen) < 0.15
            msk = real & mrand
            mc_true = xc[:, :, 0][msk].long()
            amt_true = xn[:, :, 0][msk]
            xn2 = xn.clone()
            xn2[:, :, :6][msk] = 0.0
            xc[:, :, 0][msk] = mc_mask
            xc[:, :, 1][msk] = ss_mask
            opt.zero_grad()
            h, _ = encoder(xn2, xc, msk.float())
            hm = h[msk]
            loss = ce(head_mc(hm), mc_true) + 5.0 * torch.mean((head_amt(hm).squeeze(1) - amt_true) ** 2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tot += float(loss) * len(hm)
                tot_acc += float((head_mc(hm).argmax(1) == mc_true).float().sum())
                tot_n += len(hm)
        log(f"[pre] epoch{epn + 1}: loss={tot / tot_n:.4f} masked-mcat-acc={tot_acc / tot_n:.4f}")
    os.makedirs(paths.out("nn_parts"), exist_ok=True)
    torch.save(encoder.state_dict(), SSL_ENCODER)
    pre_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    class ClfNet(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
            self.stat = nn.Linear(st_tr.shape[1], 32)
            self.head = nn.Sequential(
                nn.Linear(d_model + 32, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xn, xc, stx):
            h, pad = self.encoder(xn, xc, torch.zeros_like(xn[:, :, 0]))
            w = (~pad).float().unsqueeze(2)
            pool = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
            return self.head(torch.cat([pool, torch.relu(self.stat(stx))], 1)).squeeze(1)

    yb_t = torch.from_numpy(yb.astype(np.float32)).to(dev)
    pw = torch.tensor((1 - yb.mean()) / yb.mean(), device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def infer(model, idx, bs_infer=2048):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), bs_infer):
                b = idx[i:i + bs_infer]
                outs.append(torch.sigmoid(model(num[b].float(), cat[b], st[b].float())))
        return torch.cat(outs).float().cpu().numpy()

    te_idx = torch.arange(n_tr, n_all, device=dev)
    oof = np.zeros(n_tr)
    pred = np.zeros(n_all - n_tr)
    fbs, max_ep, pat = 512, 12, 3
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(777 + k)
        encoder.load_state_dict(pre_state)
        model = ClfNet(encoder).to(dev)
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": list(model.stat.parameters()) + list(model.head.parameters()), "lr": 1e-3},
        ], weight_decay=1e-5)
        tr_t = torch.from_numpy(tr).to(dev)
        va_t = torch.from_numpy(va).to(dev)
        best, wait, best_state = -1.0, 0, None
        for _ in range(max_ep):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(perm), fbs):
                b = perm[i:i + fbs]
                opt.zero_grad()
                loss = bce(model(num[b].float(), cat[b], st[b].float()), yb_t[b])
                loss.backward()
                opt.step()
            auc = roc_auc_score(yb[va], infer(model, va_t))
            if auc > best + 1e-5:
                best, wait = auc, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= pat:
                    break
        model.load_state_dict(best_state)
        oof[va] = infer(model, va_t)
        pred += infer(model, te_idx) / len(folds)
        log(f"  [ft] fold{k + 1}: AUC={best:.5f}")
    os.makedirs(paths.out("base_nn_clf"), exist_ok=True)
    np.savez(paths.out("base_nn_clf", "ssl_clf.npz"), oof=oof, pred=pred)
    auc_nn = roc_auc_score(yb, oof)
    log(f"[ssl] OOF AUC={auc_nn:.5f}(f_clf 0.90586 / v15 NN ens 0.9078 / 冠军 0.914)")

    bases = vf.load_bases()
    lgb_oof, lgb_pred = bases["f_clf"]

    def rk(a):
        return pd.Series(a).rank(pct=True).to_numpy()

    best_w, best_auc = 0.0, roc_auc_score(yb, rk(lgb_oof))
    for w in np.arange(0.05, 1.0, 0.05):
        auc = roc_auc_score(yb, (1 - w) * rk(lgb_oof) + w * rk(oof))
        if auc > best_auc:
            best_w, best_auc = w, auc
    log(f"rank 混合:w_nn={best_w:.2f} AUC={best_auc:.5f}(纯 lgb rank {roc_auc_score(yb, rk(lgb_oof)):.5f})")
    bl_oof = (1 - best_w) * rk(lgb_oof) + best_w * rk(oof)
    bl_pred = (1 - best_w) * rk(lgb_pred) + best_w * rk(pred)

    reg = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    t_reg = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    d_reg = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    f_reg = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    n_reg = sorted(k for k in bases if k.startswith("n_"))
    allf = (reg + t_reg + d_reg + f_reg + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + n_reg)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results = {}
    bases["p_ssl"] = (oof, pred)
    r1, _, _ = vf.evaluate(allf + ["p_ssl"], "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results["追加 p_ssl 成员"] = r0 - r1
    del bases["p_ssl"]
    bases["f_clf"] = (bl_oof, bl_pred)
    r2, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results["f_clf 替换为混合"] = r0 - r2
    for tag, d in results.items():
        log(f"  {tag}: Δ={d:+.5f}")
    best_d = max(results.values())
    log(f"判据[v27 ssl_clf]:基线={r0:.5f} 最优 Δ={best_d:+.5f} "
        f"{'✅ 通过' if best_d > 0.0005 else '❌ 不足'}")
    if best_d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
