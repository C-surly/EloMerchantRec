# -*- coding: utf-8 -*-
"""v27:含被拒交易的 SSL 分类成员 ssl_dn_clf。"""
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
from extras.ssl_full_clf import build_full_static

t0 = time.time()
L = 128
DN_CACHE = paths.out("tx_tensor_dn.npz")


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_tx_dn():
    if os.path.exists(DN_CACHE):
        return
    base = pd.read_parquet(paths.FEATURES)
    cols = ["card_id", "purchase_date", "purchase_amount", "month_lag",
            "installments", "merchant_category_id", "subsector_id", "authorized_flag"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))[cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    del hist, new
    tx["card_id"] = tx["card_id"].astype(str)
    tx = tx.sort_values(["card_id", "purchase_date"], kind="mergesort").reset_index(drop=True)
    log(f"交易合并(含被拒){tx.shape}")
    tx["dg"] = (tx.groupby("card_id", observed=True)["purchase_date"].diff()
                .dt.total_seconds().fillna(0.0) / 86400.0)
    tx["rn"] = tx.groupby("card_id", observed=True).cumcount(ascending=False)
    tx = tx[tx["rn"] < L]
    mcat, _ = pd.factorize(tx["merchant_category_id"], use_na_sentinel=True)
    ssec, _ = pd.factorize(tx["subsector_id"], use_na_sentinel=True)
    tx["mcat"] = (mcat + 2).astype(np.int16)
    tx["ssec"] = (ssec + 2).astype(np.int16)
    n_mcat, n_ssec = int(tx["mcat"].max()) + 1, int(tx["ssec"].max()) + 1
    log(f"截断近 {L} 笔:{tx.shape},vocab=({n_mcat},{n_ssec}),被拒占比 {(tx['authorized_flag'] == 0).mean():.4f}")

    cards = pd.Index(base["card_id"].astype(str))
    cidx = pd.Series(np.arange(len(cards)), index=cards)
    tx["ci"] = tx["card_id"].map(cidx)
    tx = tx[tx["ci"].notna()]
    ci = tx["ci"].to_numpy(np.float64).astype(np.int32)
    pos = (L - 1 - tx["rn"]).to_numpy(np.int32)
    dow = tx["purchase_date"].dt.dayofweek.to_numpy(np.float32)
    num = np.zeros((len(cards), L, 8), np.float16)
    num[:, :, 6] = 1.0
    feats = np.stack([
        np.log1p(np.clip(tx["purchase_amount"].to_numpy(np.float64), 0, None)) / 10.0,
        np.log1p(np.clip(tx["dg"].to_numpy(np.float64), 0, None)) / 5.0,
        tx["month_lag"].to_numpy(np.float64) / 13.0,
        dow / 6.0,
        (dow >= 5).astype(np.float32),
        np.nan_to_num(tx["installments"].to_numpy(np.float64), nan=0.0) / 12.0,
        np.zeros(len(tx)),
        (tx["authorized_flag"] == 0).to_numpy(np.float64),
    ], axis=1).astype(np.float16)
    num[ci, pos] = feats
    cats = np.zeros((len(cards), L, 2), np.int16)
    cats[ci, pos, 0] = tx["mcat"].to_numpy(np.int16)
    cats[ci, pos, 1] = tx["ssec"].to_numpy(np.int16)
    is_tr = (base["is_train"] == 1).to_numpy()
    np.savez_compressed(DN_CACHE, num_tr=num[is_tr], num_te=num[~is_tr],
                        cat_tr=cats[is_tr], cat_te=cats[~is_tr],
                        vocab=np.array([n_mcat, n_ssec]))
    log(f"含被拒张量缓存 {DN_CACHE}: {num[is_tr].shape}")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    import torch
    import torch.nn as nn

    dev_id = os.environ.get("ELO_NN_DEVICE", "0")
    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    build_tx_dn()
    z = np.load(DN_CACHE)
    n_mcat, n_ssec = int(z["vocab"][0]), int(z["vocab"][1])
    mc_mask, ss_mask = n_mcat, n_ssec
    num_all = np.concatenate([z["num_tr"], z["num_te"]])
    cat_all = np.concatenate([z["cat_tr"], z["cat_te"]])
    n_tr = len(z["num_tr"])
    xtr, xte, y = build_full_static()
    folds = ep.make_folds(y)
    yb = (y < -30).astype(int).to_numpy()
    log(f"张量 {num_all.shape}(8ch 含 denied)静态 {xtr.shape} dev={dev}")

    num = torch.from_numpy(num_all).to(dev)
    cat = torch.from_numpy(cat_all.astype(np.int32)).to(dev)
    st = torch.from_numpy(np.concatenate([xtr, xte])).to(dev)
    d_model = 96

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.em = nn.Embedding(n_mcat + 1, 16, padding_idx=0)
            self.es = nn.Embedding(n_ssec + 1, 8, padding_idx=0)
            self.proj = nn.Linear(9 + 24, d_model)
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
    head_dn = nn.Linear(d_model, 1).to(dev)

    pre_ep, bs = 3, 256
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head_mc.parameters())
                            + list(head_amt.parameters()) + list(head_dn.parameters()),
                            lr=1e-3, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    bce_dn = nn.BCEWithLogitsLoss()
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
            msk = real & (torch.rand(real.shape, device=dev, generator=gen) < 0.15)
            mc_true = xc[:, :, 0][msk].long()
            amt_true = xn[:, :, 0][msk]
            dn_true = xn[:, :, 7][msk]
            xn2 = xn.clone()
            xn2[:, :, :6][msk] = 0.0
            xn2[:, :, 7][msk] = 0.0
            xc[:, :, 0][msk] = mc_mask
            xc[:, :, 1][msk] = ss_mask
            opt.zero_grad()
            h, _ = encoder(xn2, xc, msk.float())
            hm = h[msk]
            loss = (ce(head_mc(hm), mc_true)
                    + 5.0 * torch.mean((head_amt(hm).squeeze(1) - amt_true) ** 2)
                    + bce_dn(head_dn(hm).squeeze(1), dn_true))
            loss.backward()
            opt.step()
            with torch.no_grad():
                tot += float(loss) * len(hm)
                tot_acc += float((head_mc(hm).argmax(1) == mc_true).float().sum())
                tot_n += len(hm)
        log(f"[pre-dn] epoch{epn + 1}: loss={tot / tot_n:.4f} masked-mcat-acc={tot_acc / tot_n:.4f}")
    pre_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    class ClfNet(nn.Module):
        def __init__(self, enc, st_dim):
            super().__init__()
            self.encoder = enc
            self.stat = nn.Sequential(nn.Linear(st_dim, 128), nn.ReLU(), nn.Dropout(0.2))
            self.head = nn.Sequential(
                nn.Linear(d_model + 128, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xn, xc, stx):
            h, pad = self.encoder(xn, xc, torch.zeros_like(xn[:, :, 0]))
            w = (~pad).float().unsqueeze(2)
            pool = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
            return self.head(torch.cat([pool, self.stat(stx)], 1)).squeeze(1)

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
        enc = Encoder().to(dev)
        enc.load_state_dict(pre_state)
        model = ClfNet(enc, xtr.shape[1]).to(dev)
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
        log(f"  [ftD] fold{k + 1}: AUC={best:.5f}")
    os.makedirs(paths.out("base_nn_clf"), exist_ok=True)
    np.savez(paths.out("base_nn_clf", "ssl_dn_clf.npz"), oof=oof, pred=pred)
    log(f"[sslD] OOF AUC={roc_auc_score(yb, oof):.5f}(v27B 0.89217 / f_clf 0.90586 / 冠军 0.914)")

    full_path = paths.out("base_nn_clf", "ssl_full_clf.npz")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"缺少 {full_path}; 请先运行 python src/extras/ssl_full_clf.py")
    bases = vf.load_bases()
    lgb_oof, lgb_pred = bases["f_clf"]

    def rk(a):
        return pd.Series(a).rank(pct=True).to_numpy()

    zb = np.load(full_path)
    cands = {"lgb": (rk(lgb_oof), rk(lgb_pred)), "sslB": (rk(zb["oof"]), rk(zb["pred"])),
             "sslD": (rk(oof), rk(pred))}
    best_auc, best_ws = roc_auc_score(yb, cands["lgb"][0]), (1.0, 0.0, 0.0)
    for w1 in np.arange(0, 0.45, 0.05):
        for w2 in np.arange(0, 0.45 - w1 + 1e-9, 0.05):
            w0 = 1 - w1 - w2
            mix = w0 * cands["lgb"][0] + w1 * cands["sslB"][0] + w2 * cands["sslD"][0]
            auc = roc_auc_score(yb, mix)
            if auc > best_auc:
                best_auc, best_ws = auc, (w0, w1, w2)
    w0, w1, w2 = best_ws
    log(f"三方混合:w=(lgb {w0:.2f}, sslB {w1:.2f}, sslD {w2:.2f}) AUC={best_auc:.5f}")
    bl_oof = w0 * cands["lgb"][0] + w1 * cands["sslB"][0] + w2 * cands["sslD"][0]
    bl_pred = w0 * cands["lgb"][1] + w1 * cands["sslB"][1] + w2 * cands["sslD"][1]

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
    results["追加 p_sslD 成员"] = r0 - r1
    del bases["p_ssl"]
    bases["f_clf"] = (bl_oof, bl_pred)
    r2, _, pt = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results["f_clf 替换为三方混合"] = r0 - r2
    for tag, d in results.items():
        log(f"  {tag}: Δ={d:+.5f}")
    best_d = max(results.values())
    log(f"判据[v27 ssl_dn_clf]:基线={r0:.5f} 最优 Δ={best_d:+.5f} "
        f"{'✅ 通过' if best_d > 0.0005 else '❌ 不足'}")
    if best_d > 0.0005:
        sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv(paths.out("submission_v27_ssl.csv"), index=False)
        log(f"已保存 {paths.out('submission_v27_ssl.csv')}")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
