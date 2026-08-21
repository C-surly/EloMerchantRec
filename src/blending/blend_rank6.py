from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SC5_PATH = ROOT / "outputs" / "v39" / "submission_v39_best.csv"
U2_PATH = ROOT / "outputs" / "v39" / "submission_v39b_union.csv"
REF_CARD = ROOT / "artifacts" / "card_order.csv"
REF_PRED = ROOT / "artifacts" / "f1_pred.npy"
OUT_PATH = ROOT / "submission" / "submission_rank6_3.59428.csv.gz"


def build() -> pd.DataFrame:
    sc5 = pd.read_csv(SC5_PATH)
    u2 = pd.read_csv(U2_PATH)
    if len(sc5) != len(u2):
        raise ValueError("SC5 / U2 行数不一致")
    if not sc5["card_id"].equals(u2["card_id"]):
        raise ValueError("SC5 / U2 card_id 顺序不一致")
    pred = 0.6 * u2["target"].to_numpy(float) + 0.4 * sc5["target"].to_numpy(float)
    return pd.DataFrame({"card_id": sc5["card_id"], "target": pred})


def verify(sub: pd.DataFrame) -> None:
    order = pd.read_csv(REF_CARD)
    ref = np.load(REF_PRED)
    mg = order.merge(sub, on="card_id", how="left")
    if len(mg) != len(order) or mg["target"].isna().any():
        raise ValueError("最终提交与参考卡序对齐失败")
    diff = np.abs(mg["target"].to_numpy(float) - ref)
    if float(diff.max()) > 1e-12:
        raise ValueError(f"最终结果与参考向量不一致: maxdiff={float(diff.max())}")


def export(sub: pd.DataFrame) -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_text = sub.to_csv(index=False, float_format="%.15f")
    OUT_PATH.write_bytes(gzip.compress(csv_text.encode("utf-8"), mtime=0))
    return OUT_PATH


if __name__ == "__main__":
    sub = build()
    verify(sub)
    out = export(sub)
    print(out)
    print("OK 3.59428")
