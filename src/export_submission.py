#!/usr/bin/env python3
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_OUTPUT = ROOT / "submission" / "submission_rank6_3.59428.csv.gz"
EXPECTED_ROWS = 123623


def build_submission() -> pd.DataFrame:
    order = pd.read_csv(ARTIFACT_DIR / "card_order.csv")
    pred = np.load(ARTIFACT_DIR / "f1_pred.npy")
    if len(order) != EXPECTED_ROWS or len(pred) != EXPECTED_ROWS:
        raise ValueError(f"unexpected row count: order={len(order)} pred={len(pred)}")
    return pd.DataFrame({"card_id": order["card_id"], "target": pred})


def build_bytes() -> bytes:
    csv_text = build_submission().to_csv(index=False, float_format="%.15f")
    return gzip.compress(csv_text.encode("utf-8"), mtime=0)


def export(output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_bytes())
    return output_path


if __name__ == "__main__":
    path = export()
    print(path)
