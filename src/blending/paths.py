from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUTPUTS = ROOT / "outputs"
V39 = OUTPUTS / "v39"
SUBMISSION = ROOT / "submission"
DEFAULT_FROZEN_MEMBERS = ROOT / "external" / "frozen_members"
SC5_CSV = V39 / "submission_v39_best.csv"
U2_CSV = V39 / "submission_v39b_union.csv"
REQUIRED = [
    "base/lgb.npz",
    "base_te/lgb.npz",
    "base_td/lgb.npz",
    "base_fm/lgb.npz",
    "base_nn/gru.npz",
    "base_ct/lgb.npz",
    "base_ct/clf_ct.npz",
    "base_tp/lgb.npz",
    "base_tp/pfn.npz",
    "base_sk/new_rowreg.npz",
    "base_nn_clf/ssl_dn_clf.npz",
]


def bootstrap() -> Path:
    os.chdir(ROOT)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    V39.mkdir(parents=True, exist_ok=True)
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    return ROOT


def frozen_members_dir() -> Path:
    path = Path(
        os.environ.get(
            "ELO_FROZEN_MEMBERS_DIR",
            os.environ.get("ELO_OLD_OUTPUTS_DIR", DEFAULT_FROZEN_MEMBERS),
        )
    )
    missing = [rel for rel in REQUIRED if not (path / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少冻结成员目录: {path} | 缺文件: {missing[:5]}"
        )
    return path


def old_outputs_dir() -> Path:
    return frozen_members_dir()
