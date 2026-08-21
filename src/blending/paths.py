from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_OUTPUTS = ROOT / "external" / "old_outputs"
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


def old_outputs_dir() -> Path:
    path = Path(os.environ.get("ELO_OLD_OUTPUTS_DIR", DEFAULT_OLD_OUTPUTS))
    missing = [rel for rel in REQUIRED if not (path / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少旧仓辅助输出目录: {path} | 缺文件: {missing[:5]}"
        )
    return path
