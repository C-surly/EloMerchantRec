# -*- coding: utf-8 -*-
"""仓内统一路径管理。

设计目标(发布仓不允许依赖调用者的工作目录):

1. 所有目录都以本文件位置反推仓库根,`python src/xxx.py` 与
   `python /abs/path/src/xxx.py` 在任意 cwd 下产物落点一致;
2. 每个目录都能用环境变量覆盖,方便把大文件放到别的盘;
3. `bootstrap()` 顺手把 `src/` 挂进 `sys.path`,使 `src/blending/*.py`
   可以直接执行,不再需要外部 `PYTHONPATH=src`。

约定:对外暴露的目录常量是 **字符串**(历史代码大量使用 `DIR + "_te"`
这类拼接),需要 `Path` 语义时用同名的 `*_DIR` 变量。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def _resolve(env_key: str, default: Path) -> Path:
    raw = os.environ.get(env_key)
    return Path(raw).expanduser().resolve() if raw else default


def _resolve_any(env_keys: tuple[str, ...], default: Path) -> Path:
    for key in env_keys:
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser().resolve()
    return default


DATA_DIR = _resolve("ELO_DATA_DIR", ROOT / "data")
RAW_DIR = _resolve("ELO_RAW_DIR", DATA_DIR / "raw")
PROC_DIR = _resolve("ELO_PROC_DIR", DATA_DIR / "processed")
OUT_DIR = _resolve("ELO_OUT_DIR", ROOT / "outputs")
SUB_DIR = _resolve("ELO_SUB_DIR", ROOT / "submission")
ARTIFACT_DIR = _resolve("ELO_ARTIFACT_DIR", ROOT / "artifacts")
FROZEN_DIR = _resolve_any(
    ("ELO_FROZEN_MEMBERS_DIR", "ELO_OLD_OUTPUTS_DIR"),
    ROOT / "external" / "frozen_members",
)

RAW = str(RAW_DIR)
PROC = str(PROC_DIR)
OUTPUTS = str(OUT_DIR)
SUBMISSION = str(SUB_DIR)
ARTIFACTS = str(ARTIFACT_DIR)
FROZEN_MEMBERS = str(FROZEN_DIR)

# 高频固定产物
FEATURES = str(PROC_DIR / "features.parquet")            # 主特征表
FEATURE_IMPORTANCE = str(OUT_DIR / "feature_importance.csv")
V39_DIR = OUT_DIR / "v39"                                 # SC5 / U2 落点
V39 = str(V39_DIR)

# 终局复现链路的三个约定产物(改名只需改这里)
SC5_CSV = str(V39_DIR / "submission_v39_best.csv")        # SC5:跨仓成员合流
U2_CSV = str(V39_DIR / "submission_v39b_union.csv")       # U2:并池终验
FINAL_REPRO_CSV_GZ = str(SUB_DIR / "submission_final_3.59428.csv.gz")

# 校验参考(只读,不参与训练)
CARD_ORDER = str(ARTIFACT_DIR / "card_order.csv")
F1_REF = str(ARTIFACT_DIR / "f1_pred.npy")


def out(*parts: str) -> str:
    """outputs/ 下的路径,如 out("base_td", "lgb.npz")。"""
    return str(OUT_DIR.joinpath(*parts))


def raw(*parts: str) -> str:
    """data/raw/ 下的路径。"""
    return str(RAW_DIR.joinpath(*parts))


def sub(*parts: str) -> str:
    """submission/ 下的路径。"""
    return str(SUB_DIR.joinpath(*parts))


def artifact(*parts: str) -> str:
    """artifacts/ 下的校验参考文件路径。"""
    return str(ARTIFACT_DIR.joinpath(*parts))


# SC5 / U2 可能回退读取的旧仓辅助成员。
# 当前仓若已现场产出同名文件,会优先使用 outputs/ 下的现算版本。
FROZEN_REQUIRED = [
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


def frozen_members_dir() -> Path:
    """返回冻结成员目录(仅作默认复现口径输入,不强制要求存在)。"""
    return FROZEN_DIR


def old_outputs_dir() -> Path:
    """兼容旧名。"""
    return frozen_members_dir()


def resolve_output(
    rel: str, *, allow_frozen: bool = True, require: bool = True, prefer_frozen: bool = False
) -> Path:
    """按给定优先级解析成员路径。"""
    cands = [OUT_DIR / rel]
    if allow_frozen:
        cands = [FROZEN_DIR / rel, OUT_DIR / rel] if prefer_frozen else [OUT_DIR / rel, FROZEN_DIR / rel]
    for p in cands:
        if p.exists():
            return p
    if require:
        msg = [str(OUT_DIR / rel)]
        if allow_frozen:
            msg.append(str(FROZEN_DIR / rel))
        raise FileNotFoundError(
            "缺少所需成员产物: " + " | ".join(msg)
        )
    return cands[0]


def resolve_output_dir(rel: str, *, allow_frozen: bool = True) -> Path | None:
    """按 outputs / frozen_members 顺序探测成员目录。"""
    cands = [OUT_DIR / rel]
    if allow_frozen:
        cands.append(FROZEN_DIR / rel)
    for p in cands:
        if p.is_dir():
            return p
    return None


def source_tag(path: Path) -> str:
    """把解析出的路径标记成 outputs / frozen_members / external。"""
    try:
        if path.is_relative_to(OUT_DIR):
            return "outputs"
        if path.is_relative_to(FROZEN_DIR):
            return "frozen_members"
    except AttributeError:
        s = str(path)
        if s.startswith(str(OUT_DIR)):
            return "outputs"
        if s.startswith(str(FROZEN_DIR)):
            return "frozen_members"
    return "external"


def ensure_dirs() -> None:
    """建好运行期需要写入的目录。"""
    for d in (PROC_DIR, OUT_DIR, OUT_DIR / "logs", SUB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def bootstrap() -> Path:
    """脚本入口统一调用:挂 sys.path + 建目录,返回仓库根。"""
    s = str(SRC_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)
    ensure_dirs()
    return ROOT


def check_raw() -> Path:
    """校验原始数据是否就位(train.csv 作为哨兵)。"""
    if not (RAW_DIR / "train.csv").exists():
        raise FileNotFoundError(
            f"缺少 Kaggle Elo 原始数据: {RAW_DIR}/train.csv;"
            f"可放置文件或软链,也可用 ELO_RAW_DIR 指向别处"
        )
    return RAW_DIR


if __name__ == "__main__":
    bootstrap()
    for k in ("ROOT", "RAW", "PROC", "OUTPUTS", "SUBMISSION", "ARTIFACTS"):
        print(f"{k:<12} = {globals()[k]}")
    print(f"{'FROZEN_MEM':<12} = {FROZEN_DIR}")
    print(f"{'FEATURES':<12} = {FEATURES}")
