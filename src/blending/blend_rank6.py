# -*- coding: utf-8 -*-
"""第六名终局合成:F1 = 0.6 * U2 + 0.4 * SC5。

输入(由 src/blending/pool_union.py 与 src/blending/pool_sc5.py 现场重算):
    outputs/v39/submission_v39b_union.csv   U2
    outputs/v39/submission_v39_best.csv     SC5
输出:
    submission/submission_rank6_3.59428.csv.gz

校验:与 artifacts/f1_pred.npy(私榜 3.59428 对应的冻结向量)按 card_order 对齐比对。
参考向量只用于**核对**,不参与任何计算 —— 删掉 artifacts/ 也照样能合成提交。

用法:
    python src/blending/blend_rank6.py                   # 合成 + 校验
    ELO_F1_TOL=1e-6 python src/blending/blend_rank6.py   # 放宽数值一致判定
"""
from __future__ import annotations

import gzip
import hashlib
import os
import sys

import numpy as np
import pandas as pd

# 允许 `python src/<子目录>/xxx.py` 直接执行:先把 src/ 挂进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

paths.bootstrap()

W_U2, W_SC5 = 0.6, 0.4
EXACT_TOL = 1e-12                                        # 逐位一致
NUMERIC_TOL = float(os.environ.get("ELO_F1_TOL", 1e-9))  # 浮点噪声一致


def build() -> pd.DataFrame:
    for p, who in ((paths.SC5_CSV, "SC5"), (paths.U2_CSV, "U2")):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"缺少 {who}: {p};请先运行 bash run_rank6.sh(或对应的 pool 脚本)"
            )
    sc5 = pd.read_csv(paths.SC5_CSV)
    u2 = pd.read_csv(paths.U2_CSV)
    if len(sc5) != len(u2):
        raise ValueError("SC5 / U2 行数不一致")
    if not sc5["card_id"].equals(u2["card_id"]):
        raise ValueError("SC5 / U2 card_id 顺序不一致")
    pred = (W_U2 * u2["target"].to_numpy(float)
            + W_SC5 * sc5["target"].to_numpy(float))
    print(f"[rank6] F1 = {W_U2} * U2 + {W_SC5} * SC5  n={len(pred)}", flush=True)
    return pd.DataFrame({"card_id": sc5["card_id"], "target": pred})


def verify(sub: pd.DataFrame) -> float:
    """与冻结参考向量比对,返回 maxdiff;参考文件缺失时跳过并提示。"""
    if not (os.path.exists(paths.CARD_ORDER) and os.path.exists(paths.F1_REF)):
        print("[rank6] 未找到 artifacts/ 参考向量,跳过校验", flush=True)
        return float("nan")
    order = pd.read_csv(paths.CARD_ORDER)
    ref = np.load(paths.F1_REF)
    mg = order.merge(sub, on="card_id", how="left")
    if len(mg) != len(order) or mg["target"].isna().any():
        raise ValueError("最终提交与参考卡序对齐失败")
    diff = np.abs(mg["target"].to_numpy(float) - ref)
    maxdiff, meandiff = float(diff.max()), float(diff.mean())
    print(f"[rank6] 对齐 {len(order)} 张卡  maxdiff={maxdiff:.3e}  "
          f"meandiff={meandiff:.3e}", flush=True)
    if maxdiff <= EXACT_TOL:
        print("[rank6] 与参考向量逐位一致", flush=True)
    elif maxdiff <= NUMERIC_TOL:
        print(f"[rank6] 与参考向量数值一致(浮点噪声,阈值 {NUMERIC_TOL:.1e})", flush=True)
    else:
        raise ValueError(
            f"最终结果与参考向量不一致: maxdiff={maxdiff:.3e} > {NUMERIC_TOL:.1e};"
            f"请检查 SC5/U2 的上游成员是否与冻结口径一致"
        )
    return maxdiff


def export(sub: pd.DataFrame) -> str:
    out = paths.RANK6_CSV_GZ
    os.makedirs(os.path.dirname(out), exist_ok=True)
    csv_text = sub.to_csv(index=False, float_format="%.15f")
    # mtime=0:同样的预测必然给出同样的字节,便于用哈希核对交付件
    data = gzip.compress(csv_text.encode("utf-8"), mtime=0)
    with open(out, "wb") as f:
        f.write(data)
    print(f"[rank6] 已写出 {out}", flush=True)
    print(f"[rank6] sha256 = {hashlib.sha256(data).hexdigest()}", flush=True)
    return out


def main() -> None:
    sub = build()
    verify(sub)
    export(sub)
    print("OK 3.59428", flush=True)


if __name__ == "__main__":
    main()
