#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export ELO_SEED="${ELO_SEED:-777}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

NN_DEVICE="${ELO_NN_DEVICE:-0}"

need() {
  [[ -f "$1" ]]
}

[ -e data/raw/train.csv ] || { echo "请先准备 data/raw 下的 Elo 原始 CSV"; exit 1; }

if ! need outputs/base_nn/trf.npz; then
  bash run_all.sh
fi

if ! need outputs/base_nn_clf/clf.npz; then
  python src/archive/v15_nn_clf.py train "$NN_DEVICE"
  python src/archive/v15_nn_clf.py merge
fi

if ! need outputs/base_dq/clf.npz; then
  python src/archive/v16_dq.py all
fi

if ! need outputs/v39/submission_v39_best.csv; then
  python src/blending/pool_sc5.py
fi

if ! need outputs/v39/submission_v39b_union.csv; then
  python src/blending/pool_union.py
fi

python src/blending/blend_rank6.py
