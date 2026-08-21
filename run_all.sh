#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ELO_SEED="${ELO_SEED:-777}"

NN_DEVICE="${ELO_NN_DEVICE:-0}"
TRF_DEVICE="${ELO_TRF_DEVICE:-$NN_DEVICE}"

mkdir -p data/processed outputs/logs submission
[ -e data/raw/train.csv ] || { echo "请先准备 data/raw 下的 Elo 原始 CSV"; exit 1; }

python src/elo_pipeline.py
python src/hetero.py mlp
python src/hetero.py mlp2
python src/hetero.py et
python src/target_encoding.py all
python src/timediff.py all
python src/formula.py all
python src/seq_gru.py data
python src/seq_nn.py data_ext
python src/seq_nn.py train gru "$NN_DEVICE"
python src/seq_nn.py train gru_x "$NN_DEVICE"
python src/seq_nn.py train trf "$TRF_DEVICE"
python src/fuse_final.py
