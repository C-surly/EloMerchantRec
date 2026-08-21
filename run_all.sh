#!/usr/bin/env bash
# 主链路:从原始数据一路跑到 31 元特征终局融合(submission/submission_v14_repro.csv)。
# 双卡机约 5-6 小时,单卡顺跑约 7 小时;产物全部落在仓内 outputs/。
set -euo pipefail
cd "$(dirname "$0")"

export ELO_SEED="${ELO_SEED:-777}"
NN_DEVICE="${ELO_NN_DEVICE:-0}"
TRF_DEVICE="${ELO_TRF_DEVICE:-$NN_DEVICE}"
PY="${ELO_PYTHON:-python}"

t0=$(date +%s)
step() { echo "" >&2; echo "===== [+$(( ($(date +%s) - t0) / 60 ))min] $* =====" >&2; }

# 路径与原始数据自检(缺件立刻失败,不等跑到一半)
$PY src/paths.py
$PY -c "import sys; sys.path.insert(0, 'src'); import paths; paths.check_raw()"

trf_pid=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "${trf_pid}" ]] && kill -0 "${trf_pid}" 2>/dev/null; then
    if [[ ${code} -ne 0 ]]; then kill "${trf_pid}" 2>/dev/null || true; fi
    wait "${trf_pid}" 2>/dev/null || true
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

step "1/7 主管线:285 维特征 + 特征筛选 + 基础世代 4 GBDT"
$PY src/elo_pipeline.py

step "2/7 异构基学习器:MLP ×2 + ExtraTrees"
$PY src/hetero.py mlp
$PY src/hetero.py mlp2
$PY src/hetero.py et

step "3/7 TE 世代:折外 outlier 率目标编码 36 列 × 4 GBDT"
$PY src/target_encoding.py all

step "4/7 TD 世代:交易级时间差分布 30 列 × 4 GBDT"
$PY src/timediff.py all

step "5/7 FM 世代:target 公式形状特征 16 列 × 4 GBDT"
$PY src/formula.py all

step "6/7 NN 序列族:GRU / GRU-X / Transformer(各 5 seed)"
$PY src/seq_gru.py data
$PY src/seq_nn.py data_ext
if [[ "${TRF_DEVICE}" != "${NN_DEVICE}" ]]; then
  $PY src/seq_nn.py train trf "${TRF_DEVICE}" &   # 双卡:Transformer 与 GRU 链路并行
  trf_pid=$!
  $PY src/seq_nn.py train gru   "${NN_DEVICE}"
  $PY src/seq_nn.py train gru_x "${NN_DEVICE}"
  wait "${trf_pid}"; trf_pid=""
else
  $PY src/seq_nn.py train gru   "${NN_DEVICE}"
  $PY src/seq_nn.py train gru_x "${NN_DEVICE}"
  $PY src/seq_nn.py train trf   "${TRF_DEVICE}"
fi

step "7/7 F31 终局融合(BayesianRidge 二层 + 折内 isotonic 校准)"
$PY src/fuse_final.py

step "主链路完成 → submission/submission_v14_repro.csv"
echo "接着跑第六名终局链路:bash run_rank6.sh" >&2
