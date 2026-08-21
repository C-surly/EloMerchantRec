#!/usr/bin/env bash
# 第六名终局链路:主链路 → nn_clf → dq → tp/ct/ssl/sk → SC5 → U2 → F1。
# 默认增量运行:已存在的中间产物直接复用;ELO_FORCE=1 时强制重算 SC5/U2/F1。
# 当前仓会优先使用现场产出的成员;old_outputs 只作为仍未自举历史成员的兜底输入。
set -euo pipefail
cd "$(dirname "$0")"

export ELO_SEED="${ELO_SEED:-777}"     # 折协议纪律:两仓成员必须同 seed 同折
NN_DEVICE="${ELO_NN_DEVICE:-0}"
FORCE="${ELO_FORCE:-0}"
PY="${ELO_PYTHON:-python}"

t0=$(date +%s)
step() { echo "" >&2; echo "===== [+$(( ($(date +%s) - t0) / 60 ))min] $* =====" >&2; }
skip() { echo "----- 跳过 $1(已存在 $2)" >&2; }
# 训练类步骤永远复用已有产物(重训是小时级);只有融合层受 ELO_FORCE 控制
trained() { [[ -f "$1" ]]; }
have() { [[ "${FORCE}" != "1" && -f "$1" ]]; }

# 路径自检。
$PY src/paths.py

OUT=outputs

step "1/7 主链路(elo_pipeline → 各世代 → F31 融合)"
if trained "$OUT/base_nn/trf.npz"; then skip "主链路" "$OUT/base_nn/trf.npz"; else
  bash run_all.sh
fi

step "2/7 NN outlier 概率头(v15:10 通道序列二分类,5 seed)"
if trained "$OUT/base_nn_clf/clf.npz"; then skip "nn_clf" "$OUT/base_nn_clf/clf.npz"; else
  $PY src/archive/v15_nn_clf.py train "$NN_DEVICE"
  $PY src/archive/v15_nn_clf.py merge
fi

step "3/7 DQ 恶化轨迹世代(v16:q_lgb / q_hub / q_clf / q_clean)"
if trained "$OUT/base_dq/clf.npz"; then skip "dq" "$OUT/base_dq/clf.npz"; else
  $PY src/archive/v16_dq.py all
fi

step "4/7 终局补充成员(tp/ct/ssl/sk)"
$PY -c "import sys; sys.path.insert(0, 'src'); import os, paths; assert os.path.exists(paths.FEATURES), f'缺主特征表 {paths.FEATURES},请先跑 run_all.sh'"
if trained "$OUT/base_tp/lgb.npz"; then skip "tp_lgb" "$OUT/base_tp/lgb.npz"; else
  $PY src/archive/tp_lgb.py
fi
if trained "$OUT/base_ct/lgb.npz"; then skip "ct_lgb" "$OUT/base_ct/lgb.npz"; else
  $PY src/archive/ct_lgb.py
fi
if trained "$OUT/base_ct/clf_ct.npz"; then skip "ct_clf" "$OUT/base_ct/clf_ct.npz"; else
  $PY src/archive/ct_clf.py
fi
if trained "$OUT/tx_tensor.npz"; then skip "tx_seq 缓存" "$OUT/tx_tensor.npz"; else
  $PY src/archive/tx_seq.py data
fi
if trained "$OUT/base_nn_clf/ssl_clf.npz"; then skip "ssl_clf" "$OUT/base_nn_clf/ssl_clf.npz"; else
  $PY src/archive/ssl_clf.py
fi
if trained "$OUT/base_nn_clf/ssl_full_clf.npz"; then skip "ssl_full_clf" "$OUT/base_nn_clf/ssl_full_clf.npz"; else
  $PY src/archive/ssl_full_clf.py
fi
if trained "$OUT/base_nn_clf/ssl_dn_clf.npz"; then skip "ssl_dn_clf" "$OUT/base_nn_clf/ssl_dn_clf.npz"; else
  $PY src/archive/ssl_dn_clf.py
fi
if trained "$OUT/base_tp/pfn.npz"; then skip "tp_pfn" "$OUT/base_tp/pfn.npz"; else
  $PY src/archive/tp_pfn.py
fi
if trained "$OUT/base_sk/new_rowreg.npz"; then skip "sk_row" "$OUT/base_sk/new_rowreg.npz"; else
  $PY src/archive/sk_row.py
fi

step "5/7 SC5:跨仓成员合流(E10 + ct/tp/sk/ssl 真信号成员)"
$PY -c "import sys; sys.path.insert(0, 'src'); import os, paths; assert os.path.exists(paths.FEATURES), f'缺主特征表 {paths.FEATURES},请先跑 run_all.sh'"
if have "$OUT/v39/submission_v39_best.csv"; then
  skip "SC5" "$OUT/v39/submission_v39_best.csv"
else
  $PY src/blending/pool_sc5.py
fi

step "6/7 U2:并池终验(E10 + 历史 F31 成员 + SC5 成员)"
if have "$OUT/v39/submission_v39b_union.csv"; then
  skip "U2" "$OUT/v39/submission_v39b_union.csv"
else
  $PY src/blending/pool_union.py
fi

step "7/7 终局合成与校验:F1 = 0.6 * U2 + 0.4 * SC5"
$PY src/blending/blend_rank6.py

step "第六名链路完成 → submission/submission_rank6_3.59428.csv.gz"
