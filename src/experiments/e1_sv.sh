#!/usr/bin/env bash
# E1-a: 스위트별 S_V 재측정 — 태스크당 25 ep × 40 태스크 = 1,000 ep (설계서 §5 E1).
# 공식 run_libero_eval.py 무수정, 공식 재현 프로토콜(run 수준: seed 7 고정, init_states[0..24]).
# go 기준(preregistration §1/§3): 각 스위트 S_V ≥ 0.85 / [0.75,0.85) 완화 기록 / <0.75 셀 제외 재설계.
set -uo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=$HABIT2/.hf_cache
export LIBERO_CONFIG_PATH=$HABIT2/.libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
PY=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
TRIALS=${TRIALS:-25}
SUITES=${SUITES:-"spatial object goal 10"}
LOGDIR=$HABIT2/logs/e1_sv

mkdir -p $LOGDIR
cd $HABIT2/third_party/openvla-oft

for s in $SUITES; do
  suite=libero_$s
  ckpt=moojink/openvla-7b-oft-finetuned-libero-$s
  echo "=== [E1-SV] suite=$suite trials=$TRIALS ==="
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$ckpt')" || { echo "[E1-SV-FAIL] download $ckpt"; exit 1; }
  $PY -u experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $ckpt \
    --task_suite_name $suite \
    --num_trials_per_task $TRIALS \
    --seed 7 \
    --local_log_dir $LOGDIR \
    2>&1 | tee $LOGDIR/console_$suite.log | grep -E "Task suite|Success:|Total episodes|Total successes"
  rc=${PIPESTATUS[0]:-$?}
  if [ "$rc" -ne 0 ]; then
    echo "[E1-SV-FAIL] suite=$suite exit=$rc — 집계를 진행하지 않음"
    exit 1
  fi
done

$PY -u $HABIT2/experiments/e1_sv_collect.py
