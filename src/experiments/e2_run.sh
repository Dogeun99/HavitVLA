#!/usr/bin/env bash
# E2 ★ 유일 치명 단계 — 형성 실증 (설계서 §5 E2)
#   C-L0    = libero_object task 0 (HP 튜닝 허용 유일 셀)
#   C-L1rep = libero_object task 5 (대표 1클러스터)
# 단계: 수집(120 ep × 2, hv2_oft) → 앵커⑤(ACT n=40 학습 1회 계측 겸 학습) →
#        n-grid 학습(hv2_hab) → held-out 50 평가(hv2_hab) → go/no-go 집계
set -uo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=$HABIT2/.hf_cache
export LIBERO_CONFIG_PATH=$HABIT2/.libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=$HABIT2/.torch_cache
OFT_PY=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
HAB_PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
mkdir -p $HABIT2/logs/e2 $HABIT2/data/e2 $HABIT2/results/e2

declare -A CLUSTERS=( ["libero_object_task0"]="0" ["libero_object_task5"]="5" )

# --- 1) 수집 (hv2_oft; cwd = openvla-oft 루트: experiments 패키지 해석) ---
cd $HABIT2/third_party/openvla-oft
for c in libero_object_task0 libero_object_task5; do
  t=${CLUSTERS[$c]}
  if [ -f $HABIT2/data/e2/${c}_summary.json ] && ! grep -q '"partial": true' $HABIT2/data/e2/${c}_summary.json; then
    echo "=== [E2] collect $c: already complete, skip ==="
    continue
  fi
  echo "=== [E2] collect $c (task=$t, 120 ep) ==="
  $OFT_PY -u $HABIT2/teacher/collector.py --suite libero_object --task $t --n 120 --out $HABIT2/data/e2 \
    2>&1 | tee $HABIT2/logs/e2/collect_$c.log | grep -E "\[.*\]|COLLECT"
  rc=${PIPESTATUS[0]:-$?}
  [ "$rc" -ne 0 ] && { echo "[E2-FAIL] collect $c exit=$rc"; exit 1; }
done

# --- 2) 학습 (hv2_hab): C-L0 먼저 (앵커⑤ = n=40 스텝 학습의 wall-clock은 train_summary에 기록됨) ---
cd $HABIT2
for c in libero_object_task0 libero_object_task5; do
  echo "=== [E2] train $c (n-grid 10 20 40 80) ==="
  $HAB_PY -u habits/train.py --h5 data/e2/$c.hdf5 --cluster $c --n-grid 10 20 40 80 --out checkpoints \
    2>&1 | tee logs/e2/train_$c.log | grep -E "===|step [0-9]*000/|TRAIN"
  rc=${PIPESTATUS[0]:-$?}
  [ "$rc" -ne 0 ] && { echo "[E2-FAIL] train $c exit=$rc"; exit 1; }
done

# --- 3) held-out 50 평가 (hv2_hab, paired) ---
for c in libero_object_task0 libero_object_task5; do
  t=${CLUSTERS[$c]}
  echo "=== [E2] eval $c (held-out 50 × n-grid) ==="
  $HAB_PY -u habits/evaluate.py --cluster $c --suite libero_object --task $t \
    --ckpt-dir checkpoints/$c --n-heldout 50 --out results/e2 \
    2>&1 | tee logs/e2/eval_$c.log | grep -E "===|n=|EVAL"
  rc=${PIPESTATUS[0]:-$?}
  [ "$rc" -ne 0 ] && { echo "[E2-FAIL] eval $c exit=$rc"; exit 1; }
done

# --- 4) go/no-go 집계 ---
$HAB_PY -u $HABIT2/experiments/e2_collect.py
