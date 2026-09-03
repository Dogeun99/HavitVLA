#!/usr/bin/env bash
# E3 배치 — 표준 25 클러스터 (C-T2 커스텀 2연쇄는 별도 검증 후 e3_t2_run.sh)
# 단계: 수집(신규 23) → 학습(신규 23; E2 2개는 재사용) → 평가(held-out 20 × n-grid, 신규 23;
#        E2 2개는 e3_collect.py에서 E2 기록 절단으로 도출)
# 전 단계 idempotent — 완료 산출물 있으면 skip (세션 중단 후 재실행 안전)
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
mkdir -p $HABIT2/logs/e3 $HABIT2/data/e3 $HABIT2/results/e3

# (suite task) — prereg §4e. object task0/task5는 E2 데이터 재사용이므로 제외.
CLUSTERS=(
  "libero_object 1" "libero_object 2" "libero_object 3" "libero_object 4"
  "libero_object 6" "libero_object 7" "libero_object 8" "libero_object 9"
  "libero_goal 0" "libero_goal 1" "libero_goal 2" "libero_goal 3" "libero_goal 4"
  "libero_goal 5" "libero_goal 6" "libero_goal 7" "libero_goal 8" "libero_goal 9"
  "libero_spatial 0" "libero_spatial 1"
  "libero_10 0" "libero_10 2" "libero_10 5"
)

echo "=== [E3] phase 1: collection (${#CLUSTERS[@]} clusters x 120 ep) ==="
cd $HABIT2/third_party/openvla-oft
for entry in "${CLUSTERS[@]}"; do
  read -r suite task <<< "$entry"
  c=${suite}_task${task}
  if [ -f $HABIT2/data/e3/${c}_summary.json ] && ! grep -q '"partial": true' $HABIT2/data/e3/${c}_summary.json; then
    echo "--- collect $c: complete, skip"; continue
  fi
  echo "--- collect $c"
  $OFT_PY -u $HABIT2/teacher/collector.py --suite $suite --task $task --n 120 --out $HABIT2/data/e3 \
    > $HABIT2/logs/e3/collect_$c.log 2>&1
  rc=$?
  grep -E "COLLECT-(PASS|PARTIAL)" $HABIT2/logs/e3/collect_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[E3-FAIL] collect $c exit=$rc"; exit 1; }
done
echo "[E3-PHASE1-DONE] collection"

echo "=== [E3] phase 2: training ==="
cd $HABIT2
for entry in "${CLUSTERS[@]}"; do
  read -r suite task <<< "$entry"
  c=${suite}_task${task}
  if [ -f checkpoints/$c/act_n80.pt ]; then echo "--- train $c: complete, skip"; continue; fi
  echo "--- train $c"
  $HAB_PY -u habits/train.py --h5 data/e3/$c.hdf5 --cluster $c --n-grid 10 20 40 80 --out checkpoints \
    > logs/e3/train_$c.log 2>&1
  rc=$?
  grep -E "TRAIN-PASS" logs/e3/train_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[E3-FAIL] train $c exit=$rc"; exit 1; }
done
echo "[E3-PHASE2-DONE] training"

echo "=== [E3] phase 3: held-out 20 evaluation ==="
for entry in "${CLUSTERS[@]}"; do
  read -r suite task <<< "$entry"
  c=${suite}_task${task}
  if [ -f results/e3/${c}_curve.json ]; then echo "--- eval $c: complete, skip"; continue; fi
  echo "--- eval $c"
  $HAB_PY -u habits/evaluate.py --cluster $c --suite $suite --task $task \
    --ckpt-dir checkpoints/$c --n-heldout 20 --out results/e3 \
    > logs/e3/eval_$c.log 2>&1
  rc=$?
  grep -E "EVAL-PASS" logs/e3/eval_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[E3-FAIL] eval $c exit=$rc"; exit 1; }
done
echo "[E3-PHASE3-DONE] evaluation"
echo "[E3-STANDARD-DONE] run e3_collect.py after C-T2 for full report"
