#!/usr/bin/env bash
# C-T2 배치 — 2연쇄 클러스터 2개: chained_libero_object_task{0,5} (α 판정 복원, §5 2026-08-15)
# 실행기 = chunk-break 개정판(전환 시 stale 폐기 — diag5b 확증 결함 교정). 트리거 원안 유지.
# 단계: smoke 10 ep(chained_env 검증 ③) → 수집 120 ep(상대 트리거 §1-2) → 학습 n-grid → 평가 held-out 20
# 전 단계 idempotent — 완료 산출물 있으면 skip. 마커: [T2-DONE] / [T2-FAIL] / [T2-TRIGGER-FAIL]
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
mkdir -p $HABIT2/logs/e3 $HABIT2/data/e3/t2_smoke $HABIT2/results/e3

TASKS=(0 5)

echo "=== [T2] phase 0+1: smoke(10, 검증 ③) + collect(120, 트리거 §1-2) ==="
cd $HABIT2/third_party/openvla-oft
for t in "${TASKS[@]}"; do
  c=chained_libero_object_task$t
  # --- smoke: teacher가 stage 2를 실제로 수행하는가 (수집 전 필수, chained_env.py 검증 ③)
  if [ -f $HABIT2/data/e3/${c}_summary.json ] && ! grep -q '"partial": true' $HABIT2/data/e3/${c}_summary.json; then
    echo "--- $c: 수집 완료본 존재, smoke+collect skip"; continue
  fi
  echo "--- smoke $c (10 ep)"
  $OFT_PY -u $HABIT2/teacher/collector.py --suite libero_object --task $t --n 10 --chained \
    --out $HABIT2/data/e3/t2_smoke > $HABIT2/logs/e3/t2_smoke_$c.log 2>&1
  rc=$?
  [ "$rc" -ne 0 ] && { echo "[T2-FAIL] smoke collector $c exit=$rc"; exit 1; }
  $HAB_PY -u $HABIT2/experiments/e3_t2_check.py --mode smoke --task $t \
    | tee -a $HABIT2/logs/e3/t2_smoke_$c.log
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[T2-FAIL] smoke gate $c"; exit 1; }
  # --- collect 120
  echo "--- collect $c (120 ep)"
  $OFT_PY -u $HABIT2/teacher/collector.py --suite libero_object --task $t --n 120 --chained \
    --out $HABIT2/data/e3 > $HABIT2/logs/e3/t2_collect_$c.log 2>&1
  rc=$?
  grep -E "COLLECT-(PASS|PARTIAL)" $HABIT2/logs/e3/t2_collect_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[T2-FAIL] collect $c exit=$rc"; exit 1; }
  # --- 상대 트리거 (§1-2): 기대 S_V,k² 대비 단측 이항 α=0.01
  $HAB_PY -u $HABIT2/experiments/e3_t2_check.py --mode trigger --task $t \
    | tee -a $HABIT2/logs/e3/t2_collect_$c.log
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[T2-TRIGGER-FAIL] $c — 중단, 연구원 보고"; exit 1; }
done
echo "[T2-PHASE1-DONE] smoke + collection"

echo "=== [T2] phase 2: training ==="
cd $HABIT2
for t in "${TASKS[@]}"; do
  c=chained_libero_object_task$t
  if [ -f checkpoints/$c/act_n80.pt ]; then echo "--- train $c: complete, skip"; continue; fi
  echo "--- train $c"
  $HAB_PY -u habits/train.py --h5 data/e3/$c.hdf5 --cluster $c --n-grid 10 20 40 80 --out checkpoints \
    > logs/e3/t2_train_$c.log 2>&1
  rc=$?
  grep -E "TRAIN-PASS" logs/e3/t2_train_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[T2-FAIL] train $c exit=$rc"; exit 1; }
done
echo "[T2-PHASE2-DONE] training"

echo "=== [T2] phase 3: held-out 20 evaluation (chained 대역, paired) ==="
for t in "${TASKS[@]}"; do
  c=chained_libero_object_task$t
  if [ -f results/e3/${c}_curve.json ]; then echo "--- eval $c: complete, skip"; continue; fi
  echo "--- eval $c"
  $HAB_PY -u habits/evaluate.py --cluster $c --suite libero_object --task $t --chained \
    --ckpt-dir checkpoints/$c --n-heldout 20 --out results/e3 \
    > logs/e3/t2_eval_$c.log 2>&1
  rc=$?
  grep -E "EVAL-PASS" logs/e3/t2_eval_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[T2-FAIL] eval $c exit=$rc"; exit 1; }
done
echo "[T2-PHASE3-DONE] evaluation"

echo "=== [T2] phase 4: covariates 갱신 (chained 편입 → COMPLETE(27)) ==="
$HAB_PY -u experiments/e3_covariates.py > logs/e3/t2_covariates.log 2>&1
grep -E "COVARIATES" logs/e3/t2_covariates.log | tail -1

echo "[T2-DONE] 27/27 — next: experiments/e3_collect.py (§6 집계·H2 판정)"
