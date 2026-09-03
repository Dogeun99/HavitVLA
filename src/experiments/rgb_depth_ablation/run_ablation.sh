#!/usr/bin/env bash
# RGB-D vs RGB-only 스크리닝 ablation — Stage 1 (6 클러스터)
# 지시서 2026-08-28. 기존 결과는 절대 수정하지 않는다(별도 출력 경로).
set -euo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$HABIT2"
PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
OUT=results/rgb_depth_ablation
CK=checkpoints/rgb_only_ablation
mkdir -p "$OUT" "$CK" logs/rgb_depth_ablation

# 선정 6 클러스터 (ABLA_RGBD_CLUSTER_SELECTION.md에서 결과 산출 전 고정)
CLUSTERS=(
  "libero_object:1:libero_object_task1"
  "libero_object:0:libero_object_task0"
  "libero_goal:1:libero_goal_task1"
  "libero_goal:0:libero_goal_task0"
  "libero_spatial:1:libero_spatial_task1"
  "libero_10:0:libero_10_task0"
)
N_HELDOUT=50

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 1. RGB-only 학습 (기존 RGB-D 체크포인트는 재사용하므로 학습하지 않는다)
for c in "${CLUSTERS[@]}"; do
  IFS=: read -r suite task cl <<< "$c"
  if [ -f "$CK/$cl/act_n80.pt" ]; then log "  $cl RGB-only 학습 완료분 존재 — 건너뜀"; continue; fi
  log "RGB-only 학습: $cl"
  # ls 다중 경로는 일부 미존재 시 exit 2를 내고 pipefail이 이를 전파해 스크립트가 죽는다.
  H5=""
  for cand in "data/e3/$cl.hdf5" "data/e2/$cl.hdf5"; do
    [ -f "$cand" ] && { H5="$cand"; break; }
  done
  [ -n "$H5" ] || { log "★ 학습 데이터 없음: $cl — 정지"; exit 1; }
  $PY -u habits/train.py --h5 "$H5" --cluster "$cl" \
      --n-grid 10 20 40 80 --out "$CK" --no-depth \
      > "logs/rgb_depth_ablation/train_${cl}.log" 2>&1
  log "  완료 $(grep -c 'TRAIN-PASS' logs/rgb_depth_ablation/train_${cl}.log)"
done

# ---- 2. 평가: 두 조건을 **동일 held-out 50 스펙**에서
for c in "${CLUSTERS[@]}"; do
  IFS=: read -r suite task cl <<< "$c"
  for cond in rgbd rgb; do
    f="$OUT/${cl}_${cond}_h${N_HELDOUT}.json"
    [ -f "$f" ] && { log "  $cl/$cond 평가분 존재 — 건너뜀"; continue; }
    dir=$([ "$cond" = rgbd ] && echo "checkpoints/$cl" || echo "$CK/$cl")
    log "평가 $cond: $cl"
    $PY -u habits/evaluate.py --suite "$suite" --task "$task" --cluster "$cl" \
        --ckpt-dir "$dir" --n-heldout "$N_HELDOUT" --out "$OUT" \
        > "logs/rgb_depth_ablation/eval_${cl}_${cond}.log" 2>&1
    mv "$OUT/${cl}_curve.json" "$f"
  done
done
log "ABLATION-DONE"
