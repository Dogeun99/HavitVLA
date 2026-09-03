#!/usr/bin/env bash
# C-T2 held-out 확대 (§4e·§5 이력 2026-08-15, 결과 판독 전 등재):
# 배치 완료 후 보충 평가(스펙 21–50 = start 20) → 50개 병합. [T2-DONE] 이후에만 실행.
set -uo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=$HABIT2/.hf_cache
export LIBERO_CONFIG_PATH=$HABIT2/.libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_HOME=$HABIT2/.torch_cache
HAB_PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
mkdir -p $HABIT2/results/e3/t2_h50 $HABIT2/logs/e3
cd $HABIT2

# α 판정 (§5 2026-08-15): 체인 = task0 + task5 복원 — T1 참조는 양쪽 E2 50 (h50 신설 불요).
for t in 0 5; do
  c=chained_libero_object_task$t
  [ -f results/e3/${c}_curve.json ] || { echo "[H50-FAIL] $c 기본 20 곡선 부재 — 배치 미완료"; exit 1; }
  if [ -f results/e3/t2_h50/${c}_h50supp_curve.json ]; then
    echo "--- supp $c: complete, skip"
  else
    echo "--- supp eval $c (스펙 21-50, 30 ep × n-grid)"
    $HAB_PY -u habits/evaluate.py --cluster ${c}_h50supp --suite libero_object --task $t --chained \
      --ckpt-dir checkpoints/$c --n-heldout 50 --heldout-start 20 --out results/e3/t2_h50 \
      > logs/e3/t2_h50_$c.log 2>&1
    rc=$?
    grep -E "EVAL-PASS" logs/e3/t2_h50_$c.log | tail -1
    [ "$rc" -ne 0 ] && { echo "[H50-FAIL] supp eval $c exit=$rc"; exit 1; }
  fi
done
# (task6 싱글 h50 경로는 α 판정으로 회귀 제거 — §5 이력 참조)

$HAB_PY -u experiments/e3_t2_h50_merge.py || { echo "[H50-FAIL] merge"; exit 1; }
echo "[T2-H50-DONE]"
