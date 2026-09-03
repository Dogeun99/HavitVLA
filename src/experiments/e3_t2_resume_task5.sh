#!/usr/bin/env bash
# task5 수집 재개 (연구원 판정 2026-08-15, §5 이력 참조):
#   - 스모크 재실행 금지 (6/10 발동 = 통계적 판정 — 선택적 재시도 배제)
#   - 구속 가드 = 사전등록 상대 트리거 (단측 이항 α=0.01, p₀=0.871)
#   - 사전 약정: 트리거 발동 시 즉시 정지·보고 (재량 진행 금지)
#   - 통과 시 e3_t2_run.sh로 이관 (idempotent — 완료분 skip 후 학습·평가·covariates)
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

c=chained_libero_object_task5
if [ -f $HABIT2/data/e3/${c}_summary.json ] && ! grep -q '"partial": true' $HABIT2/data/e3/${c}_summary.json; then
  echo "--- collect $c: complete, skip"
else
  echo "--- collect $c (120 ep, 스모크 생략 — 연구원 판정 §5)"
  cd $HABIT2/third_party/openvla-oft
  $OFT_PY -u $HABIT2/teacher/collector.py --suite libero_object --task 5 --n 120 --chained \
    --out $HABIT2/data/e3 > $HABIT2/logs/e3/t2_collect_$c.log 2>&1
  rc=$?
  grep -E "COLLECT-(PASS|PARTIAL)" $HABIT2/logs/e3/t2_collect_$c.log | tail -1
  [ "$rc" -ne 0 ] && { echo "[T2-FAIL] collect $c exit=$rc"; exit 1; }
fi
$HAB_PY -u $HABIT2/experiments/e3_t2_check.py --mode trigger --task 5
[ $? -ne 0 ] && { echo "[T2-TRIGGER-FAIL] $c — 사전 약정대로 즉시 정지, 연구원 보고"; exit 1; }

exec bash $HABIT2/experiments/e3_t2_run.sh
