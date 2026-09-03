#!/usr/bin/env bash
# E0-5: 스위트별 10-ep 스모크 (태스크 10개 × 1 ep = 스위트당 10 ep)
# 공식 run_libero_eval.py를 무수정 사용 (설계 §2.2의 표준 구성 그대로).
# 실행 전제: E0-4a 다운로드 완료(스위트별 순차 대기는 snapshot_download 재호출로 처리).
set -uo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=$HABIT2/.hf_cache
export LIBERO_CONFIG_PATH=$HABIT2/.libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
PY=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
# 재확인(ISSUE-7) 시: TRIALS=2 SUITES="10" 처럼 override — trials=2는 태스크당 init_states[0..1]
# 사용이라 1차(trials=1, init_states[0])와 달리 추가 표본을 실제로 공급한다.
TRIALS=${TRIALS:-1}
SUITES=${SUITES:-"spatial object goal 10"}

mkdir -p $HABIT2/logs/e0_5
cd $HABIT2/third_party/openvla-oft

for s in $SUITES; do
  suite=libero_$s
  ckpt=moojink/openvla-7b-oft-finetuned-libero-$s
  echo "=== [E0-5] suite=$suite ckpt=$ckpt trials=$TRIALS ==="
  # 체크포인트 준비 대기 (이미 캐시면 즉시 반환, 다운로드 중이면 이어받아 완료)
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$ckpt')" || { echo "[E0-5-FAIL] download $ckpt"; exit 1; }
  $PY -u experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $ckpt \
    --task_suite_name $suite \
    --num_trials_per_task $TRIALS \
    --seed 7 \
    --local_log_dir $HABIT2/logs/e0_5 \
    2>&1 | tee $HABIT2/logs/e0_5/console_$suite.log | grep -E "Task suite|success rate|Success:|Total episodes|Total successes|Overall"
  # pipefail: run_libero_eval의 종료 코드가 파이프라인 코드로 전파됨 — 크래시를 침묵시키지 않는다
  rc=${PIPESTATUS[0]:-$?}
  if [ "$rc" -ne 0 ]; then
    echo "[E0-5-FAIL] suite=$suite exit=$rc — 집계를 진행하지 않음"
    exit 1
  fi
done

# 결과 집계 → JSON
$PY -u $HABIT2/experiments/e0_smoke_collect.py
