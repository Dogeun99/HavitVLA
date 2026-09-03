#!/usr/bin/env bash
# 표준 래퍼 (연구원 지시 2026-08-16 §2): 장기 작업 + heartbeat 동반 실행 한 줄화.
# 사용: with_heartbeat.sh <plan.json> <driver.log> <until-정규식> [--milestone <pct>] -- <명령...>
set -uo pipefail
PLAN=$1; LOG=$2; UNTIL=$3; shift 3
MS=0
if [ "${1:-}" = "--milestone" ]; then MS=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
HB_PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
$HB_PY -u $HABIT2/tools/progress_heartbeat.py \
  --plan "$PLAN" --log "$LOG" --status $HABIT2/logs/progress_status.txt \
  --until "$UNTIL" --milestone-pct "$MS" &
HB_PID=$!
"$@"
RC=$?
sleep 1
kill $HB_PID 2>/dev/null
exit $RC
