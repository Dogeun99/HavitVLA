#!/bin/bash
# WEEKEND_RUN_STATUS를 5분마다 갱신한다. stage 사이에만 갱신되면 진행 중 상태가 낡아 보인다.
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$HABIT2"
while pgrep -f "rgb_only_rerun/run_all.py" > /dev/null; do
  ${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python} -u experiments/rgb_only_rerun/status.py > /dev/null 2>&1
  sleep 300
done
${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python} -u experiments/rgb_only_rerun/status.py > /dev/null 2>&1
