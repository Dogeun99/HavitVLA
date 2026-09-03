#!/usr/bin/env bash
# CF 배치 완료를 감지해 판독 패키지를 완성본으로 재생성한다.
# seeds 2·3은 착수하지 않는다 — 연구원 지시 대기 (§4h 결정 6).
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
while [ ! -f results/e5/cf_summary_0.json ]; do
  if ! grep -q "E5CF" logs/e5/cf0.log 2>/dev/null; then :; fi
  if ! ${HV2_TMUX:-$HOME/miniconda3/envs/hv2_tools/bin/tmux} has-session -t e5chain 2>/dev/null; then
    [ -f results/e5/cf_summary_0.json ] || { echo "[E5PACK-ABORT] CF 세션 종료했으나 요약 없음 — cf0.log 확인 필요"; exit 1; }
  fi
  sleep 120
done
echo "[E5PACK] CF 완료 감지 $(date '+%H:%M') → 완성본 패키지 생성"
bash tools/make_e5_reading_pack.sh 0
