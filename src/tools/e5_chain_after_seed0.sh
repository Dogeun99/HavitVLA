#!/usr/bin/env bash
# seed 1 완주 감지 → counterfactual 배치 자동 착수 (인터넷 단절과 무관하게 진행).
# seed 2·3은 **중간 판독 후 연구원 지시** 사항이므로 자동 착수하지 않는다.
set -uo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
LOG=logs/e5/seed0_v2.log
until grep -q "E5-SEED0-EXIT" "$LOG" 2>/dev/null; do sleep 60; done
if ! grep -q "E5-DONE" "$LOG"; then
  echo "[E5-CHAIN-ABORT] seed 1이 E5-DONE 없이 종료 — CF 미착수" >> logs/e5/chain.log; exit 1
fi
echo "[E5-CHAIN] seed 1 완주 확인 → counterfactual 배치 착수 $(date +%H:%M)" >> logs/e5/chain.log
${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python} -u experiments/e5_counterfactual.py --seed-idx 0 \
  >> logs/e5/cf0.log 2>&1
echo "[E5-CHAIN-CF-EXIT-$?] $(date +%H:%M)" >> logs/e5/chain.log
echo "[E5-CHAIN] seed 2·3은 중간 판독 후 연구원 지시 대기 — 자동 착수하지 않음" >> logs/e5/chain.log
