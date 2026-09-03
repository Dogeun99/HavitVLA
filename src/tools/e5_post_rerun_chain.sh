#!/usr/bin/env bash
# 재실행 완주 → 판독 산출 → CF 배치 → 완성 패키지. seed 1·2는 착수하지 않는다(연구원 지시).
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
S="${1:-0}"
PYH=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
PYO=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "seed $S 완주 대기"
while [ ! -f results/e5/summary_${S}.json ]; do
  pgrep -f "e5_driver.py --seed-idx $S" >/dev/null || { log "★ 드라이버 종료했으나 summary 없음 — 정지"; exit 1; }
  sleep 180
done
log "완주 확인 → 1차 판독 산출"
$PYH -u experiments/e5_analyze.py --seed-idx "$S" >/dev/null 2>&1 && log "  판독 OK"
$PYH -u experiments/e5_ineligible_postmortem.py --seed-idx "$S" >/dev/null 2>&1 && log "  부적격 사후분석 OK"
$PYH -u experiments/fig_e5_reading.py --seed-idx "$S" >/dev/null 2>&1 && log "  그림 OK"

log "counterfactual 배치 착수 (결정성 사전 검증 포함)"
$PYO -u experiments/e5_counterfactual.py --seed-idx "$S" > logs/e5/cf${S}_run.log 2>&1
RC=$?
log "CF 종료 rc=$RC"
if [ "$RC" -ne 0 ]; then
  log "★ CF 실패 — 패키지는 스트림 단독으로 생성"
  grep -E "E5CF-DET-FAIL|E5CF-FAIL|RuntimeError" logs/e5/cf${S}_run.log | tail -3
fi
log "완성 패키지 생성"
bash tools/make_e5_reading_pack.sh "$S"
log "완료 — 다음 seed는 연구원 지시 대기"
