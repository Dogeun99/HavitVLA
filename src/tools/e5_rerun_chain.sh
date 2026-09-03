#!/usr/bin/env bash
# CF 완주 → seed 0 격리 → 스모크 관문 3단언 → 통과 시에만 재실행 착수.
# 연구원 판정 2026-08-17 (B-2 확정) 집행. 미통과 시 정지·보고.
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
PY=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# CF 완주 대기. 단 이미 격리된 상태(cf_summary가 격리 디렉토리에 있음)면 대기하지 않는다 —
# 그렇지 않으면 재기동 시 무한 대기에 빠진다.
if [ ! -f results/e5/seed0_normstats_invalid/INVALID_cf_summary_0.json ]; then
  log "CF 완주 대기"
  while [ ! -f results/e5/cf_summary_0.json ]; do
    pgrep -f e5_counterfactual >/dev/null || { log "★ CF 프로세스 종료했으나 요약 없음 — 정지"; exit 1; }
    sleep 120
  done
  log "CF 완주 확인 → seed 0 격리"
else
  log "이미 격리됨 — 격리 단계 건너뜀"
fi

# --- 1. 격리 (연구원 지시 1)
D=results/e5/seed0_normstats_invalid
mkdir -p "$D"
for f in stream_0.jsonl summary_0.json cf_0.jsonl cf_queue_0.jsonl cf_summary_0.json \
         cf_determinism_0.json reading_0.json ineligible_postmortem_0.json formation_gap_0.json \
         fig_e5_s0_behavior.png fig_e5_s0_mechanism.png; do
  [ -f "results/e5/$f" ] && mv "results/e5/$f" "$D/INVALID_$f"
done
mv checkpoints/e5_s0 "$D/INVALID_checkpoints_e5_s0" 2>/dev/null
mv data/e5_s0 "$D/INVALID_data_e5_s0" 2>/dev/null
cat > "$D/README.md" <<'RM'
# ★ 무효 데이터 — 인용 금지

seed 0 최초 실행분 전량. **정규화 통계 이동 결함**(§5 2026-08-17)으로 연구원 판정에 따라 무효화됐다.

- 결함: lazy 재학습이 `--n-grid {n}` 단일값을 넘겨 정규화 통계가 재학습마다 다른 풀에서
  산출됐고, warm-start가 정규화 공간을 가로질렀다. §5 "전 n-grid 단계 동결" 조항 위반.
- 영향: 학습을 직접 훼손하며 비계통적이라 사후 보정 불가. 부적격 10건이 "온라인 형성의 성질"인지
  "결함의 산물"인지 구분 불가, 성숙 속도에 종속된 H4a 곡선도 신뢰 불가.
- **본 디렉토리의 어떤 수치도 논문·판독에 인용하지 않는다.**
- 보존 사유: 감사 추적, 그리고 CF 파이프라인의 실전 검증 기록(결정성 5/5 PASS, 일치율 ~0.97).
RM
log "격리 완료 → $D"

# --- 2. 스모크 관문 (연구원 지시 2)
# 드라이버는 출력 경로가 있으면 기동을 거부한다(덮어쓰기 금지). 이전 실행분을 먼저 치운다.
for p in results/e5/smoke data/e5_smoke checkpoints/e5_smoke; do
  [ -e "$p" ] && mv "$p" "$D/INVALID_prior_smoke_$(basename $p)_$(date +%H%M%S)"
done
log "스모크 관문 착수 (3단언: 정규화 자기풀 일치 / 스텝 지정값 / |B_k| 3중 대조)"
$PY -u experiments/e5_driver.py --seed-idx 0 --smoke > logs/e5/smoke_rerun.log 2>&1
RC=$?
PASS=$(grep -c "GATE-PASS" logs/e5/smoke_rerun.log)
FAIL=$(grep -cE "GATE-[ABC]-FAIL" logs/e5/smoke_rerun.log)
log "스모크 종료 rc=$RC · GATE-PASS $PASS건 · GATE-FAIL $FAIL건"
if [ "$RC" -ne 0 ] || [ "$FAIL" -ne 0 ] || [ "$PASS" -lt 2 ]; then
  log "★ 관문 미통과 — 재실행 착수하지 않음. logs/e5/smoke_rerun.log 확인"
  grep -E "GATE-[ABC]-FAIL|Traceback|RuntimeError" logs/e5/smoke_rerun.log | tail -5
  exit 1
fi
log "관문 통과 (재학습 $PASS회 전부 단언 통과) → seed 0 재실행 착수"

# --- 3. 재실행
$PY -u experiments/e5_driver.py --seed-idx 0 > logs/e5/seed0_v3.log 2>&1
log "재실행 종료 rc=$? — $(grep -c 'GATE-PASS' logs/e5/seed0_v3.log) 재학습 단언 통과"
