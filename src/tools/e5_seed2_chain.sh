#!/usr/bin/env bash
# seed 1 CF 완료 + 패키지 생성 확인 → 착수 전 확인 3종 → seed 2 착수.
# GPU 경합(OFT 7B 16.5GB × 2 > 32GB)으로 병행 불가하므로 CF 종료 즉시 승계한다.
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
PYH=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
PYO=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "seed 1 CF 완료 대기 (GPU 승계용)"
while [ ! -f results/e5/cf_summary_1.json ]; do
  pgrep -f "e5_counterfactual.py --seed-idx 1" >/dev/null || { log "★ CF 종료했으나 요약 없음 — seed 2 착수 보류"; exit 1; }
  sleep 120
done
log "CF 완료 확인 → 패키지 생성 대기"
for i in $(seq 1 30); do
  ls e5_reading_pack_s1_*.tar.gz >/dev/null 2>&1 && break
  sleep 60
done
ls e5_reading_pack_s1_*.tar.gz >/dev/null 2>&1 && log "  패키지 확인: $(ls -1t e5_reading_pack_s1_*.tar.gz | head -1)" \
                                              || log "  ★ 패키지 미생성 — 계속 진행(수동 확인 필요)"

# GPU 해제 대기
for i in $(seq 1 20); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$FREE" -gt 25000 ] && break
  sleep 30
done
log "GPU 여유 ${FREE} MiB"

# 착수 전 확인 3종 (연구원 판정 2026-08-19)
log "착수 전 확인 3종"
$PYH - <<'PYEOF' || exit 1
import sys; sys.path.insert(0,'.')
from experiments.e5_driver import BATCH_EQUIV_STEPS as B, GRID_FULL, PROBE_FULL
from envs.stream import assert_six_bands_disjoint
import inspect, experiments.e5_driver as drv
src = inspect.getsource(drv)
assert GRID_FULL == (20, 80) and PROBE_FULL == 20, "B-2 상수 불일치"
assert B[20] == 10000 and B[80] == 28000, f"배치 등가 스텝 불일치 {B}"
assert "--no-warm-start" in src and "assert_retrain_contract(" in src, "런타임 단언/scratch 미적용"
bands = assert_six_bands_disjoint(2)
print(f"  (1) B-2 상수 OK: grid={GRID_FULL} probe={PROBE_FULL} steps n=20:{B[20]:,} n=80:{B[80]:,}")
print(f"  (2) 런타임 단언·scratch OK")
print(f"  (3) 6대역 disjoint(seed 2) OK: {bands}")
PYEOF

[ -f results/e5/stream_2.jsonl ] && { log "★ seed 2 출력 경로 이미 존재 — 정지"; exit 1; }
log "seed 2 착수"
$PYO -u experiments/e5_driver.py --seed-idx 2 > logs/e5/seed2.log 2>&1
log "seed 2 종료 rc=$? — 단언 $(grep -c 'GATE-PASS' logs/e5/seed2.log)건 통과"
