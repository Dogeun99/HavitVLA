#!/usr/bin/env bash
# E5 상태 한 눈 보기 — 재접속 직후 1회 실행하면 전체 상황이 복원된다.
# 사용: bash tools/e5_status.sh [seed]   (생략 시 최신 seed 자동 감지)
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
S="${1:-$(ls -1 results/e5/stream_*.jsonl 2>/dev/null | sed 's/.*stream_//;s/\.jsonl//' | sort -n | tail -1)}"
S="${S:-0}"
TMUX=${HV2_TMUX:-$HOME/miniconda3/envs/hv2_tools/bin/tmux}
PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}

echo "=== seed $S · $(date '+%m-%d %H:%M') ==="
echo "--- tmux / GPU ---"
$TMUX ls 2>&1 | sed 's/^/  /'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'

echo "--- 완료된 seed ---"
for f in results/e5/cf_summary_*.json; do
  [ -f "$f" ] || continue
  i=$(echo "$f" | sed 's/.*cf_summary_//;s/.json//')
  $PY -c "
import json
r=json.load(open('results/e5/reading_$i.json')); a,b=r['H4a_call_rate_reduction'],r['H4b_noninferiority']
print(f\"  seed $i: H4a {a['p_first']}→{a['p_last']} {a['verdict']} | H4b Δ{b['diff']:+.4f} CI[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] {b['verdict']} | 위험 {r['risk_control']['pr_fail_given_fire']}\")"
done

echo "--- seed $S 진행 ---"
if [ -f "results/e5/stream_${S}.jsonl" ]; then
  $PY - "$S" <<'PYEOF'
import json, collections, sys, datetime, os
s = sys.argv[1]
rows = [json.loads(l) for l in open(f'results/e5/stream_{s}.jsonl')]
last = {}
for r in rows: last[r['cluster']] = r['lifecycle_state']
st = collections.Counter(last.values()); h = [r for r in rows if r['executor'] == 'habit']
rv = lambda x: sum(1 for r in x if r['executor'] == 'vla') / len(x)
rt = [r for r in rows if r['retrain_event']]
print(f"  ep {len(rows)}/4000 ({len(rows)/40:.0f}%) | M{st['M']} I{st['I']} X{st['X']} U{st['U']}")
print(f"  r_V 첫1000 {rv(rows[:1000]):.3f} → 최근300 {rv(rows[-300:]):.3f}" if len(rows) >= 1000
      else f"  r_V 최근300 {rv(rows[-300:]):.3f}")
print(f"  발화 {len(h)}건 (성공률 {sum(1 for r in h if r['outcome']=='success')/max(len(h),1):.3f})")
print(f"  형성 {len(rt)}회 (통과 {sum(1 for r in rt if r['retrain_event']['passed'])})")
if len(rows) < 4000:
    mt = os.path.getmtime(f'results/e5/stream_{s}.jsonl')
    print(f"  마지막 기록 {datetime.datetime.fromtimestamp(mt):%H:%M:%S}")
PYEOF
else
  echo "  (미착수)"
fi
echo "  런타임 단언: PASS $(grep -c "GATE-PASS" logs/e5/seed${S}.log 2>/dev/null; true) · FAIL $(grep -cE "GATE-[ABC]-FAIL" logs/e5/seed${S}.log 2>/dev/null; true)"

echo "--- CF / 패키지 ---"
[ -f "results/e5/cf_${S}.jsonl" ] && echo "  CF $(wc -l < results/e5/cf_${S}.jsonl)건 진행" || echo "  CF 미착수"
ls -1t e5_reading_pack_s*.tar.gz 2>/dev/null | head -3 | sed 's/^/  /'

echo "--- 체인 로그 ---"
for L in logs/e5/post_seed${S}.log logs/e5/seed2_chain.log; do
  [ -f "$L" ] && tail -2 "$L" | sed "s|^|  |"
done

echo "--- 오류 ---"
n=$(grep -cE "RuntimeError|GATE-[ABC]-FAIL|Traceback|★" logs/e5/seed${S}.log logs/e5/post_seed${S}.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
echo "  치명 오류 라인: $n (0이면 정상)"
echo "--- 다음 ---"
echo "  seed $S 완주 → 판독 → CF → 패키지 (e5post${S} 체인 자동) → 3 seed 종합"
