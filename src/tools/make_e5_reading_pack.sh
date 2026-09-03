#!/usr/bin/env bash
# E5 중간 판독 패키지 생성 (판정자 전달용).
# CF 미완이면 1차(스트림 단독), 완료면 완성본(H4b 포함)이 그대로 담긴다.
# 사용: bash tools/make_e5_reading_pack.sh [seed_idx]
set -euo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$HABIT2"
S="${1:-0}"
PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
# 패키지는 프로젝트 루트에 둔다(기존 pack 관행). gitignore 대상.
STAMP=$(date '+%Y%m%d')
PK="$HABIT2/e5_reading_pack_s${S}_${STAMP}"
rm -rf "$PK"; mkdir -p "$PK/results" "$PK/prereg"

$PY -u experiments/e5_analyze.py --seed-idx "$S" >/dev/null

$PY -u experiments/fig_e5_reading.py --seed-idx "$S" >/dev/null
$PY -u experiments/e5_ineligible_postmortem.py --seed-idx "$S" >/dev/null

# CLAUDE.md §6 패키지 규정: ①REPORT ②원자료 ③그림 ④스크립트 ⑤사전등록·log 전문 ⑥git 이력
mkdir -p "$PK/figures" "$PK/scripts" "$PK/raw"
cp results/e5/reading_${S}.json results/e5/summary_${S}.json "$PK/results/"
[ -f results/e5/cf_summary_${S}.json ] && cp results/e5/cf_summary_${S}.json "$PK/results/"
[ -f results/e5/cf_determinism_${S}.json ] && cp results/e5/cf_determinism_${S}.json "$PK/results/"
cp results/e1/e1_latency.json results/e1/e1_sv.json "$PK/results/"
cp results/e5/ineligible_postmortem_${S}.json results/e3/e3_curves.json "$PK/results/"
# 정규화 계약 증거 = 런타임 단언 로그 (formation_gap 진단은 무효 실행 전용이라 미포함)
grep -E "GATE-PASS|GATE-[ABC]-FAIL" logs/e5/seed${S}.log 2>/dev/null > "$PK/results/runtime_gate_assertions.txt" || \
  grep -E "GATE-PASS|GATE-[ABC]-FAIL" logs/e5/seed0_v3.log 2>/dev/null > "$PK/results/runtime_gate_assertions.txt"
cp results/e5/fig_e5_s${S}_*.png "$PK/figures/"
cp experiments/e5_analyze.py experiments/e5_driver.py experiments/e5_counterfactual.py \
   experiments/fig_e5_reading.py experiments/e5_ineligible_postmortem.py \
   habits/train.py \
   gates/two_stage.py "$PK/scripts/"
# 원자료: 스트림 전량(판정자가 독립 재산출할 수 있어야 한다) + CF 진행분
gzip -c results/e5/stream_${S}.jsonl > "$PK/raw/stream_${S}.jsonl.gz"
[ -f results/e5/cf_${S}.jsonl ] && gzip -c results/e5/cf_${S}.jsonl > "$PK/raw/cf_${S}.jsonl.gz"
gzip -c results/e5/cf_queue_${S}.jsonl > "$PK/raw/cf_queue_${S}.jsonl.gz"
cp configs/preregistration.md log.md "$PK/prereg/"
git log --oneline -25 > "$PK/prereg/git_history.txt"
git log -1 --format="%H %ci %s" >> "$PK/prereg/git_history.txt"

$PY - "$S" "$PK" <<'EOF'
import json, os, sys
s, pk = sys.argv[1], sys.argv[2]
d = json.load(open(f"results/e5/reading_{s}.json"))
o = d["overview"]; h4a = d["H4a_call_rate_reduction"]; h4b = d["H4b_noninferiority"]
rc = d["risk_control"]; fm = d["formation_ledger"]; md = d["maturity_dual_report"]
sj = d["shadow_jurisdiction_counterfactual"]; dem = d["demotions"]
tail = d["r_V_tail_decomposition"]; nv = d["novel_injection"]
cf_done = h4b.get("cf_complete", False)
L = []
A = L.append
A(f"# E5 seed {int(s)+1} 중간 판독 — 판정 요청\n")
A(f"상태: **{'완성본 (CF 포함)' if cf_done else '1차 (스트림 단독, CF 진행 중)'}**\n")
A("## 0. 한 줄\n")
A(f"4,000 ep 스트림에서 VLA 호출률이 {h4a['p_first']:.3f} → {h4a['p_last']:.3f}로 감소했고"
  f"(단측 p={h4a['p_report']}, **{h4a['verdict']}**), 발화 위험은 "
  f"Pr(fail|fire)={rc['pr_fail_given_fire']:.4f} ≤ ε={rc['epsilon']}로 상한 내에 있다.\n")
A("## 1. 사전등록 판정 규칙 대비 결과\n")
A("| 항목 | 규칙 (§) | 결과 | 판정 |")
A("|---|---|---|---|")
A(f"| H4-a 호출률 감소 | 첫/끝 1,000 ep 단측 (§1) | {h4a['p_first']:.3f} → {h4a['p_last']:.3f} "
  f"(Δ{h4a['diff']:.3f}, z={h4a['z']}, p={h4a['p_report']}) | **{h4a['verdict']}** |")
if cf_done:
    A(f"| H4-b 비열등 | paired bootstrap 95% CI, margin −3%p (§1) | system {h4b['system_rate']:.4f} vs "
      f"full-VLA {h4b['full_vla_rate']:.4f}, Δ={h4b['diff']:+.4f}, CI [{h4b['ci95'][0]:+.4f}, "
      f"{h4b['ci95'][1]:+.4f}] | **{h4b['verdict']}** |")
else:
    A(f"| H4-b 비열등 | paired bootstrap 95% CI, margin −3%p (§1) | CF 배치 진행 중 "
      f"({h4b.get('n_cf_missing','?')}건 미완) | **보류** |")
A(f"| 위험 통제 | Pr(fail\\|fire) ≤ ε=0.2 (§3.5) | {rc['pr_fail_given_fire']:.4f} "
  f"CI {rc['ci95_wilson']} | **{'PASS' if rc['within_bound'] else 'FAIL'}** |")
A("")
A("## 2. 형성 (H1의 스트림판)\n")
gl = " / ".join(f"n={k}: {v['passed']}/{v['attempts']} 통과({v['pass_rate']:.3f})"
                for k, v in fm['by_grid_n'].items())
A(f"- 재학습 {fm['n_retrain']}회, probe 통과 {fm['n_passed']}회 (통과율 {fm['pass_rate']:.3f}) — {gl}")
A(f"- 성숙 도달 {md['n_reached_maturity']}/{md['n_clusters']} 클러스터, "
  f"소요 노출 중앙값 **{md['exposures_to_maturity_median']}회** (범위 {md['exposures_to_maturity_range']})")
A(f"- 형성 장부: {fm['formation_wall_s']/3600:.2f}h / {fm['formation_episodes']} ep "
  f"(**지연 주장에 불산입** — §4h 3장부)")
A(f"- 최종 상태 분포: " + ", ".join(
    f"{st} {sum(1 for v in d['lifecycle'].values() if v['final_state']==st)}" for st in "MIX") + "\n")
A("## 3. 그림자 관할 반사실 — 사전 예측치 대조 (§5 등재)\n")
A("| 지표 | 사전 예측 | 실측 |")
A("|---|---|---|")
pv = sj["prediction_vs_observed"]
A(f"| VLA 라우팅 증가 | +{pv['routing_increase_pp']['predicted']}%p | "
  f"**+{pv['routing_increase_pp']['observed']}%p** |")
A(f"| 질의 지연 배수 | {pv['latency_ratio']['predicted']}× | "
  f"**{pv['latency_ratio']['observed']}×** ({sj['observed']['query_latency_off_ms']} → "
  f"{sj['observed']['query_latency_on_ms']} ms) |")
dg = sj["divergence_diagnosis"]
A(f"\n**격차 원인 규명**: 예측치는 E4-R **원 보정** q의 기각률 {dg['e4r_reject_rate_at_w001_original_q']}에서 "
  f"유도됐으나, E5 그림자는 사전등록(§5)이 지정한 **재보정** q를 썼다. 실측 발화 기각률 "
  f"{dg['observed_stream_reject_rate']:.4f}는 재보정 FR {dg['recalibrated_fr_reference'].get('mean_fr')}"
  f"(원 {dg['recalibrated_fr_reference'].get('mean_fr_before')})와 정합한다 — "
  f"**실행이 사전등록을 따랐고, 예측치 유도가 그 조항과 불일치했다.**\n")
if cf_done and "conditional_gain_per_ep" in sj["observed"]:
    A(f"관할 ON 성공률 반사실: 조건부 이득 **{sj['observed']['conditional_gain_per_ep']:+.4f}/ep** "
      f"(예측 +{sj['prereg_prediction']['conditional_gain_per_ep']}/ep), "
      f"성공률 {o['system_success_rate']:.4f} → {sj['observed']['success_rate_on']:.4f}\n")
pm = json.load(open(f"results/e5/ineligible_postmortem_{s}.json"))
cdg = pm["rule2_batch_vs_stream_gap"]["cause_diagnosis"]
mg, ct = cdg["maturity_criterion_gap"], cdg["round2_carryover_trap"]
rc = cdg["per_cluster_probe_reconstruction"]
A("## 4. 부적격(X) 클러스터 사후 분석 — **탐색적** (§5 2026-08-17 등재)\n")
A(f"후반 VLA 호출의 **{pm['share_of_tail_vla_calls']:.1%}**가 부적격 {pm['n_ineligible']}개 클러스터에서 발생한다. "
  f"추가 rollout 없이 로그만으로 산출했다.\n")
A(f"- **규칙 1**: {pm['rule1_retry_would_have_helped']['n_clusters']}/{pm['n_ineligible']}개가 "
  f"부적격 확정 후에도 BC 풀이 마지막 grid(80)를 넘겨 축적됐다 "
  f"(중앙 잉여 **+{pm['rule1_retry_would_have_helped']['median_surplus']}**) — "
  f"재도전 규칙이 있었다면 재학습 자격이 있었을 클러스터 수.")
A(f"- **규칙 2**: E3 배치에서 N*≤80이던 클러스터 **{pm['rule2_batch_vs_stream_gap']['n_clusters']}/{pm['n_ineligible']}개**가 "
  f"스트림에서는 부적격이 됐다 → 배치·스트림 형성 조건 차이가 실재한다.\n")
A("### 원인 규명 (사후 진단)\n")
A(f"**발견 ① 성숙 문턱의 이름 충돌.** E3 성숙은 점추정 `ŝ≥0.8`이고 E5 성숙은 "
  f"`Pr(s≥0.8|D)≥0.9`다. P=20에서 후자는 **{mg['e5_required_probe_successes_round1']} "
  f"(={mg['e5_implied_success_rate_round1']})**를 요구한다 — 같은 τ를 쓰지만 실질 문턱이 0.80 대 0.95로 다르다.\n")
A(f"**발견 ② 라운드2 이월 함정.** c={ct['c_reinit']} 재초기화가 라운드1 실패를 φ로 승계하므로, "
  f"**라운드1 실패가 {ct['f1_threshold_for_unwinnable']}회 이상이면 라운드2에서 20/20 만점을 받아도 Pr<0.9**다. "
  f"즉 라운드2 시작 시점에 X가 이미 확정돼 있다. 해당 **{ct['n_unwinnable']}/{pm['n_ineligible']}개**이며, "
  + (f"그중 `{ct['scored_full_marks_in_round2_but_failed'][0]}`는 **실제로 라운드2 만점(20/20)을 "
     f"받고 탈락**했다.\n" if ct['scored_full_marks_in_round2_but_failed']
     else "다만 라운드2에서 만점을 받고도 탈락한 사례는 **없다**.\n"))
A("| 클러스터 | E3 ŝ(80) | 스트림 probe r1 → r2 | 라운드2 사전 확정 |")
A("|---|---|---|---|")
for c, v in sorted(rc.items(), key=lambda x: -(x[1]["best_probe_rate"] or 0)):
    rr = {x["round"]: x for x in v["rounds"]}
    f = lambda k: f"{rr[k]['probe_successes']}/20" if k in rr else "—"
    A(f"| {c.replace('libero_','')} | {v['e3_heldout_at_80']} | {f(1)} → {f(2)} | "
      f"{'**예**' if v['round2_was_unwinnable'] else '아니오'} |")
A("")
# 형성 부진(스트림 probe가 E3 held-out보다 낮음) vs 판정 탈락(E3 수준인데 문턱 미달)을
# 프로그래밍 분류한다 — 수치·클러스터명 수동 입력 금지(CLAUDE.md §6).
near = [(c, v) for c, v in rc.items()
        if v["e3_heldout_at_80"] and v["best_probe_rate"] >= v["e3_heldout_at_80"] - 0.05]
poor = [(c, v) for c, v in rc.items()
        if v["e3_heldout_at_80"] and v["best_probe_rate"] < v["e3_heldout_at_80"] - 0.05]
fmt = lambda xs: ", ".join(f"{c.replace('libero_','')}(스트림 {v['best_probe_rate']:.2f} vs E3 "
                           f"{v['e3_heldout_at_80']:.2f})" for c, v in sorted(xs))
A(f"**형성 부진과 판정 탈락은 구분된다.** 스트림 probe 최고 성적이 E3 held-out과 같은 수준"
  f"(±0.05 이내)인데 탈락한 클러스터 **{len(near)}개**: {fmt(near) or '없음'}. "
  f"형성 자체가 부진했던 클러스터 **{len(poor)}개**: {fmt(poor) or '없음'}.\n")
na = sum(1 for _ in open(f"{pk}/results/runtime_gate_assertions.txt")) if os.path.exists(f"{pk}/results/runtime_gate_assertions.txt") else 0
nf = sum(1 for l in open(f"{pk}/results/runtime_gate_assertions.txt") if "FAIL" in l) if na else 0
A("## 5. 정규화 계약 — 런타임 단언 (B-2)\n")
A(f"재학습마다 `assert_retrain_contract()`가 세 항목을 검증하며, 위반 시 즉시 정지한다. "
  f"본 실행에서 **{na - nf}건 통과 · {nf}건 실패**.\n")
A("- **(a)** 체크포인트의 stats == 자기 학습 데이터(`episodes[:n]`)에서 재산출한 값 (l2 상대차 ≤ 1e-6) "
  "→ warm-start가 정규화 공간을 가로지르지 않음")
A("- **(b)** 학습 스텝 == 배치 등가 지정값 (n=20 → 10,000 / n=80 → 28,000)")
A("- **(c)** `|B_k|` == 참조 HDF5 meta 성공 수 == episodes 그룹 수 (3중 대조)")
A("")
A("> 무효 실행(2026-08-17)의 `formation_gap` 진단은 **B-2 실행에 적용하지 않는다** — B-2는 scratch "
  "학습이라 재학습 간 stats가 다른 것이 정상이므로, 그 진단의 (c) 판정을 돌리면 정상 동작을 "
  "결함으로 오판한다. 원본 진단은 `results/e5/seed0_normstats_invalid/INVALID_formation_gap_0.json`에 "
  "보존되어 있다.\n")
A("## 6. 판정자 확인 요청 사항\n")
A(f"**(1) 후반 r_V 정체의 해석.** 마지막 1,000 ep의 r_V {tail['r_V_observed']:.3f} 중 "
  f"X(부적격) {tail['share_of_window'].get('X',0):.3f} + I(미성숙) {tail['share_of_window'].get('I',0):.3f}. "
  f"모든 적격 클러스터가 성숙해도 습관 담당 상한은 {tail['ceiling_if_all_eligible_matured']:.4f}다. "
  f"이 분해를 **사후 분해로 명시**해 보고하는 것이 맞는지 (판정 근거는 사전등록 단측 검정 단독).\n")
ps = ", ".join("%.3f" % e["p_ge_tau"] for e in dem["events"])
A(f"**(2) 성숙 초기 취약성 — 신규 발견 등재 여부.** 강등 {dem['n_demotions']}건이 모두 동일 패턴이다: "
  f"발화 {dem['median_fires_before_demotion']}회 중 1회 실패 → ACI τ 상향 → p가 1−δ 바로 아래로 "
  f"({ps}) → 강등. "
  f"재초기화 c=0.25가 σ를 압축하므로 성숙 직후 구간이 구조적으로 취약하다. "
  f"{dem['n_regained']}건만 재형성에 성공했다. 이를 **신규 발견으로 등재**할지.\n")
A(f"**(3) R_max 소진 후 영구 정체.** X {sum(1 for v in d['lifecycle'].values() if v['final_state']=='X')}개는 "
  f"라운드 1·2 모두 probe 미통과로 확정됐고, 강등 후 R_max가 소진된 클러스터는 BC 풀이 "
  f"{max(v['final_bc_pool'] for v in d['lifecycle'].values() if v['final_state']=='I')}까지 쌓여도 "
  f"재학습이 불가능하다(§4h 결정 4의 의도된 귀결). 이 비용을 Discussion에 명시할지.\n")
A(f"**(4) novel 주입 분리 보고.** 주입 {nv['n_novel']}건(r_V {nv['novel_r_V']:.3f} vs 정규 "
  f"{nv['regular_r_V']:.3f})에서 {len(nv['novel_clusters_matured'])}개 클러스터가 성숙에 도달했다"
  f"({', '.join(nv['novel_clusters_matured']) or '없음'}). 별도 절로 보고할지.\n")
A(f"**(5) seed 2·3 착수 여부** — 본 판독으로 설계 변경 없이 진행해도 되는지.\n")
A("## 7. 패키지 구성 (CLAUDE.md §6)\n")
A("| 경로 | 내용 |")
A("|---|---|")
A(f"| `results/reading_{s}.json` | **판독 수치의 단일 진입점** — 본 문서·그림의 모든 수치는 여기서 프로그래밍 산출 |")
A(f"| `results/summary_{s}.json` | 드라이버 원 산출 (3장부·최종 상태) |")
A("| `results/e1_latency.json`, `e1_sv.json` | 지연 앵커·teacher 성공률 (attn=sdpa, RTX 5090) |")
A("| `figures/*_behavior.png` | r_V 궤적·시스템 성공률·lifecycle 구성 |")
A("| `figures/*_mechanism.png` | 형성 probe 결과·강등 사후확률 궤적·그림자 관할 대조 |")
A(f"| `raw/stream_{s}.jsonl.gz` | **스트림 전량 4,000행** — 독립 재산출용 |")
A(f"| `raw/cf_{s}.jsonl.gz`, `cf_queue_{s}.jsonl.gz` | counterfactual 결과·큐 |")
A(f"| `results/ineligible_postmortem_{s}.json` | 부적격 사후 분석 (탐색적) |")
A("| `results/runtime_gate_assertions.txt` | 정규화 계약 런타임 단언 전 기록 |")
A("| `results/e3_curves.json` | E3 배치 곡선·N* (규칙 2 대조 원본) |")
A("| `scripts/` | 판독기·드라이버·CF·그림·게이트 구현 |")
A("| `prereg/preregistration.md` | **사전등록 전문** (§5 변경 이력 포함) |")
A("| `prereg/log.md` | 작업 로그 전문 |")
A("| `prereg/git_history.txt` | git 이력 25건 |")
A("")
A("## 8. 독립 재산출\n")
A("```")
A(f"gunzip raw/stream_{s}.jsonl.gz && python scripts/e5_analyze.py --seed-idx {s}")
A("```")
A("판독기는 사전등록에 등재된 규칙만 집행하며, 예측치도 하드코딩이 아니라 "
  "`preregistration.md` §5 원문에서 정규식으로 추출한다.")
open(f"{pk}/README.md", "w").write("\n".join(L))
EOF

tar czf "${PK}.tar.gz" -C "$(dirname "$PK")" "$(basename "$PK")"
echo "[E5PACK] ${PK}.tar.gz"
head -30 "$PK/README.md"
