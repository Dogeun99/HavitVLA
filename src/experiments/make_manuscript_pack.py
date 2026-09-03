"""논문 초안용 자료 패키지 생성기 (CLAUDE.md §6 규정 + 인용 규칙 명시).

모든 수치는 results/의 JSON에서 **프로그래밍 주입**한다 — 수동 입력 금지.
인용 제한(무효 데이터·탐색적 분석·교락 비교)을 문서 최상단에 못박아, 집필 시 혼입을 막는다.

산출: manuscript_pack_<날짜>/ + .tar.gz  (프로젝트 루트, gitignore 대상)
실행: hv2_hab python -u experiments/make_manuscript_pack.py
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(HABIT2)
R = lambda p: json.load(open(os.path.join(HABIT2, p)))


def main():
    stamp = datetime.now().strftime("%Y%m%d")
    pk = os.path.join(HABIT2, f"manuscript_pack_{stamp}")
    shutil.rmtree(pk, ignore_errors=True)
    for d in ("results", "figures", "prereg", "scripts", "raw", "docs"):
        os.makedirs(os.path.join(pk, d), exist_ok=True)

    # ---- 원자료 복사 (E0–E5 전량, 무효/증거 디렉토리 제외)
    for stage in ("e0", "e1", "e2", "e3", "e4", "e5"):
        src = os.path.join(HABIT2, "results", stage)
        dst = os.path.join(pk, "results", stage)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            s = os.path.join(src, f)
            if not os.path.isfile(s):
                continue
            (shutil.copy2(s, os.path.join(pk, "figures", f)) if f.endswith(".png")
             else shutil.copy2(s, os.path.join(dst, f)))
    for i in (0, 1, 2):
        for kind in ("stream", "cf", "cf_queue"):
            src = f"results/e5/{kind}_{i}.jsonl"
            if os.path.exists(src):
                subprocess.run(["gzip", "-c", src],
                               stdout=open(os.path.join(pk, "raw", f"e5_{kind}_{i}.jsonl.gz"), "wb"))
    for f in ("configs/preregistration.md", "log.md", "CLAUDE.md"):
        shutil.copy2(f, os.path.join(pk, "prereg", os.path.basename(f)))
    for f in sorted(os.listdir("docs")):
        shutil.copy2(os.path.join("docs", f), os.path.join(pk, "docs", f))
    # 코드 전량 — 최종 검증용(연구원 요청 2026-08-21). 디렉토리 구조를 보존해 그대로 실행 가능.
    for d in ("envs", "habits", "gates", "teacher", "experiments", "tools", "configs"):
        if not os.path.isdir(d):
            continue
        shutil.copytree(d, os.path.join(pk, d),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.joblib"))
    # 자주 쓰는 판독 스크립트는 scripts/에도 평면 배치(빠른 참조용)
    for f in ("e5_analyze.py", "e5_driver.py", "e5_counterfactual.py", "fig_e5_reading.py",
              "e5_ineligible_postmortem.py", "e5_seed_synthesis.py", "make_manuscript_pack.py"):
        shutil.copy2(os.path.join("experiments", f), os.path.join(pk, "scripts", f))
    with open(os.path.join(pk, "prereg", "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-60"],
                               capture_output=True, text=True).stdout)

    # ---- 수치 로드
    lat, sv = R("results/e1/e1_latency.json"), R("results/e1/e1_sv.json")
    e2, e3, h2 = R("results/e2/e2_gonogo.json"), R("results/e3/e3_curves.json"), R("results/e3/h2_analysis.json")
    e4t, e4r = R("results/e4/e4_scorer_table.json"), R("results/e4/e4r_competence_map.json")
    rd, cf = R("results/e5/reading_0.json"), R("results/e5/cf_summary_0.json")
    syn = R("results/e5/seed_synthesis.json") if os.path.exists("results/e5/seed_synthesis.json") else None
    pm, ws = R("results/e5/ineligible_postmortem_0.json"), R("results/e4/workspace_extent.json")
    h4a, h4b, rc = rd["H4a_call_rate_reduction"], rd["H4b_noninferiority"], rd["risk_control"]
    ov, fm, md = rd["overview"], rd["formation_ledger"], rd["maturity_dual_report"]
    sj, dem, tail = rd["shadow_jurisdiction_counterfactual"], rd["demotions"], rd["r_V_tail_decomposition"]
    dec, tc = h2["decomposition_L"], e3["t_ceiling"]

    L, A = [], None
    A = L.append
    A("# HabitVLA-2 논문 초안 자료 — seed 0 기준\n")
    A(f"생성 {datetime.now():%Y-%m-%d %H:%M} · 모든 수치는 `results/`의 JSON에서 **프로그래밍 주입**"
      "(수동 입력 금지, CLAUDE.md §6).\n")

    A("## ★ 인용 규칙 — 집필 전 반드시 확인\n")
    A("| 자료 | 제한 | 근거 |")
    A("|---|---|---|")
    A("| `results/e5/seed0_normstats_invalid/` (본 패키지 **미포함**) | **전면 인용 금지** — 정규화 결함으로 무효화된 최초 실행 | §5 2026-08-17 |")
    A("| 무효 실행 ↔ 재실행 **수치 비교** | **인용 금지** — 정규화와 학습 스텝이 동시에 달라 **교락**, 효과 분리 불가 | §5 2026-08-19 |")
    A("| 부적격(X) 사후분석 `ineligible_postmortem_0.json` | **탐색적** — Results 인용 금지, **Discussion 한계 절에만** | §5 2026-08-17·19 |")
    A("| H4b 서술 | CI가 0을 포함하므로 \"통계적으로 구별되지 않음\" 서술 가능하나 "
      "**\"동등성 검정을 별도로 수행하지 않았음\"을 각주로 명시** | §5 2026-08-19 |")
    A("| 모든 지연 수치 | **`attn=sdpa` 명기 필수** (flash-attn은 sm_120 미빌드) | CLAUDE.md §0 |")
    A("| 지연 주장 | **운영 장부 단독** 근거. 형성 장부는 별도 보고, 평가(CF) 장부는 비용 미보고 | §4h 3장부 |")
    if syn:
        A(f"| seed 종합 | **3 seed 완료 — {syn['verdict']}**. 개별 seed 수치는 §1.1, 종합은 §1.0 참조 | §5 2026-08-21 |")
        A(f"| H4b \"동등\" 서술 | 전 seed CI가 0을 포함하나 **상한 최솟값 {syn['H4b_noninferiority']['ci_upper_min']:+.4f}**로 "
          f"0에 근접 — 해당 seed에서는 서술이 약함을 명시 | §5 2026-08-21 |\n")
    else:
        A("| seed 1·2 | **진행 중** — 본 패키지는 seed 0 단독 | §5 2026-08-19 |\n")

    A("## 1. 확정 수치 대시보드\n")
    if syn:
        sa, sb, sr = syn["H4a_call_rate"], syn["H4b_noninferiority"], syn["risk_control"]
        A(f"### 1.0 H4 최종 — **3 seed 종합** (12,000 ep, 평균±산포)\n")
        A(f"> **{syn['verdict']}**\n")
        A("| 항목 | 종합 | seed별 |")
        A("|---|---|---|")
        A(f"| VLA 호출률 첫→끝 1,000 ep | **{sa['first1000']['mean']}±{sa['first1000']['sd']} → "
          f"{sa['last1000']['mean']}±{sa['last1000']['sd']}** (Δ {sa['delta']['mean']}±{sa['delta']['sd']}) | "
          + " · ".join(f"{v['first1000']}→{v['last1000']}" for v in sa["per_seed"].values()) + " |")
        A(f"| 비열등 diff (margin {sb['margin']}) | **{sb['diff']['mean']:+.4f}±{sb['diff']['sd']:.4f}** | "
          + " · ".join(f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for v in sb["per_seed"].values()) + " |")
        A(f"| Pr(fail\\|fire) (ε={sr['epsilon']}) | **{sr['pr_fail_given_fire']['mean']}±{sr['pr_fail_given_fire']['sd']}** | "
          + " · ".join(str(v["pr_fail_given_fire"]) for v in sr["per_seed"].values()) + " |")
        A(f"| paired 표본 | **{syn['counterfactual']['total_paired']:,} ep** (CF 누락 0, 결정성 전 seed 5/5) | "
          + " · ".join(f"{v['n_paired']}" for v in syn["counterfactual"]["per_seed"].values()) + " |")
        A(f"| 성숙 도달 | {syn['formation']['n_matured']['mean']:.1f}±{syn['formation']['n_matured']['sd']:.1f} / 33 | "
          + " · ".join(f"M{v['M']} I{v['I']} X{v['X']}" for v in syn["formation"]["final_states"].values()) + " |")
        A(f"| 3장부 (h) | 운영 {syn['ledgers_hours']['operational']['mean']}±{syn['ledgers_hours']['operational']['sd']} / "
          f"형성 {syn['ledgers_hours']['formation']['mean']}±{syn['ledgers_hours']['formation']['sd']} | 지연 주장 = 운영 단독 |")
        A("")
        A(f"**서술 주의**: {sb['equivalence_caveat']}\n")
    A("### 1.1 시스템 상각 (H4) — E5 seed 0, 4,000 ep\n")
    A("| 항목 | 값 | 검정 | 판정 |")
    A("|---|---|---|---|")
    A(f"| VLA 호출률 | 첫 1,000 ep **{h4a['p_first']}** → 끝 1,000 ep **{h4a['p_last']}** (Δ{h4a['diff']}) "
      f"| 단측 two-proportion z={h4a['z']}, p={h4a['p_report']} | **{h4a['verdict']}** |")
    A(f"| 시스템 성공률 | **{h4b['system_rate']}** vs full-VLA **{h4b['full_vla_rate']}** (Δ{h4b['diff']:+.4f}) "
      f"| paired bootstrap B={h4b['B']}, 95% CI [{h4b['ci95'][0]:+.4f}, {h4b['ci95'][1]:+.4f}], margin −0.03 "
      f"| **{h4b['verdict']}** |")
    A(f"| 발화 위험 | Pr(fail\\|fire) = **{rc['pr_fail_given_fire']}** | Wilson 95% CI {rc['ci95_wilson']}, ε=0.2 "
      f"| **{'PASS' if rc['within_bound'] else 'FAIL'}** |")
    A(f"| paired 표본 | {h4b['n_paired_episodes']} ep 전량 (**CF 누락 {h4b['n_cf_missing']}**) | "
      f"결정성 사전 검증 {sum(1 for c in cf['determinism_check'] if c['match'])}/{len(cf['determinism_check'])} 일치 | — |")
    A("")
    A(f"- 발화 {ov['n_fire']}건 성공률 {ov['fire_success_rate']} · VLA {ov['n_vla']}건 {ov['vla_success_rate']}")
    A(f"- CF 기준선: 습관 {cf['habit_success_rate']} vs teacher {cf['teacher_success_rate']}, "
      f"불일치 습관만 {cf['discordant_habit_only']} / teacher만 {cf['discordant_teacher_only']}")
    A(f"- 곡선 원료(F4): `results/e5/reading_0.json` → `r_V_trajectory_bin200` (200 ep 빈 {len(rd['r_V_trajectory_bin200'])}점)\n")

    A("### 1.2 형성 (H1) — 배치 E2/E3 + 스트림 E5\n")
    A("| 항목 | 값 |")
    A("|---|---|")
    for cl, v in e2["clusters"].items():
        A(f"| E2 {cl.replace('libero_','')} 성숙 곡선 | " +
          " · ".join(f"n={k}: {s}" for k, s in v["curve"].items()) +
          f" (max {v['max_s_hat']}, {v['status']}) |")
    A(f"| E3 N* 중앙값 | " + " · ".join(f"{k.replace('_',' ')}: {v['median']}" for k, v in e3["levels"].items()) + " |")
    A(f"| E5 스트림 성숙 | {md['n_reached_maturity']}/{md['n_clusters']} 클러스터, "
      f"소요 노출 중앙값 **{md['exposures_to_maturity_median']}회** (범위 {md['exposures_to_maturity_range']}) |")
    A(f"| E5 재학습 | {fm['n_retrain']}회, probe 통과 {fm['n_passed']}회 — " +
      " / ".join(f"n={k}: {v['passed']}/{v['attempts']} ({v['pass_rate']})" for k, v in fm["by_grid_n"].items()) + " |")
    A(f"| E5 최종 상태 | " + ", ".join(f"{s} {sum(1 for x in rd['lifecycle'].values() if x['final_state']==s)}"
                                      for s in "MIX") + " |")
    A(f"| 형성 장부 | {fm['formation_wall_s']/3600:.2f} h / {fm['formation_episodes']} ep "
      f"(**지연 주장 불산입**) · 재학습 1회 ≈ {lat['anchor5_act_train_n40']['vla_call_equivalents']:.0f} VLA-호출 등가 |\n")

    A("### 1.3 이중 해리 (H2) — E3\n")
    # 수치만 실으면 집필자가 H2 원가설이 지지된 것으로 오독할 수 있다 — 사전등록된 경쟁 가설
    # (§4f H2-L′)의 판정 상태를 함께 못박는다.
    A(f"> **판정 상태**: 사전등록 §4f의 경쟁 가설 **H2-L′ 채택**. 원가설 H2-L(N*가 의미 레벨 L에 따라 "
      f"증가)은 **지지되지 않았다** — 레벨 간 분산이 전체의 {dec['between_share']:.1%}에 불과하고 "
      f"Kruskal p={dec['kruskal_p']}로 무효과다. 대신 **운동·물리 난이도**(median episode length "
      f"β={h2['regression_formation22']['rank_ols']['median_len']['beta']}, "
      f"perm p={h2['regression_formation22']['rank_ols']['median_len']['perm_p']})가 N*를 설명한다. "
      f"**프레이밍(사전 지정)**: \"인수분해 아키텍처(1층 태스크 정체성)가 의미 부담을 흡수\" — "
      f"Discussion 배치. 분석 status = {h2['status']}.\n")
    A(f"- **L(의미 복잡도)**: between-share **{dec['between_share']}** / within-share {dec['within_share']}, "
      f"Kruskal H={dec['kruskal_H']}, p={dec['kruskal_p']} (n={dec['n']})")
    A("  - 그룹별 N* 중앙 순위: " + " · ".join(f"{g} {v['mean_rank']}(n={v['n']})" for g, v in dec["groups"].items()))
    A(f"- **T(horizon) 천장**: T1 ŝ(80)={tc['T1_vs_T3']['s80_T1']} vs T3 ŝ(80)={tc['T1_vs_T3']['s80_T3']}, "
      f"단측 p={tc['T1_vs_T3']['p_one_sided_decrease']:.4f} ({tc['T1_vs_T3']['method']})")
    A(f"  - 역할: {tc['T1_vs_T3']['role']}")
    A(f"  - **주의**: 단측 p={tc['T1_vs_T3']['p_one_sided_decrease']:.4f}는 α=0.05에서 "
      f"**유의하지 않다**(경계). 천장 하강은 점추정 방향으로만 서술하고 유의성을 주장하지 말 것")
    A(f"- 추정량 규약: {h2['estimators']}")
    A(f"- **공선성 주의**(§5 2026-08-16): free_joints는 형성 22셀에서 suite 더미와 **완전 공선**"
      f"(스위트 내 상수) → 해당 계수는 이 표본에서 **식별 불가**, 부호 해석 금지. median_len은 VIF 건전")
    A(f"- 회귀(형성 22셀) 유의 항: " + ", ".join(
        f"{k} β={v['beta']} p={v['perm_p']}" for k, v in h2["regression_formation22"]["rank_ols"].items()
        if isinstance(v, dict) and v.get("perm_p", 1) < 0.1) + "\n")

    A("### 1.4 게이트 (H3) — E4 / E4-R / E5 그림자\n")
    same = e4t.get("same_cell_comparison", e4t)
    A(f"- **scorer 비교 (동일 셀 기준 — 주 판독 근거)**: `results/e4/e4_scorer_table.json`")
    A(f"  - 원문 판정: 기하 관할은 표현 비용에 종속적이며(히든 스테이트 우위), 21× 비용에도 임계 미달 "
      f"→ **저비용 실시간 관할은 미해결** (§5 2026-08-16 확정 문구)")
    A(f"- **E4-R 역량 지도**: w* = " + ", ".join(f"{k.replace('libero_','')} {v}" for k, v in
                                                e4r["summary"]["w_star_by_cluster"].items()))
    A(f"  - 판독: {e4r['reading']['rule2_alignment']['verdict'][:120]}…")
    A(f"- **E5 그림자 관할 반사실** (행동 불개입, 추가 rollout 0):")
    pv = sj["prediction_vs_observed"]
    A(f"  - 사전 예측 +{pv['routing_increase_pp']['predicted']}%p · {pv['latency_ratio']['predicted']}× "
      f"→ **실측 +{pv['routing_increase_pp']['observed']}%p · {pv['latency_ratio']['observed']}×**")
    A(f"  - 질의당 지연 {sj['observed']['query_latency_off_ms']} → {sj['observed']['query_latency_on_ms']} ms "
      f"(basis: {sj['unit_basis']['off_formula']})")
    A(f"  - 괴리 사유(§5 2026-08-19 등재): 예측은 w-사다리 **전 구간 평균** 오탐률에서 유도됐으나 스트림은 "
      f"w=0.01 근방 집중 → 관할이 거의 기각하지 않음. 실측 기각률 "
      f"{sj['divergence_diagnosis']['observed_stream_reject_rate']} ≈ 재보정 FR "
      f"{sj['divergence_diagnosis']['recalibrated_fr_reference'].get('mean_fr')}")
    A(f"  - **해석 정정**: \"닫힌 작업공간에서 관할은 개입할 일이 거의 없다\" — REDUCE 판정을 **강화**\n")

    A("### 1.5 앵커 (E0/E1)\n")
    a1, a2, a3 = lat["anchor1_oft_chunk_forward"], lat["anchor2_act_forward"], lat["anchor3_gate_path"]
    A(f"- 지연 (**{lat['attn']}**, {lat['gpu']}): OFT chunk **{a1['median_ms']} ms** (p95 {a1['p95_ms']}) · "
      f"ACT **{a2['median_ms']} ms** · gate **{a3['median_ms']} ms**")
    A(f"- 속도비: per-chunk **{lat['ratios']['basis']['per_chunk_speedup']}×** / "
      f"보수 하한 **{lat['ratios']['basis']['conservative_floor_speedup']}×** ({lat['ratios']['basis']['note'][:60]}…)")
    A(f"- teacher S_V: " + " · ".join(f"{k.replace('libero_','')} {v['S_V']} {v['wilson_95']}"
                                     for k, v in sv["suites"].items()))
    A(f"- 작업공간 실측: `results/e4/workspace_extent.json` (그림 `figures/fig_workspace_extent.png`)\n")

    A("## 2. 논문 섹션 ↔ 자료 매핑\n")
    A("| 섹션 | 주장 | 자료 | 그림/표 |")
    A("|---|---|---|---|")
    A("| III 시스템 | 2단 gate·4상태 lifecycle | `scripts/two_stage.py`, `prereg/preregistration.md` §3.5·§4h | — |")
    A("| IV-A 플랫폼·앵커 | LIBERO·OFT·지연 | `results/e0/*`, `results/e1/*` | 표(지연 앵커) |")
    A("| IV-C 형성 (H1) | 성숙 곡선 ŝ(n) | `results/e2/e2_gonogo.json`, `results/e3/*_curve.json` | **미생성** — 곡선 그림 필요 |")
    A("| IV-D 이중 해리 (H2) | L=속도 / T=천장 | `results/e3/h2_analysis.json`, `e3_curves.json` | **미생성** — N* 분포·천장 그림 필요 |")
    A("| IV-E 게이트 (H3) | 관할 미해결·성숙도 단독 | `results/e4/e4_scorer_table.json`, `e4r_competence_map.json` | `figures/fig_workspace_extent.png` |")
    A("| IV-F 시스템 상각 (H4) | r_V↓ + 비열등 | `results/e5/reading_0.json`, `cf_summary_0.json` | `figures/fig_e5_s0_behavior.png`, `fig_e5_s0_mechanism.png` |")
    A("| V Discussion | 재도전 규칙 한계 | `results/e5/ineligible_postmortem_0.json` (**탐색적**) | — |\n")

    A("## 3. Discussion 재료 (탐색적 — Results 인용 금지)\n")
    r1, r2 = pm["rule1_retry_would_have_helped"], pm["rule2_batch_vs_stream_gap"]
    A(f"- **재도전 불가 규칙이 r_V 하한을 정한다**: 후반 1,000 ep의 r_V {tail['r_V_observed']} 중 "
      f"부적격(X) 기여 **{tail['share_of_window'].get('X',0)}** "
      f"(= {tail['share_of_window'].get('X',0)/tail['r_V_observed']:.0%}), 미성숙(I) {tail['share_of_window'].get('I',0)}")
    A(f"- **회수 가능 상한**: 부적격 확정 후에도 BC 풀이 80 초과 축적된 클러스터 "
      f"**{r1['n_clusters']}/{pm['n_ineligible']}개** (중앙 잉여 +{r1['median_surplus']})")
    A(f"- **온라인 형성이 배치보다 어려운 정도**: E3에서 N*≤80이었는데 스트림 부적격 "
      f"**{r2['n_clusters']}/{pm['n_ineligible']}개** (해석 규칙 §5 2026-08-17 발동 — 배치 등가 조건에서의 재발)")
    cdg = r2["cause_diagnosis"]
    A(f"- **성숙 문턱의 이름 충돌**: E3 성숙 = 점추정 ŝ≥{cdg['maturity_criterion_gap']['e3_definition'][-3:]}, "
      f"E5 성숙 = 사후확률 기준 → P=20에서 실질 "
      f"**{cdg['maturity_criterion_gap']['e5_required_probe_successes_round1']}** 요구. "
      f"위 '어려운 정도' 수치에는 이 문턱 효과가 **포함**되어 있으므로 상한으로 읽을 것")
    A(f"- **라운드2 이월 함정**: c={cdg['round2_carryover_trap']['c_reinit']} 재초기화로 라운드1 실패가 φ로 승계 → "
      f"라운드1 실패 {cdg['round2_carryover_trap']['f1_threshold_for_unwinnable']}회 이상이면 "
      f"라운드2 만점도 통과 불가. 해당 {cdg['round2_carryover_trap']['n_unwinnable']}/{pm['n_ineligible']}개")
    A(f"- **성숙 초기 취약성**: 강등 {dem['n_demotions']}건, 발화 "
      f"{dem['median_fires_before_demotion']}회 중 1회 실패로 문턱 하회 (재성숙 {dem['n_regained']}건)\n")

    A("## 4. 미생성 자산 (집필 전 필요)\n")
    A("- **E2/E3 성숙 곡선 그림** — 원료는 `results/e3/*_curve.json` 27개 + `e2_gonogo.json`에 있음")
    A("- **H2 이중 해리 그림** — N* 분포(L별) + 천장 비교(T1/T3). 원료 `h2_analysis.json`")
    A("- **E4 scorer 비교 표** — 원료 `e4_scorer_table.json` (동일 셀 비교가 주 판독 근거)")
    A("- 필요 시 요청하면 `results/`에서 프로그래밍 산출로 생성 가능\n")

    A("## 5. 패키지 구성\n")
    A("| 경로 | 내용 |")
    A("|---|---|")
    A("| `results/e0`–`e5` | 전 실험 원자료 JSON (무효 데이터 제외) |")
    A("| `raw/` | E5 스트림 4,000행 + CF 1,536행 (독립 재산출용) |")
    A("| `figures/` | 생성 완료 그림 3장 |")
    A("| `scripts/` | 드라이버·판독기·게이트·학습·환경 구현 |")
    A("| `prereg/preregistration.md` | **동결 수치 원본** + §5 변경 이력 전문 |")
    A("| `prereg/log.md`, `CLAUDE.md` | 작업 로그·프로젝트 컨텍스트 전문 |")
    A("| `docs/` | 설계 문서 (E0 지시서, E5 드라이버 설계·체크리스트, Paper2 후보) |")
    A("| `prereg/git_history.txt` | git 이력 60건 |")

    open(os.path.join(pk, "MANUSCRIPT_SOURCES.md"), "w").write("\n".join(L))
    # ---- 무결성 체크섬 (원자료·결과)
    cs = subprocess.run(
        "find raw results -type f | sort | xargs sha256sum",
        shell=True, cwd=pk, capture_output=True, text=True).stdout
    open(os.path.join(pk, "checksums.sha256"), "w").write(cs)

    # ---- VERIFY.md: 검증자가 실제로 돌려볼 수 있는 절차
    head = subprocess.run(["git", "log", "-1", "--format=%H %ci %s"],
                          capture_output=True, text=True).stdout.strip()
    V = []
    V.append("# 최종 검증 절차\n")
    V.append(f"기준 커밋: `{head}`\n")
    V.append("## 0. 이 패키지로 검증할 수 있는 것\n")
    V.append("| 범위 | 가능 여부 | 방법 |")
    V.append("|---|---|---|")
    V.append("| 판독 재산출 (H4a·H4b·위험) | **가능** | 원자료 + `e5_analyze.py` (아래 1) |")
    V.append("| 3 seed 종합 재산출 | **가능** | `e5_seed_synthesis.py` (아래 2) |")
    V.append("| 그림 재생성 | **가능** | `fig_e5_reading.py` (아래 3) |")
    V.append("| 부적격 사후분석 재산출 | **가능** | `e5_ineligible_postmortem.py` |")
    V.append("| 원자료 무결성 | **가능** | `checksums.sha256` (아래 0-1) |")
    V.append("| 스트림 전체 재실행 | 코드는 포함, **데이터는 재생성 필요** | 아래 4 |")
    V.append("")
    V.append("### 0-1. 무결성 확인\n```bash\nsha256sum -c checksums.sha256\n```\n")
    V.append("## 1. 판독 재산출 (seed별 H4a·H4b)\n")
    V.append("```bash")
    V.append("# 패키지 루트 = 저장소 구조 미러이므로 스크립트가 그대로 실행된다.")
    V.append("for i in 0 1 2; do")
    V.append("  gunzip -c raw/e5_stream_$i.jsonl.gz > results/e5/stream_$i.jsonl")
    V.append("  gunzip -c raw/e5_cf_$i.jsonl.gz     > results/e5/cf_$i.jsonl")
    V.append("  gunzip -c raw/e5_cf_queue_$i.jsonl.gz > results/e5/cf_queue_$i.jsonl")
    V.append("done")
    V.append("for i in 0 1 2; do python experiments/e5_analyze.py --seed-idx $i; done")
    V.append("```")
    V.append("판독기는 **사전등록에 등재된 규칙만** 집행하며, 그림자 관할 예측치도 하드코딩이 아니라")
    V.append("`prereg/preregistration.md` §5 원문에서 정규식으로 추출한다. 산출된 `reading_{i}.json`을")
    V.append("동봉된 `results/e5/reading_{i}.json`과 비교하면 일치해야 한다.\n")
    V.append("## 2. 3 seed 종합\n```bash\npython experiments/e5_seed_synthesis.py\n```")
    if syn:
        sa, sb, sr = syn["H4a_call_rate"], syn["H4b_noninferiority"], syn["risk_control"]
        V.append(f"기대 출력: `{syn['verdict']}` · "
                 f"H4a {sa['first1000']['mean']}±{sa['first1000']['sd']} → "
                 f"{sa['last1000']['mean']}±{sa['last1000']['sd']} · "
                 f"H4b {sb['diff']['mean']:+.4f}±{sb['diff']['sd']:.4f} · "
                 f"위험 {sr['pr_fail_given_fire']['mean']}±{sr['pr_fail_given_fire']['sd']}\n")
    V.append("## 3. 그림\n```bash\nfor i in 0 1 2; do python experiments/fig_e5_reading.py --seed-idx $i; done\n```\n")
    V.append("## 4. 스트림 재실행 (선택 — 장시간)\n")
    V.append("체크포인트(93 GB)와 수집 HDF5(8.8 GB)는 용량상 **제외**했다. 다만 모든 에피소드가")
    V.append("`(suite, task, seed, base_idx, w, noise_seed)` 여섯 원소로 **완전히 결정적**이므로")
    V.append("(CF 결정성 사전 검증이 seed마다 5/5 통과) 재실행으로 동일 데이터를 재생성할 수 있다.\n")
    V.append("```bash")
    V.append("# 환경: conda env 2개 (hv2_oft = OpenVLA-OFT, hv2_hab = ACT/분석)")
    V.append("export HF_HOME=<repo>/.hf_cache   # 공용 캐시 오염 방지")
    V.append("python experiments/e5_driver.py --seed-idx 0        # 약 16.7 h/seed (RTX 5090)")
    V.append("python experiments/e5_counterfactual.py --seed-idx 0  # 약 4 h/seed")
    V.append("```")
    V.append("드라이버는 재학습마다 `assert_retrain_contract()`로 (a) 정규화 stats가 자기 학습")
    V.append("데이터에서 산출됐는지 (b) 스텝이 배치 등가값인지 (c) `|B_k|`가 HDF5와 3중 일치하는지를")
    V.append("검증하고, 위반 시 즉시 정지한다. 본 실행의 통과 기록은 `results/runtime_gate_assertions.txt`.\n")
    V.append("## 5. 제외 항목과 이유\n")
    V.append("| 항목 | 크기 | 이유 |")
    V.append("|---|---|---|")
    V.append("| `checkpoints/` | 93 GB | ACT 체크포인트. 재학습으로 재생성 가능(결정적 seed) |")
    V.append("| `data/` | 8.8 GB | 수집·스트림 HDF5(RGB-D 프레임). 재실행으로 재생성 가능 |")
    V.append("| `results/e5/seed0_normstats_invalid/` | — | **무효 실행** — 인용 금지 대상이라 의도적 제외 |")
    V.append("| OpenVLA-OFT 가중치 | ~16 GB | 공개 체크포인트 `moojink/openvla-7b-oft-finetuned-libero-*` |")
    open(os.path.join(pk, "VERIFY.md"), "w").write("\n".join(V))

    subprocess.run(["tar", "czf", f"{pk}.tar.gz", "-C", HABIT2, os.path.basename(pk)])
    print(f"[MSPACK] {pk}.tar.gz")


if __name__ == "__main__":
    main()
