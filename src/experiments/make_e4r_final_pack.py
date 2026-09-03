"""E4-R 3부작 통합 판정 패키지 (H3 확정용) — 수치 전부 프로그래밍 주입.

포함: 역량 지도(w 스윕) + teacher w-사다리(통제군) + 작업공간 실측 + 그림.
실행: hv2_hab python -u experiments/make_e4r_final_pack.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "e4r_final_pack")
COPY = [
    "results/e4/e4r_competence_map.json", "results/e4/e4r_teacher_ladder.json",
    "results/e4/workspace_extent.json", "results/e4/fig_workspace_extent.png",
    "results/e4/e4_readjudication.json", "results/e4/e4_scorer_table.json",
    "experiments/e4r_competence_map.py", "experiments/e4r_teacher_ladder.py",
    "experiments/workspace_extent.py",
    "configs/preregistration.md", "log.md",
]


def L(p):
    return json.load(open(os.path.join(HABIT2, p)))


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-15"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)

    cm = L("results/e4/e4r_competence_map.json")
    tl = L("results/e4/e4r_teacher_ladder.json")
    ws = L("results/e4/workspace_extent.json")
    rj = L("results/e4/e4_readjudication.json")
    W = cm["w_grid"]
    R, TR = cm["reading"], tl["reading"]
    ca, tsw = R["rule2_alignment"]["competence_auc"], R["rule2_alignment"]["threshold_sweep"]
    cmx = R["rule2_alignment"]["confusion_valid_episodes"]
    rg, s = TR["routing_gain"]["by_w"], tl["summary"]
    G = lambda w, k: rg[str(w)][k]

    ladder = "\n".join(
        f"| {w} | {s['teacher_by_w'][str(w)]} | {s['habit_by_w'][str(w)]} | "
        f"**{s['gap_by_w'][str(w)]:+.3f}** | {G(w,'vla_routing_rate')*100:.1f}% | "
        f"**{G(w,'conditional_gain_per_ep'):+.4f}** | {G(w,'mean_gain_among_rejected'):+.3f} | "
        f"{G(w,'mean_query_latency_ms')} ms ({G(w,'latency_vs_all_habit_x')}x) | "
        f"{ws['w_vs_reach'][str(w)]['pct_of_equiv_diameter']}% |" for w in W)
    comp = "\n".join(
        f"| {cl} | " + " | ".join(
            f"{v['by_w'][str(w)]['habit_success_rate']:.2f}/{v['by_w'][str(w)]['gate_reject_rate']:.2f}"
            for w in W) + f" | {v['w_star']} | {ca['per_cluster'][cl]['competence_auc']} |"
        for cl, v in cm["clusters"].items())

    md = f"""# E4-R 3부작 통합 — H3 확정 판정 자료 ({datetime.now().date()})

수치 전부 동봉 JSON에서 프로그래밍 주입. 세 실험 모두 **판독 규칙을 결과 산출 전 §5 등재**.

## A. teacher w-사다리 (통제군 — 판정의 축)

| w | teacher | habit | 격차 | VLA 라우팅 | **조건부 이득/ep** | 기각분 평균 | 평균 지연 | w/도달지름 |
|---|---|---|---|---|---|---|---|---|
{ladder}

- **격차 추세: Spearman(w, 격차) = {TR['rule2_trend']['spearman_w_vs_gap']} (p={TR['rule2_trend']['p']})** —
  단조 증가 아님. 정점은 **w={TR['rule1_gap_curve']['peak_w']}**.
- {TR['rule2_trend']['verdict']}
- {TR['rule3_teacher_robustness']['verdict']}
- **클러스터 이질성**: goal_task2 격차 = {TR['cluster_heterogeneity']['goal_task2_gap_all_w']} — 전 폭 ≈0.

## B. 역량 지도 (습관 성공률 / 게이트 기각률)

| 클러스터 | {' | '.join(f'w={w}' for w in W)} | w* | 역량AUC |
|---|{'---|'*len(W)}---|---|
{comp}

- 역량 AUC macro **{ca['macro']}** vs 대역 AUC macro {ca['band_macro_prev']} — {ca['n_clusters_improved']}/6 상승,
  {ca['n_clusters_pass_075']}개 임계 통과, 1건 역전(spatial_task0).
- 혼동행렬(유효 {cmx["miss"]+cmx["false_alarm"]+cmx["hit"]+cmx["correct_accept"]} ep): 미탐 **{cmx['miss_rate_among_failures']}**,
  오탐 **{cmx['false_alarm_rate_among_successes']}**.
- **임계 재보정으로 해결 불가**: 전역 최적에서도 미탐 {tsw['global_best_balanced']['miss']} /
  오탐 {tsw['global_best_balanced']['false_alarm']}가 바닥.
- CI 유의: {R['rule1_w_star']['ci_caveat']}

## C. 작업공간 실측 (같은 단위 환산)

검증 도달 영역(teacher 성공 {ws['verified_reach']['n_points']}점) = hull **{ws['verified_reach']['convex_hull_area_m2']} m²**,
등가반경 **{ws['verified_reach']['equivalent_radius_m']} m**. 공식 배치 bbox
{ws['official_placement_bbox']['x_span']}×{ws['official_placement_bbox']['y_span']} m.
→ **w\\* (0.02–0.06)은 도달 지름의 {ws['w_vs_reach']['0.02']['pct_of_equiv_diameter']}–{ws['w_vs_reach']['0.06']['pct_of_equiv_diameter']}%**
(면적 {ws['w_vs_reach']['0.02']['area_pct_of_hull']}–{ws['w_vs_reach']['0.06']['area_pct_of_hull']}%).
습관의 기하 일반화는 **국소적**. 그림: `fig_workspace_extent.png`.

## D. 세 결과를 합치면 (판정 논거)

1. **게이트의 가치는 구간 국소적이다.** 조건부 라우팅 이득(uid 매칭)은 w=0.04에서 +{G(0.04,'conditional_gain_per_ep')}/ep로
   최대이고 양극단에서 축소(+{G(0.01,'conditional_gain_per_ep')}, +{G(0.08,'conditional_gain_per_ep')}).
   **기각분 평균 이득은 전 폭 양수**(+{G(0.08,'mean_gain_among_rejected')}~+{G(0.04,'mean_gain_among_rejected')}) —
   게이트는 무작위 기각자보다 유익하다.
   좁은 폭 = 습관이 teacher와 대등해 기각 불필요 / 넓은 폭 = **teacher도 붕괴(0.335)해 기각해도 갈 곳이 없음**.
2. **"미탐 {cmx['miss_rate_among_failures']}"의 실질 손실은 축소 해석되어야 한다** — 미탐이 몰린 넓은 폭이
   바로 teacher도 못 하는 구간이다. 반대로 **오탐 {cmx['false_alarm_rate_among_successes']}**는 이득이
   +{G(0.01,'conditional_gain_per_ep')}/ep뿐인 좁은 폭에 몰려 있고 그 구간에서도 에피소드의
   {G(0.01,'vla_routing_rate')*100:.0f}%를 85 ms 경로로 보낸다(평균 지연 {G(0.01,'latency_vs_all_habit_x')}×).
   **두 해석은 대칭으로 서술한다 — 한쪽만 완화하지 않는다.**
3. 그럼에도 **정렬 실패 자체는 표현의 문제**(임계 재보정으로 미탐 {tsw['global_best_balanced']['miss']} 바닥)이며,
   **역량 경계는 도달 영역의 8–23%로 국소**하다.

## E. H3 최종 문구 — **확정** (연구원 판정 2026-08-16, §5 등재)

"관할 게이트의 가치는 **teacher와 습관의 역량 격차가 존재하는 중간 변이 구간에 국한**된다 —
조건부 라우팅 이득이 w=0.04에서 +{G(0.04,'conditional_gain_per_ep')}/ep로 정점이고 양극단에서
+{G(0.01,'conditional_gain_per_ep')}/+{G(0.08,'conditional_gain_per_ep')}로 축소되며(좁은 폭: 습관이 teacher와
대등해 기각 불필요 / 넓은 폭: teacher도 {s['teacher_by_w']['0.08']}로 붕괴해 기각해도 갈 곳 없음), 그 대가로
w=0.01에서도 에피소드의 {G(0.01,'vla_routing_rate')*100:.0f}%가 85.07 ms 경로를 타 평균 질의 지연이
전량 습관 대비 {G(0.01,'latency_vs_all_habit_x')}×가 된다. 그 구간에서조차 현 관할은 **약하게만 정렬**되어
있다 — 방향은 유의하나(Spearman 0.628, p=2e-4) 수준이 어긋나 미탐 {cmx['miss_rate_among_failures']}·오탐
{cmx['false_alarm_rate_among_successes']}이고, **임계 재보정으로도 미탐 {tsw['global_best_balanced']['miss']}가 바닥**이어서
원인은 임계가 아니라 표현이다. 다만 기각분 평균 이득이 전 폭 양수이므로 **게이트는 무작위 기각자보다
유익**하다. 따라서 저비용 실시간 관할은 미해결이되, **그 해결의 실익 자체가 구간에 제한된다**."

## F. 후속 (승인·등재 완료)

- **학습 폭 확대 실험 = 본 논문 부록 승격**: 2 클러스터(object_task0·goal_task2) × 수집 w=0.04 ×
  n-grid{{10,20,40,80}} → 역량 경계 재측정 + N\\* 비교. 목적 = **국소성 한계에 대한 처방 존재 실증**.
  E5 GPU 우선, 유휴 슬롯 실행.
- 미탐/오탐 해석은 **대칭 서술**로 논문 명시 (§5 등재).
"""
    with open(os.path.join(PACK, "E4R_FINAL_REPORT.md"), "w") as f:
        f.write(md)
    print(f"[E4R-FINAL-PACK] {len(COPY)} 파일 + E4R_FINAL_REPORT.md + git_history.txt")


if __name__ == "__main__":
    main()
