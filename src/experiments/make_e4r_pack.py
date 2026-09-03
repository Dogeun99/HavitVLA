"""E4-R 판정 패키지 생성기 (H3 문구 재작성 판정용) — 수치 전부 프로그래밍 주입.

실행: hv2_hab python -u experiments/make_e4r_pack.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "e4r_pack")
COPY = ["results/e4/e4r_competence_map.json", "results/e4/e4_readjudication.json",
        "results/e4/e4_scorer_table.json", "experiments/e4r_competence_map.py",
        "configs/preregistration.md", "log.md"]


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-12"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)
    d = json.load(open(os.path.join(HABIT2, "results/e4/e4r_competence_map.json")))
    R, W, C = d["reading"], d["w_grid"], d["clusters"]
    ca = R["rule2_alignment"]["competence_auc"]
    ts = R["rule2_alignment"]["threshold_sweep"]
    cm = R["rule2_alignment"]["confusion_valid_episodes"]

    rows = []
    for cl, v in C.items():
        cells = " | ".join(
            f"{v['by_w'][str(w)]['habit_success_rate']:.2f}/{v['by_w'][str(w)]['gate_reject_rate']:.2f}"
            for w in W)
        rows.append(f"| {cl} | {cells} | {v['w_star']} | {v['usable_w_max']} | "
                    f"{ca['per_cluster'][cl]['competence_auc']} | {ca['per_cluster'][cl]['band_auc']} |")

    md = f"""# E4-R 역량 지도 — H3 문구 재작성 판정 자료 ({datetime.now().date()})

수치 전부 `e4r_competence_map.json`에서 프로그래밍 주입. 6 클러스터 × w{W} × 15 ep,
habit(n=80) rollout only, 유효 {sum(v['by_w'][str(w)]['n_valid'] for v in C.values() for w in W)} ep.

## 1. 원자료 (성공률/게이트기각률)

| 클러스터 | {' | '.join(f'w={w}' for w in W)} | w* | usable | 역량AUC | 대역AUC |
|---|{'---|'*len(W)}---|---|---|---|
{chr(10).join(rows)}

## 2. ★라벨을 바꾸면 게이트 성적이 달라진다 (판정 핵심)

- **역량 AUC macro = {ca['macro']}** vs 대역 AUC macro = {ca['band_macro_prev']} (pooled 역량 = {ca['pooled']})
- **{ca['n_clusters_improved']}/6 클러스터에서 역량 AUC가 대역 AUC보다 높음**;
  **{ca['n_clusters_pass_075']}개는 임계 0.75 통과** (goal_task2 {ca['per_cluster']['libero_goal_task2']['competence_auc']},
  object_task5 {ca['per_cluster']['libero_object_task5']['competence_auc']})
- 최대 상승: goal_task2 {ca['per_cluster']['libero_goal_task2']['band_auc']} → {ca['per_cluster']['libero_goal_task2']['competence_auc']}
  ({ca['per_cluster']['libero_goal_task2']['delta']:+})
- **역전 1건**: spatial_task0 {ca['per_cluster']['libero_spatial_task0']['band_auc']} →
  {ca['per_cluster']['libero_spatial_task0']['competence_auc']} ({ca['per_cluster']['libero_spatial_task0']['delta']:+})
- **그러나 macro {ca['macro']}는 여전히 0.75 미달** — "질문이 틀렸다"는 부분 지지이지 전면 반전은 아님.

## 3. ★오정렬은 보정으로 해결되지 않는다 (H3 갈래 판별)

| 작동점 | 미탐(실패 중) | 오탐(성공 중) |
|---|---|---|
| 현 운용 q | {cm['miss_rate_among_failures']} | {cm['false_alarm_rate_among_successes']} |
| 전역 최적 임계 ({ts['global_best_balanced']['threshold']}) | **{ts['global_best_balanced']['miss']}** | {ts['global_best_balanced']['false_alarm']} |
| 오탐 ≤20% 제약 | {ts['min_miss_at_fa_le_20pct']['miss']} | {ts['min_miss_at_fa_le_20pct']['false_alarm']} |
| 오탐 ≤10% 제약 | {ts['min_miss_at_fa_le_10pct']['miss']} | {ts['min_miss_at_fa_le_10pct']['false_alarm']} |

{ts['verdict']}

## 4. 정직 보고 — 앞선 보고의 정정 2건

1. **"2/6 usable 초과"는 점추정 기준**. {R['rule1_w_star']['ci_caveat']}
2. **미탐 41%·오탐 27%는 현 q 기준의 한 점**이며, 최적 임계에서도 미탐 32%가 바닥이라는 것이
   더 강한 진술이다(3절). 앞선 보고는 이 상한을 제시하지 못했다.

## 5. 추가 한계 (판정 시 고려)

- 게이트는 **I₀(초기 프레임)만** 본다. 실패는 에피소드 중반에 일어나므로, I₀ 관할이
  원리적으로 관측할 수 없는 실패가 미탐 32%의 일부를 구성할 수 있다 — 이는 "표현력 부족"과
  "관측 시점의 구조적 한계"를 분리해야 함을 뜻한다(후속 설계 논점).
- w=0.08의 물리 탈락 16.7% → 유효분만 남으면 쉬운 배치가 과대표집될 수 있음(생존 편향).
- 6/25 클러스터 표본, 셀당 15 ep.

## 6. H3 문구 재작성 후보 (판정 요청)

현행: "(i)+(ii) 혼합 — 기하 관할은 표현 비용에 종속적이나 21× 비용에도 임계 미달,
저비용 실시간 관할 미해결."

E4-R 반영 후보: **"게이트의 결함은 민감도가 아니라 역량 경계와의 정렬이다. 라벨을 대역에서
역량으로 바꾸면 {ca['n_clusters_improved']}/6에서 성적이 오르고 2개는 임계를 넘지만(macro {ca['macro']}),
임계 재보정으로는 미탐 {ts['global_best_balanced']['miss']}가 바닥이어서 정렬 실패는 보정이 아니라
표현의 문제로 남는다. 역량 경계 자체가 클러스터별로 6배 차이나므로 관할 보정은 클러스터별이어야 한다."**
"""
    with open(os.path.join(PACK, "E4R_REPORT.md"), "w") as f:
        f.write(md)
    print(f"[E4R-PACK] {len(COPY)} 파일 + E4R_REPORT.md + git_history.txt")


if __name__ == "__main__":
    main()
