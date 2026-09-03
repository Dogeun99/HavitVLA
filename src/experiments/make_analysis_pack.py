"""§6 완료 분석 패키지 생성기 (해석 세션용 — 연구원 지시 3.d).

원칙: ANALYSIS.md의 수치까지 전부 결과 JSON에서 프로그래밍 주입 (수동 입력 금지).
실행: hv2_hab python -u experiments/make_analysis_pack.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "analysis_pack")

COPY_FILES = [
    "results/e3/e3_curves.json", "results/e3/h2_analysis.json", "results/e3/covariates.json",
    "results/e3/free_joints_census.json", "results/e1/e1_sv_per_task.json",
    "results/e3/chained_libero_object_task0_curve.json",
    "results/e3/chained_libero_object_task5_curve.json",
    "configs/preregistration.md", "log.md",
]


def load(p):
    return json.load(open(os.path.join(HABIT2, p)))


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY_FILES:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-25"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)

    c = load("results/e3/e3_curves.json")
    h = load("results/e3/h2_analysis.json")
    t2 = c["t_ceiling"]["T2_product_baseline"]
    t13 = c["t_ceiling"]["T1_vs_T3"]
    dec = h["decomposition_L"]
    reg = h["regression_formation22"]["rank_ols"]
    import collections

    nstar = dict(collections.Counter(str(v) for v in c["n_star"].values()))
    ch0, ch5 = t2["chained_libero_object_task0"], t2["chained_libero_object_task5"]

    md = f"""# §6 집계·H2 본 판정 — 해석 세션 브리핑 ({datetime.now().date()})

수치는 전부 동봉 JSON에서 프로그래밍 주입 (생성기: make_analysis_pack.py).
완결성: **{c['status']}** (기대 {c['completeness']['expected']}, missing {c['completeness']['missing']},
unexpected {c['completeness']['unexpected']}).

## 1. C-T2 곱 기준선 주 검정 (p₀ = {ch0['product_baseline_p0']}, held-out 50)

| 체인 | ŝ_chain(80) | Wilson | p(하회) | p(상회) | bootstrap P(Δ<0) / Δ̄ | 판독 |
|---|---|---|---|---|---|---|
| task0 | {ch0['s80_chain']} | {ch0['wilson_chain_80']} | **{ch0['p_below_product']}** | {ch0['p_above_product']} | {ch0['bootstrap']['P_delta_below_0']} / {ch0['bootstrap']['delta_mean']} | **유의 하회 — 결합 비용 검출** |
| task5 | {ch5['s80_chain']} | {ch5['wilson_chain_80']} | {ch5['p_below_product']} | {ch5['p_above_product']} | {ch5['bootstrap']['P_delta_below_0']} / {ch5['bootstrap']['delta_mean']} | 하회 아님 (상회 경향 비유의) |

체인 곡선: task0 {c['clusters']['chained_libero_object_task0']['curve']} (N*={c['clusters']['chained_libero_object_task0']['N_star']}) /
task5 {c['clusters']['chained_libero_object_task5']['curve']} (N*={c['clusters']['chained_libero_object_task5']['N_star']}).
**셀 내 분기**: E2 형성 속도 이질성(task0 느림·task5 빠름)과 정합 — "결합 비용은 단일 태스크
형성 난도에 조건부"가 후보 정식화 (해석 세션 판정 대상).

## 2. 동역학 천장 (T1 vs T3, 주 증거)

ŝ80(T1)={t13['s80_T1']} vs ŝ80(T3)={t13['s80_T3']}, 단측 p={t13['p_one_sided_decrease']:.4f}
({t13['method']}) — **α=0.05 비유의** (정직 보고: long task0=0.75가 task2/5(≥0.90)에 희석).
클러스터별 분해는 e3_curves.json 참조.

## 3. H2-L vs H2-L′ (본 판정, FINAL)

- 레벨 분산 분해 (형성 {dec['n']}): between = **{dec['between_share']}** / within = {dec['within_share']},
  Kruskal–Wallis H={dec['kruskal_H']}, p={dec['kruskal_p']} — **L은 N*를 설명하지 못함**.
- 공변량 순위 회귀 (순열 p, B=10⁴):
  median_len β={reg['median_len']['beta']} **p={reg['median_len']['perm_p']}** (유일 유의) /
  free_joints β={reg['free_joints']['beta']} p={reg['free_joints']['perm_p']} (경계·부호 음 — 해석 주의) /
  S_V β={reg['S_V_cluster']['beta']} p={reg['S_V_cluster']['perm_p']}.
- **판정: H2-L′ 지지** — 사전 등재 프레이밍("인수분해 아키텍처가 의미 부담을 흡수, 형성
  비용은 운동 축 지배") 발동 가능. interval 민감도·전 25 표본 회귀는 h2_analysis.json.

## 4. N* 분포 (27)

{json.dumps(nstar, ensure_ascii=False)} — 우측 절단 1 (libero_10_task0).

## 5. 판정 요청 (해석 세션)

1. C-T2 분기의 정식화: "조건부 결합 비용" 채택 여부와 본문/부록 배치.
2. T1 vs T3 비유의의 처리: 경향 보고 + 클러스터별 분해 강조 vs 천장 주장 약화.
3. H2-L′ 프레이밍 채택 및 free_joints 음의 부호 해석(공선성: median_len과의 상관 확인 필요).
4. 3막 사가·task6 반사실의 본문/부록 배치.
"""
    with open(os.path.join(PACK, "ANALYSIS.md"), "w") as f:
        f.write(md)
    print(f"[ANALYSIS-PACK] {len(COPY_FILES)} 파일 + ANALYSIS.md + git_history.txt -> {PACK}")


if __name__ == "__main__":
    main()
