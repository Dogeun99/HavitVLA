"""E5 설계 리뷰 패키지 생성기 (연구원 지시 2026-08-16 §5) — 코드 미작성 단계.

실행: hv2_hab python -u experiments/make_e5_design_pack.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "e5_design_pack")

COPY_FILES = [
    "docs/E5_DRIVER_DESIGN.md", "docs/E5_DRIVER_CHECKLIST.md",
    "docs/PAPER2_E5_POSTHOC_CANDIDATES.md",
    "results/e4/e4r_teacher_ladder.json", "results/e4/e4r_competence_map.json",
    "results/e4/workspace_extent.json",
    "configs/preregistration.md", "log.md",
    # 설계 근거 산출물
    "results/e1/e1_latency.json", "results/e0/e0_7_walltime.json",
    "results/e3/e3_curves.json", "results/e4/e4_readjudication.json",
    "results/e4/e4_scorer_table.json",
    # 재사용 대상 코드 (설계가 전제하는 기존 구현)
    "gates/two_stage.py", "envs/stream.py", "envs/chained_env.py",
    "experiments/gate_regression.py", "experiments/executor_chunkbreak_test.py",
]


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY_FILES:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-40"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)

    lat = json.load(open(os.path.join(HABIT2, "results/e1/e1_latency.json")))
    tl = json.load(open(os.path.join(HABIT2, "results/e4/e4r_teacher_ladder.json")))
    rg01 = tl["reading"]["routing_gain"]["by_w"]["0.01"]
    wt = json.load(open(os.path.join(HABIT2, "results/e0/e0_7_walltime.json")))
    # 결함2 반영: 25 클러스터 스위트 가중 평균 (long 520 스텝 포함)
    comp = {"libero_object": 10, "libero_goal": 10, "libero_spatial": 2, "libero_10": 3}
    ep_s = sum(wt["per_suite"][k]["mean_s"] * n for k, n in comp.items()) / sum(comp.values())
    train_s = lat["anchor5_act_train_n40"]["train_seconds"]
    n_cl = 25  # 결함2 반영 (libero_10 3 편입)
    exposure = 0.9 * 4000 / n_cl
    budget = {"stream_h_per_seed": round(4000 * ep_s / 3600, 1),
              "retrain_h_max": round(n_cl * 2 * train_s / 3600, 1),
              "probe_h_max": round(n_cl * 2 * 20 * ep_s / 3600, 1),
              "counterfactual_h_worst": round(0.9 * 4000 * ep_s / 3600, 1)}
    budget["total_worst_3seed_h"] = round(3 * sum(budget.values()), 1)

    md = f"""# E5 설계 리뷰 패키지 ({datetime.now().date()}) — 코드 미작성

## 제출물
- **E5_DRIVER_DESIGN.md** — 설계서 **v0.2** (필수 5항목: 상태 기계 / 로깅 스키마 /
  counterfactual 큐 / 재학습·probe 일시정지 재개 / heartbeat 통합) + **결정 6건 반영 완료**
  + **50 ep 스모크 관문(§9) 신설** + 결함 2건 반영(시간 3장부 §0b, 25 클러스터)
- **E5_DRIVER_CHECKLIST.md** — 사전 함정 6렌즈 (설계서 §7에 절별 매핑)
- 사전등록 전문 + log.md + 설계 근거 산출물(앵커·E3 곡선·E4 판정) + 재사용 코드 5종

## 설계 근거 수치 (앵커 프로그래밍 산출)
- 에피소드 평균 {ep_s:.1f}s / 재학습 1회 {train_s}s (= VLA 호출 {lat['anchor5_act_train_n40']['vla_call_equivalents']}회 등가)
- 클러스터 **{n_cl}개**(libero_10 편입), 에피소드 단가 = 스위트 **가중** 평균, 클러스터당 노출 **{exposure:.1f}**
- 예산(최악): 스트림 {budget['stream_h_per_seed']}h + 재학습 ≤{budget['retrain_h_max']}h +
  probe ≤{budget['probe_h_max']}h + counterfactual ≤{budget['counterfactual_h_worst']}h
  = **3 seed 합 ≈ {budget['total_worst_3seed_h']}h**

## ★E4-R이 E5 설계에 주는 함의 (신규 — 리뷰 반영 요청)

E5 스트림의 in-distribution 폭은 **w_id = 0.01**이고, E4-R은 정확히 그 폭에서 관할을 켰을 때의
효과를 실측했다:

| 지표 (w=0.01) | 값 | E5 함의 |
|---|---|---|
| 조건부 라우팅 이득 | **+{rg01['conditional_gain_per_ep']}/ep** | 관할을 켜도 성공률 이득이 미미 |
| VLA 라우팅 비율 | **{rg01['vla_routing_rate']*100:.0f}%** | 호출률이 31%p 증가 — **r_V 감소 주장과 정면 충돌** |
| 평균 질의 지연 | {rg01['mean_query_latency_ms']} ms ({rg01['latency_vs_all_habit_x']}×) | 상각 효과 대부분 상쇄 |

→ **관할을 발화 결정에서 뺀 REDUCE 판정이 E5 목적(H4: r_V 감소 + 비열등)과 정합**함을 사전 예측
수치로 뒷받침한다. **그림자 관할 로깅의 분석 계획**을 이 예측의 사후 검증으로 명시할 것을 제안:
스트림에서 "관할을 켰다면" 의 반사실 호출률·성공률을 로그만으로 산출(추가 rollout 불요).

## 리뷰 결정 요청

**기판정 (2026-08-16, 설계서 §8·§5 반영 완료)**: 1 스트림 대역 신설 / 2 counterfactual 종료 후
배치 / 3 재학습 트리거 = |B_k| / 4 강등 후 r_k 승계·R_max=2 / 5 novel 정상 lifecycle 편입 /
6 예산 유지 + seed 순차. **잔여 판정 대기 2건:**

7. **[신규] 그림자 관할 분석 계획**: 위 반사실 산출(관할 ON 가정의 r_V·성공률)을 E5 산출물에
   포함할지 — 포함 시 로깅 스키마는 이미 충분(shadow_jur 점수·판정 기록)하며 추가 비용 0.
8. **[신규] 스모크 트리거 (a)/(b)** 확인 — 설계서 §9 미결 (실행측 권고 = (a) 스모크 전용 축소 트리거)

**리뷰 통과 후에만 구현 착수** — 현재 코드 0줄.
"""
    with open(os.path.join(PACK, "SUBMISSION.md"), "w") as f:
        f.write(md)
    with open(os.path.join(PACK, "design_budget.json"), "w") as f:
        json.dump({"episode_s": round(ep_s, 2), "retrain_s": train_s,
                   "n_clusters": n_cl, "exposure_per_cluster": round(exposure, 1),
                   "budget_h": budget}, f, indent=2, ensure_ascii=False)
    print(f"[E5-DESIGN-PACK] {len(COPY_FILES)} 파일 + SUBMISSION.md + design_budget.json")


if __name__ == "__main__":
    main()
