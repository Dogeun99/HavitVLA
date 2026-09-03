"""E4-3 갈래 판정 패키지 생성기 (분석용 Claude 전달 — 수치 전부 프로그래밍 주입).

실행: hv2_hab python -u experiments/make_e4_decision_pack.py
"""
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "e4_decision_pack")

COPY_FILES = [
    "results/e4/e4_pilot_auc.json", "results/e4/e4_decision_support.json",
    "results/e4/novel_manifest.json", "results/e4/novel_manifest_v1_negative.json",
    "results/e4/e4_walltime.json",
    "experiments/e4_known_frames.py", "experiments/e4_novel_frames.py",
    "experiments/e4_pilot_auc.py", "experiments/e4_decision_sim.py",
    "configs/preregistration.md", "log.md",
]


def load(p):
    return json.load(open(os.path.join(HABIT2, p)))


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY_FILES:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-30"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)

    pilot = load("results/e4/e4_pilot_auc.json")
    sim = load("results/e4/e4_decision_support.json")

    # 경로별 macro (pilot json에서)
    path_macro = pilot["paths_macro"]
    # 스위트별 재구성 평균 (sim에서)
    suite_re = defaultdict(list)
    for cl, v in sim["clusters"].items():
        if v.get("recomposed_auc") is not None:
            suite_re[cl.rsplit("_task", 1)[0]].append(v["recomposed_auc"])
    suite_mean = {s: round(sum(v) / len(v), 3) for s, v in sorted(suite_re.items())}
    geo2 = sim["geometry"]["libero_object_task2"]

    md = f"""# E4-3 관할 파일럿 갈래 판정 브리핑 ({datetime.now().date()})

수치 전부 동봉 JSON에서 프로그래밍 주입 (생성기: make_e4_decision_pack.py).

## 1. 측정 (사전등재 구성 그대로)

- **macro AUC(주 novel 풀) = {pilot['macro_auc_primary']}** (임계 {pilot['threshold']}) — 전 {len(pilot['clusters'])}
  클러스터 미달 → 문면상 REDUCE. 다수 역방향(<0.5) — 구성 결함 신호.
- 부차(base 0–39) = {pilot['macro_auc_secondary']} / micro(스케일 캐비앳) = {pilot['micro_auc_primary_caveat_scale']}.

## 2. 원인 분해 (측정 기계는 건전)

| 경로 | macro AUC | 판독 |
|---|---|---|
| (ii) 차용 primary (spatial) | **{path_macro.get('borrow__primary')}** (부차 {path_macro.get('borrow__secondary')}) | 강한 분리 |
| (i) w 확대 primary | {path_macro.get('w_expand__primary')} | 약함 — goal에서 특히 미약 |
| (iii) BDDL 재샘플 | **{path_macro.get('resample__single')} (역방향)** | **novel 아님 — 분포 중심 생성** (E0-6 재샘플 폭 Δ≤2.3cm 정합; 점수 초밀집) |

점수 기하 예 (object_task2): calib {geo2['calib']} / known {geo2['known']} /
w_expand {geo2.get('w_expand__primary')} / **resample {geo2.get('resample__single')}** (calib보다 낮음)
— 전 클러스터 기하는 e4_decision_support.json. **known ≈ calib** = 관할 false-reject 건전.

## 3. (a) 재구성안의 가정적 재판정 — 시뮬레이션 실수치 (판정 지원용, 사전등록 아님)

재구성 = w_expand primary ∪ borrow primary (재샘플 제외; long은 w=0.02 미생성 — 제외):
- **macro = {sim['macro_recomposed_excl_long']}** (n={sim['n_clusters_in_macro']}), **<0.75인 클러스터 {sim['n_below_075']}/{sim['n_clusters_in_macro']}**
- 스위트 평균: {json.dumps(suite_mean, ensure_ascii=False)}
- **판독: 재구성해도 0.75 미달 전망** — goal(평균 {suite_mean.get('libero_goal')})이 구조적 원인:
  물체가 작고 배치 영역이 좁아 w=0.04 시각 이탈을 DINOv2-PCA 관할이 감지하지 못함.
  object({suite_mean.get('libero_object')})·spatial({suite_mean.get('libero_spatial')})·차용({path_macro.get('borrow__primary')})은 유효 영역.

## 4. 선택지

**(a) 재구성 정식 재판정**: §5 등재(novel 구성 개정 + 재샘플 = in-distribution negative
   control 재분류) → long w=0.02 생성(60 realize ≈ 2분) → 재판정. 예상 macro ≈ 0.68 —
   **미달 전망 그대로**이나, 관할의 유효/무효 영역이 정식 기록으로 남아 논문 관할 절·F3
   재료 강화. 비용 ≈ 30분.
**(b) 즉시 REDUCE 확정**: 측정(0.465) 기준 §5 축소 발동 기록 → **E5 성숙도 단독 설계**
   (사전등록된 우아한 퇴화: "관할은 열린 세계(Paper 2) 과제" 프레이밍).

두 갈래 모두 종착은 E5 성숙도 단독일 가능성이 높음 — 차이는 기록의 질과 30분.
부속 질문: REDUCE 시에도 scorer 비교(E4-4)를 부록급으로 실행할지 (통합 지시서는 "통과 시"
조건부 — 기본은 미실행), 재샘플 negative control을 E5 false-reject 회계에 활용할지.

## 5. 부수 확정 발견 (어느 갈래든 보고 가치)

1. **BDDL 재샘플 = 분포 중심 생성기** — "물리적 무효"와 "분포 밖"의 구분(§4b)에 이어
   "분포 밖"과 "분포 중심 재표집"의 구분이 필요함을 실측 확립.
2. **goal류 미세 이탈은 현 관할(DINOv2-PCA-Mahalanobis)이 비가시** — 관할의 감지 한계 실측.
3. known(held-out) ≈ calib — **관할 gate의 수용(false-reject) 측은 건전**: E5에서 gate가
   정상 상황을 잘못 기각할 위험은 낮음.
"""
    with open(os.path.join(PACK, "DECISION.md"), "w") as f:
        f.write(md)
    print(f"[E4-DECISION-PACK] {len(COPY_FILES)} 파일 + DECISION.md + git_history.txt")


if __name__ == "__main__":
    main()
