"""E4 종결 보고 패키지 생성기 (연구원 요청 2026-08-16) — 수치 전부 프로그래밍 주입.

실행: hv2_hab python -u experiments/make_e4_closure_pack.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "e4_closure_pack")

COPY_FILES = [
    "results/e4/e4_readjudication.json", "results/e4/e4_scorer_table.json",
    "results/e4/e4_scorer_diag.json", "results/e4/e4_hidden_scorer.json",
    "results/e4/e4_knn_scorer.json", "results/e4/e4_pilot_auc.json",
    "results/e4/e4_decision_support.json", "results/e4/novel_manifest.json",
    "results/e4/novel_manifest_v1_negative.json",
    "docs/PAPER2_E5_POSTHOC_CANDIDATES.md", "docs/E5_DRIVER_CHECKLIST.md",
    "configs/preregistration.md", "log.md",
]


def load(p):
    return json.load(open(os.path.join(HABIT2, p)))


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY_FILES:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-35"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)

    rj = load("results/e4/e4_readjudication.json")
    tb = load("results/e4/e4_scorer_table.json")
    r = tb["rows"]
    g = tb["same_cell_comparison"]["goal4"]  # 정정 1: 판독 근거 = 동일 셀 goal4

    def row(key, label, extra=""):
        v = r[key]
        auc_v = v.get("macro_auc", v.get("macro_auc_scope6"))
        fr = v.get("mean_fr", "—")
        return f"| {label} | {v.get('cost_ms', '—')}ms | {auc_v} | {fr} | {extra} |"

    md = f"""# E4 종결 보고 ({datetime.now().date()}) — 수치 프로그래밍 주입

## 1. 정식 재판정 (1회 약정) — **REDUCE 확정**

- macro AUC = **{rj['macro_auc']}** (임계 {rj['threshold']}, 미달 {rj['n_below_threshold']}/25)
- 스위트: {json.dumps(rj['per_suite_macro'], ensure_ascii=False)}
- 경로: {json.dumps(rj['per_path_macro'], ensure_ascii=False)}
- 운용 q false-reject 평균 = {rj['mean_false_reject']} / negative control(재샘플) 수용률 = {rj['mean_negative_control_accept']}
- §3 우아한 퇴화 발동 → E5 = 성숙도 단독 + 그림자 관할 로깅 (§5 등재).

## 2. scorer 진단 표 (6행)

| 행 | 비용 | macro AUC | mean FR | 비고 |
|---|---|---|---|---|
{row('1_jurisdiction_mahalanobis', '1 관할 Mahalanobis', f"동일셀 goal4 {g['jurisdiction']}")}
{row('2_feasibility_head', '2 feasibility head', '실패 I₀ 재렌더 학습')}
{row('3_hidden_L32_visual_mean', '3 히든 L32 visual-mean', f"동일셀 goal4 {g['hidden_L32']} — 6셀 범위")}
{row('4_oracle_band_gt', '4 oracle (대역 GT)', '명목 상한')}
{row('5_jurisdiction_recalibrated', '5 관할+q 재보정', f"FR {r['5_jurisdiction_recalibrated']['mean_fr_before']}→{r['5_jurisdiction_recalibrated']['mean_fr']}, AUC 불변")}
{row('6_knn_jurisdiction_k5', '6 kNN k=5', f"k10 {r['6_knn_jurisdiction_k10']['macro_auc']}")}

각주 (범위 불일치): {tb['footnote_scope_mismatch']}

### 2b. ★동일 셀 비교 (정정 1 — 주 판독 근거)

| 범위 | 관할 | 히든 L32 | Δ(히든) | kNN k5 | Δ(kNN) |
|---|---|---|---|---|---|
| 히든 6셀 | {tb['same_cell_comparison']['scope6']['jurisdiction']} | {tb['same_cell_comparison']['scope6']['hidden_L32']} | **{tb['same_cell_comparison']['scope6']['delta_hidden_minus_jur']:+}** | {tb['same_cell_comparison']['scope6']['knn_k5']} | {tb['same_cell_comparison']['scope6']['delta_knn5_minus_jur']:+} |
| goal 4셀 | {tb['same_cell_comparison']['goal4']['jurisdiction']} | {tb['same_cell_comparison']['goal4']['hidden_L32']} | **{tb['same_cell_comparison']['goal4']['delta_hidden_minus_jur']:+}** | {tb['same_cell_comparison']['goal4']['knn_k5']} | {tb['same_cell_comparison']['goal4']['delta_knn5_minus_jur']:+} |

## 3. H3 최종 문구 = **{tb['h3_branch']['verdict']}**

{tb['h3_branch']['statement']}

## 4. E5 설계 반영 (§5 기등재)

주 arm = 성숙도 단독 / 관할 단독 arm 폐지 / 그림자 관할 로깅(불개입, +4.0ms, q = 재보정
절차 — FR {r['5_jurisdiction_recalibrated']['mean_fr']} 실증). [등재만] 후보 목록 = PAPER2_E5_POSTHOC_CANDIDATES.md (실행 금지).

## 5. 다음 마일스톤

E5 드라이버 설계서 초안 (코드 금지) → 함정 체크리스트 6렌즈(E5_DRIVER_CHECKLIST.md 동봉)
→ 대화 검토 패키지 제출 → 설계 리뷰 통과 후 구현 착수.
"""
    with open(os.path.join(PACK, "REPORT.md"), "w") as f:
        f.write(md)
    print(f"[E4-CLOSURE-PACK] {len(COPY_FILES)} 파일 + REPORT.md + git_history.txt")


if __name__ == "__main__":
    main()
