"""E4 scorer 표 조립 (정정판 — 연구원 판정 2026-08-16).

정정 1: **동일 셀 비교 열** 신설 — 히든 6셀 / goal 4셀 범위에서 관할·kNN·히든을 같은
        클러스터 집합으로 재집계. 25셀 vs 6셀 비교는 범위 불일치이므로 각주로 강등.
정정 2: H3 최종 문구 = (i)+(ii) 혼합 (단독 (ii) 폐기).
정정 3: feasibility head = "열등"이 아니라 **성공률 96% 레짐의 학습 신호 희소(양성 ~3%)**
        조건부 — 조건 변경 시 재평가 대상으로 명기.

산출: results/e4/e4_scorer_table.json
실행: hv2_hab python -u experiments/e4_scorer_table.py
"""
import json
import os

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E4 = os.path.join(HABIT2, "results", "e4")


def load(name):
    return json.load(open(os.path.join(E4, name)))


def main():
    rj = load("e4_readjudication.json")
    diag = load("e4_scorer_diag.json")
    hid = load("e4_hidden_scorer.json")
    knn = load("e4_knn_scorer.json")

    scope6 = hid["scope"]
    goal4 = [c for c in scope6 if c.startswith("libero_goal")]

    def mean_jur(cells):
        return round(float(np.mean([rj["clusters"][c]["auc_primary"] for c in cells])), 4)

    def mean_hid(cells):
        return round(float(np.mean([hid["clusters"][c]["auc"] for c in cells])), 4)

    def mean_knn(cells, k):
        return round(float(np.mean([knn["per_cluster"][c][f"k{k}"]["auc"] for c in cells])), 4)

    same_cell = {
        "note": "정정 1 (연구원 판정): 동일 클러스터 집합 기준 비교. 25셀 macro와 6셀 범위의 "
                "직접 비교는 범위 불일치 — 각주 처리.",
        "scope6_cells": scope6, "goal4_cells": goal4,
        "scope6": {"jurisdiction": mean_jur(scope6), "hidden_L32": mean_hid(scope6),
                   "knn_k5": mean_knn(scope6, 5), "knn_k10": mean_knn(scope6, 10),
                   "delta_hidden_minus_jur": round(mean_hid(scope6) - mean_jur(scope6), 4),
                   "delta_knn5_minus_jur": round(mean_knn(scope6, 5) - mean_jur(scope6), 4)},
        "goal4": {"jurisdiction": mean_jur(goal4), "hidden_L32": mean_hid(goal4),
                  "knn_k5": mean_knn(goal4, 5), "knn_k10": mean_knn(goal4, 10),
                  "delta_hidden_minus_jur": round(mean_hid(goal4) - mean_jur(goal4), 4),
                  "delta_knn5_minus_jur": round(mean_knn(goal4, 5) - mean_jur(goal4), 4)},
    }

    feas = dict(diag["rows"]["2_feasibility_head"])
    feas["note"] = ("정정 3 (연구원 판정): '열등'이 아니라 **성공률 96% 레짐의 학습 신호 희소** "
                    "(양성 ~3%, 실패 I₀ 재렌더로 복원해도 소수) — 조건부 결과. teacher 성공률이 "
                    "낮거나 실패 표본이 풍부한 레짐에서는 재평가 대상.")

    s6, g4 = same_cell["scope6"], same_cell["goal4"]
    h3 = {
        "verdict": "(i)+(ii) 혼합 — 단독 (ii) 폐기 (연구원 판정 2026-08-16)",
        "statement": ("기하 관할은 **표현 비용에 종속적**이며(히든 L32가 동일 셀에서 "
                      f"goal +{g4['delta_hidden_minus_jur']} / 6셀 +{s6['delta_hidden_minus_jur']}, "
                      f"반면 집계만 바꾼 kNN은 goal {g4['delta_knn5_minus_jur']:+.4f} / 6셀 "
                      f"{s6['delta_knn5_minus_jur']:+.4f}로 무효과 → 원인은 표현이지 집계가 아님), "
                      "그럼에도 21× 비용(85.07 vs 4.0ms)에도 임계 0.75 미달 → "
                      "**저비용 실시간 관할은 미해결**."),
        "evidence_basis": {"same_cell_goal4": g4, "same_cell_scope6": s6,
                           "cost_ratio": round(85.07 / 4.0, 1),
                           "categorical_shift_demonstrated": rj["per_path_macro"]["borrow__primary"]},
    }

    table = {
        "note": "E4 scorer 진단 표 (정정판). AUC = known(held-out) vs novel주. FR = false-reject. "
                "행별 범위 상이 — 동일 셀 비교는 same_cell_comparison 참조 (주 판독 근거).",
        "rows": {
            "1_jurisdiction_mahalanobis": diag["rows"]["1_jurisdiction_mahalanobis"],
            "2_feasibility_head": feas,
            "3_hidden_L32_visual_mean": {
                "cost_ms": 85.07, "macro_auc_scope6": hid["macro_auc_scope"],
                "goal_macro_auc": hid["goal_macro_auc"], "mean_fr": hid["mean_fr"],
                "scope": scope6, "note": "6셀 범위 (예산 제약 — 25셀 macro와 직접 비교 금지)"},
            "4_oracle_band_gt": diag["rows"]["4_oracle_band_gt"],
            "5_jurisdiction_recalibrated": diag["rows"]["5_jurisdiction_recalibrated"],
            "6_knn_jurisdiction_k5": knn["rows"]["6_knn_jurisdiction_k5"],
            "6_knn_jurisdiction_k10": knn["rows"]["6_knn_jurisdiction_k10"],
        },
        "same_cell_comparison": same_cell,
        "footnote_scope_mismatch": ("행1·2·5·6은 25셀 macro, 행3은 6셀 — 표의 macro 열끼리의 "
                                    "직접 비교는 범위 불일치. 판독은 same_cell_comparison으로 한다."),
        "h3_branch": h3,
    }
    with open(os.path.join(E4, "e4_scorer_table.json"), "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(f"[E4-TABLE-PASS] 동일셀 goal4: 관할 {g4['jurisdiction']} → 히든 {g4['hidden_L32']} "
          f"(Δ{g4['delta_hidden_minus_jur']:+}), kNN Δ{g4['delta_knn5_minus_jur']:+} | H3 = (i)+(ii)")


if __name__ == "__main__":
    main()
