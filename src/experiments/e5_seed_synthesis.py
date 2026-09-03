"""E5 3 seed 종합 — 평균±산포 (연구원 판정 2026-08-19 판독 규칙 4).

seed별 H4a·H4b는 각 `reading_{s}.json`에서 이미 산출됐고, 본 스크립트는 그것을 합산만 한다.
수치 수동 입력 금지 — 모든 값은 results/e5/reading_*.json에서 읽는다.

산출: results/e5/seed_synthesis.json
실행: hv2_hab python -u experiments/e5_seed_synthesis.py
"""
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(HABIT2)
SEEDS = [0, 1, 2]


def ms(v, nd=4):
    """평균±표준편차 (n=3이므로 표본 표준편차, ddof=1)."""
    a = np.asarray(v, float)
    return {"mean": round(float(a.mean()), nd), "sd": round(float(a.std(ddof=1)), nd),
            "values": [round(float(x), nd) for x in a]}


def main():
    R = {s: json.load(open(f"results/e5/reading_{s}.json")) for s in SEEDS}
    C = {s: json.load(open(f"results/e5/cf_summary_{s}.json")) for s in SEEDS}
    a = {s: R[s]["H4a_call_rate_reduction"] for s in SEEDS}
    b = {s: R[s]["H4b_noninferiority"] for s in SEEDS}
    rc = {s: R[s]["risk_control"] for s in SEEDS}
    ov = {s: R[s]["overview"] for s in SEEDS}
    fm = {s: R[s]["formation_ledger"] for s in SEEDS}
    md = {s: R[s]["maturity_dual_report"] for s in SEEDS}
    dem = {s: R[s]["demotions"] for s in SEEDS}
    life = {s: R[s]["lifecycle"] for s in SEEDS}

    all_pass = all(a[s]["verdict"] == "PASS" for s in SEEDS) and \
               all(b[s]["verdict"] == "PASS" for s in SEEDS) and \
               all(rc[s]["within_bound"] for s in SEEDS)
    # H4b CI가 0을 포함하는가 (연구원 판독 규칙: 포함 시 "구별되지 않음" 서술 가능)
    ci_zero = {s: bool(b[s]["ci95"][0] <= 0 <= b[s]["ci95"][1]) for s in SEEDS}

    out = {
        "n_seeds": len(SEEDS),
        "prereg_rule": "§5 2026-08-19 판독 4 — seed별 H4a·H4b 각각 산출, 종합은 평균±산포. "
                       "어느 seed에서든 판정이 뒤집히면 즉시 정지·보고(재량 진행 금지).",
        "verdict": "3/3 seed 전 항목 PASS" if all_pass else "★ 판정 불일치 — 정지 대상",
        "H4a_call_rate": {
            "per_seed": {str(s): {"first1000": a[s]["p_first"], "last1000": a[s]["p_last"],
                                  "delta": a[s]["diff"], "z": a[s]["z"], "p": a[s]["p_report"],
                                  "verdict": a[s]["verdict"]} for s in SEEDS},
            "first1000": ms([a[s]["p_first"] for s in SEEDS]),
            "last1000": ms([a[s]["p_last"] for s in SEEDS]),
            "delta": ms([a[s]["diff"] for s in SEEDS]),
            "all_pass": all(a[s]["verdict"] == "PASS" for s in SEEDS)},
        "H4b_noninferiority": {
            "per_seed": {str(s): {"system": b[s]["system_rate"], "full_vla": b[s]["full_vla_rate"],
                                  "diff": b[s]["diff"], "ci95": b[s]["ci95"],
                                  "n_paired": b[s]["n_paired_episodes"],
                                  "cf_missing": b[s]["n_cf_missing"],
                                  "ci_contains_zero": ci_zero[s],
                                  "verdict": b[s]["verdict"]} for s in SEEDS},
            "diff": ms([b[s]["diff"] for s in SEEDS]),
            "system_rate": ms([b[s]["system_rate"] for s in SEEDS]),
            "full_vla_rate": ms([b[s]["full_vla_rate"] for s in SEEDS]),
            "margin": b[0]["margin"],
            "all_pass": all(b[s]["verdict"] == "PASS" for s in SEEDS),
            "all_ci_contain_zero": all(ci_zero.values()),
            "ci_upper_min": round(min(b[s]["ci95"][1] for s in SEEDS), 4),
            "equivalence_caveat": "CI가 0을 포함하면 '통계적으로 구별되지 않음' 서술이 가능하나 "
                                  "**동등성 검정을 별도로 수행하지 않았음**을 각주로 명시할 것 "
                                  "(§5 2026-08-19). 상한이 0에 근접한 seed는 이 서술이 약하다."},
        "risk_control": {
            "per_seed": {str(s): {"pr_fail_given_fire": rc[s]["pr_fail_given_fire"],
                                  "ci95": rc[s]["ci95_wilson"]} for s in SEEDS},
            "pr_fail_given_fire": ms([rc[s]["pr_fail_given_fire"] for s in SEEDS]),
            "epsilon": rc[0]["epsilon"],
            "all_within_bound": all(rc[s]["within_bound"] for s in SEEDS)},
        "system": {
            "r_V_overall": ms([ov[s]["r_V_overall"] for s in SEEDS]),
            "system_success_rate": ms([ov[s]["system_success_rate"] for s in SEEDS]),
            "n_fire": ms([ov[s]["n_fire"] for s in SEEDS], 1),
            "fire_success_rate": ms([ov[s]["fire_success_rate"] for s in SEEDS])},
        "formation": {
            "n_retrain": ms([fm[s]["n_retrain"] for s in SEEDS], 1),
            "n_passed": ms([fm[s]["n_passed"] for s in SEEDS], 1),
            "pass_rate": ms([fm[s]["pass_rate"] for s in SEEDS]),
            "by_grid_pass_rate": {n: ms([fm[s]["by_grid_n"][n]["pass_rate"] for s in SEEDS])
                                  for n in ("20", "80")},
            "n_matured": ms([md[s]["n_reached_maturity"] for s in SEEDS], 1),
            "exposures_to_maturity_median": ms([md[s]["exposures_to_maturity_median"] for s in SEEDS], 1),
            "final_states": {str(s): {k: sum(1 for v in life[s].values() if v["final_state"] == k)
                                      for k in "MIX"} for s in SEEDS},
            "n_demotions": ms([dem[s]["n_demotions"] for s in SEEDS], 1),
            "n_regained": ms([dem[s]["n_regained"] for s in SEEDS], 1)},
        "counterfactual": {
            "per_seed": {str(s): {"n_paired": C[s]["n_paired"],
                                  "habit": C[s]["habit_success_rate"],
                                  "teacher": C[s]["teacher_success_rate"],
                                  "determinism": f"{sum(1 for x in C[s]['determinism_check'] if x['match'])}"
                                                 f"/{len(C[s]['determinism_check'])}"} for s in SEEDS},
            "total_paired": sum(C[s]["n_paired"] for s in SEEDS)},
        "ledgers_hours": {
            "operational": ms([json.load(open(f"results/e5/summary_{s}.json"))["ledger_s"]["operational_s"] / 3600
                               for s in SEEDS], 2),
            "formation": ms([json.load(open(f"results/e5/summary_{s}.json"))["ledger_s"]["formation_s"] / 3600
                             for s in SEEDS], 2),
            "note": "지연 주장은 운영 장부 단독. 형성은 별도 보고, 평가(CF)는 비용 미보고 (§4h)"},
    }
    with open("results/e5/seed_synthesis.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[E5SYN-DONE] results/e5/seed_synthesis.json")
    print(f"판정: {out['verdict']}")
    h4a, h4b, r = out["H4a_call_rate"], out["H4b_noninferiority"], out["risk_control"]
    print(f"  H4a r_V {h4a['first1000']['mean']}±{h4a['first1000']['sd']} → "
          f"{h4a['last1000']['mean']}±{h4a['last1000']['sd']} (Δ {h4a['delta']['mean']}±{h4a['delta']['sd']})")
    print(f"  H4b diff {h4b['diff']['mean']:+.4f}±{h4b['diff']['sd']:.4f} · "
          f"CI 상한 최솟값 {h4b['ci_upper_min']:+.4f} · 전 seed CI 0 포함 {h4b['all_ci_contain_zero']}")
    print(f"  위험 {r['pr_fail_given_fire']['mean']}±{r['pr_fail_given_fire']['sd']} ≤ {r['epsilon']}")


if __name__ == "__main__":
    main()
