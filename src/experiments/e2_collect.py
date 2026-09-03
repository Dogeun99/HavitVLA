"""E2 go/no-go 집계 — ★유일 치명 단계의 사전등록 판정.

판정 (configs/preregistration.md §2 / 설계서 §5 E2):
  클러스터별로:
    (a) max_n ŝ(n) ≥ 0.8
    (b) ŝ(80) > ŝ(10) — 단측 two-proportion (pooled z), α = 0.05.
        held-out n < 40 또는 기대도수 < 5면 Fisher exact 단측으로 대체 (사전등록 §2).
  종합 = 두 클러스터(C-L0, C-L1rep) 모두 (a)∧(b) → GO.
  한쪽만 통과 → PARTIAL (연구원 판정 대상 — no-go 원인 트리 1회 반복 규칙 §3).
  둘 다 실패 → NO-GO (원인 트리: 데이터 품질 / HP / 관측 구성).

paired 구조(동일 held-out spec)이므로 two-proportion은 보수적 근사 — paired 구조를 활용한
McNemar 부가 보고를 함께 남긴다(판정은 사전등록된 two-proportion이 원본).
"""
import json
import os
from math import sqrt

from scipy.stats import fisher_exact, norm

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
OUT = os.path.join(HABIT2, "results", "e2", "e2_gonogo.json")
CLUSTERS = ["libero_object_task0", "libero_object_task5"]
ALPHA = 0.05
MATURITY_BAR = 0.8


def one_sided_two_prop(k1, n1, k0, n0):
    """H1: p1 > p0. pooled z. 소표본이면 Fisher 단측."""
    if min(n1, n0) == 0:
        return None, "undefined"
    p1, p0 = k1 / n1, k0 / n0
    pool = (k1 + k0) / (n1 + n0)
    exp = min(pool * n1, pool * n0, (1 - pool) * n1, (1 - pool) * n0)
    if exp < 5:
        _, p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")
        return float(p), "fisher_greater"
    se = sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
    if se == 0:
        return 1.0, "degenerate"
    z = (p1 - p0) / se
    return float(1 - norm.cdf(z)), "two_prop_z"


def mcnemar_report(curve):
    """paired 부가 보고: n=10 vs n=80 에피소드별 성공 여부 교차표."""
    by_n = {c["n"]: c for c in curve if "per_episode" in c}
    if 10 not in by_n or 80 not in by_n:
        return None
    o10 = {e["uid"]: e["outcome"] for e in by_n[10]["per_episode"] if e["outcome"] != "infra_error"}
    o80 = {e["uid"]: e["outcome"] for e in by_n[80]["per_episode"] if e["outcome"] != "infra_error"}
    common = set(o10) & set(o80)
    b = sum(1 for u in common if o10[u] == "fail" and o80[u] == "success")  # 개선
    c = sum(1 for u in common if o10[u] == "success" and o80[u] == "fail")  # 악화
    return {"n_pairs": len(common), "improved_b": b, "worsened_c": c}


def main():
    report = {"clusters": {}, "status": "FAIL"}
    verdicts = []
    for cl in CLUSTERS:
        path = os.path.join(HABIT2, "results", "e2", f"{cl}_curve.json")
        if not os.path.exists(path):
            report["clusters"][cl] = {"status": "MISSING"}
            verdicts.append(False)
            continue
        d = json.load(open(path))
        curve = [c for c in d["curve"] if c.get("s_hat") is not None]
        by_n = {c["n"]: c for c in curve}
        if not curve or 10 not in by_n or 80 not in by_n:
            report["clusters"][cl] = {"status": "INCOMPLETE", "curve": curve}
            verdicts.append(False)
            continue
        smax = max(c["s_hat"] for c in curve)
        c80, c10 = by_n[80], by_n[10]
        p, method = one_sided_two_prop(c80["n_success"], c80["n_eval"], c10["n_success"], c10["n_eval"])
        a_ok = smax >= MATURITY_BAR
        b_ok = p is not None and p < ALPHA and c80["s_hat"] > c10["s_hat"]
        entry = {
            "curve": {str(c["n"]): c["s_hat"] for c in curve},
            "max_s_hat": smax,
            "criterion_a_max_ge_0.8": bool(a_ok),
            "s80_vs_s10_p_one_sided": p,
            "test_method": method,
            "criterion_b_s80_gt_s10": bool(b_ok),
            "mcnemar_supplementary": mcnemar_report(d["curve"]),
            "status": "GO" if (a_ok and b_ok) else "NO-GO",
        }
        report["clusters"][cl] = entry
        verdicts.append(a_ok and b_ok)

    if all(verdicts):
        report["status"] = "GO"
    elif any(verdicts):
        report["status"] = "PARTIAL"  # 연구원 판정 대상
    else:
        report["status"] = "NO-GO"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[E2-PASS] item=E2 status={report['status']} json=results/e2/e2_gonogo.json")


if __name__ == "__main__":
    main()
