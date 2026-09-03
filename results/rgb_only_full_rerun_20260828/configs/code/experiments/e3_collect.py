"""E3 집계 — 성숙 곡선 27 클러스터(distinct, §5 이력 2026-08-15), N*(L), T-천장 검정
(설계서 §4.3·§5 E3, preregistration §2·§4).

산출: results/e3/e3_curves.json (§6 단일 진입점 — 파일명 통일, R1d)
  - 완결성 27 구성원 (§5 최신 이력 기준): 표준 25 + chained {task0, task6}
    (task5 체인은 트리거 폐기 — 부록·negative 참조만, R1a)
  - 클러스터별 곡선 ŝ_k(n) + Wilson CI
  - 이원 보고: ① 클러스터 수준 N*(k) = min{n ∈ grid : ŝ ≥ 0.8} (미도달 >80 우측절단)
              ② 스트림 수준: 수집 스트림에서 N*번째 성공이 나온 에피소드 인덱스 (재발률×S_V 실효속도)
  - H2-L: 레벨(L1/2=object, L3=goal, L4a=spatial)별 N* 분포
  - H2-T (§4e 개정 2026-08-15):
      · 동역학 천장 주 증거 = T1 vs T3(Long 생태 앵커) 단측 감소 (pooled two-proportion)
      · C-T2 주 검정 = ŝ_chain(80) vs 곱 기준선 (ŝ_T1(80))² 단측 이항 (하회 = 결합 비용 /
        상회 = 시연 2배 효과 후보, 사전 등재). 원 기준(T1 vs T2 단측 감소)은 병행 보고

E2 재사용: object task0/task5의 E3 곡선(held-out 20)은 E2 per-episode 기록의 앞 20 스펙 절단으로 도출.
"""
import json
import os
import sys
from math import sqrt

from scipy.stats import binom, fisher_exact, norm

HABIT2 = "/home/asmr/workspace/habitvla2"
sys.path.insert(0, HABIT2)
OUT = os.path.join(HABIT2, "results", "e3", "e3_curves.json")  # §6 단일 진입점 (통합 지시서)
GRID = [10, 20, 40, 80]
BAR = 0.8

LEVELS = {
    "L1_2_object": [("libero_object", t) for t in range(10)],
    "L3_goal": [("libero_goal", t) for t in range(10)],
    "L4a_spatial": [("libero_spatial", 0), ("libero_spatial", 1)],
}
T3_LONG = [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
T1 = [("libero_object", 0), ("libero_object", 5)]  # T-스윕 T1 앵커 (C-L0·C-L1 대표 — 불변)
# α 판정 (§5 2026-08-15): 체인 = task0 + task5 **복원** (원 paired 설계). task6 경로는
# §5 이력으로만 존치 — 교체 논거(task5 앵커 오염)가 diag5/5b로 실행기 아티팩트로 재귀속.
T2_CHAINED = ["chained_libero_object_task0", "chained_libero_object_task5"]
# 곱 기준선의 T1 참조 (50 ep) — 양 태스크 모두 E2 50 곡선 (원 설계)
T2_T1_REF = {
    "chained_libero_object_task0": ("e2", "libero_object_task0_curve.json"),
    "chained_libero_object_task5": ("e2", "libero_object_task5_curve.json"),
}
# R1c: 완결성 = 정확히 이 27 구성원 (개수 아닌 집합 검사)
EXPECTED_CLUSTERS = (
    [f"libero_object_task{t}" for t in range(10)]
    + [f"libero_goal_task{t}" for t in range(10)]
    + ["libero_spatial_task0", "libero_spatial_task1"]
    + ["libero_10_task0", "libero_10_task2", "libero_10_task5"]
    + T2_CHAINED
)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def one_sided_decrease(k_hi, n_hi, k_lo, n_lo):
    """H1: p_lo < p_hi (천장 하강). 반환 p-value for decrease."""
    if min(n_hi, n_lo) == 0:
        return None, "undefined"
    pool = (k_hi + k_lo) / (n_hi + n_lo)
    exp = min(pool * n_hi, pool * n_lo, (1 - pool) * n_hi, (1 - pool) * n_lo)
    if exp < 5:
        _, p = fisher_exact([[k_lo, n_lo - k_lo], [k_hi, n_hi - k_hi]], alternative="less")
        return float(p), "fisher_less"
    se = sqrt(pool * (1 - pool) * (1 / n_hi + 1 / n_lo))
    if se == 0:
        return 1.0, "degenerate"
    z = (k_lo / n_lo - k_hi / n_hi) / se
    return float(norm.cdf(z)), "two_prop_z"


def load_curve_e3(cluster):
    p = os.path.join(HABIT2, "results", "e3", f"{cluster}_curve.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {c["n"]: c for c in d["curve"] if c.get("s_hat") is not None}


def load_curve_from_e2(cluster, suite, task):
    """E2 곡선(50)의 앞 20 스펙 절단."""
    from envs.stream import heldout_specs

    p = os.path.join(HABIT2, "results", "e2", f"{cluster}_curve.json")
    if not os.path.exists(p):
        return None
    keep = {s.uid for s in heldout_specs(suite, task, 20)}
    d = json.load(open(p))
    out = {}
    for c in d["curve"]:
        if "per_episode" not in c:
            continue
        eps = [e for e in c["per_episode"] if e["uid"] in keep and e["outcome"] != "infra_error"]
        k = sum(1 for e in eps if e["outcome"] == "success")
        out[c["n"]] = {"n": c["n"], "n_eval": len(eps), "n_success": k,
                       "s_hat": round(k / len(eps), 4) if eps else None}
    return out


def stream_maturation(cluster, data_dir):
    """수집 메타에서: 스트림 수준 성숙 = N*번째 성공 궤적이 등장한 스트림 에피소드 인덱스."""
    import h5py

    p = os.path.join(HABIT2, "data", data_dir, f"{cluster}.hdf5")
    if not os.path.exists(p):
        return None
    with h5py.File(p, "r") as f:
        meta = json.loads(f["meta_json"][()])
    succ_positions = [i + 1 for i, m in enumerate(meta) if m["outcome"] == "success"]
    return succ_positions  # 1-indexed 스트림 위치; succ_positions[N*-1] = 스트림 성숙 시점


def main():
    report = {"clusters": {}, "n_star": {}, "levels": {}, "t_ceiling": {}, "status": "PARTIAL"}
    all_std = [(s, t) for lv in LEVELS.values() for (s, t) in lv] + T3_LONG

    for suite, task in all_std:
        cl = f"{suite}_task{task}"
        if (suite, task) in [("libero_object", 0), ("libero_object", 5)]:
            curve = load_curve_from_e2(cl, suite, task)
            data_dir = "e2"
        else:
            curve = load_curve_e3(cl)
            data_dir = "e3"
        if not curve:
            report["clusters"][cl] = {"status": "MISSING"}
            continue
        entry = {
            "curve": {str(n): curve[n]["s_hat"] for n in GRID if n in curve},
            "wilson": {str(n): wilson(curve[n]["n_success"], curve[n]["n_eval"]) for n in GRID if n in curve},
        }
        nstar = next((n for n in GRID if n in curve and curve[n]["s_hat"] >= BAR), None)
        entry["N_star"] = nstar if nstar is not None else ">80"
        sp = stream_maturation(cl, data_dir)
        if sp and nstar is not None and len(sp) >= nstar:
            entry["stream_episodes_to_N_star"] = sp[nstar - 1]
        report["clusters"][cl] = entry
        report["n_star"][cl] = entry["N_star"]

    # chained (있으면)
    for cl in T2_CHAINED:
        curve = load_curve_e3(cl)
        if curve:
            entry = {
                "curve": {str(n): curve[n]["s_hat"] for n in GRID if n in curve},
                "wilson": {str(n): wilson(curve[n]["n_success"], curve[n]["n_eval"]) for n in GRID if n in curve},
            }
            nstar = next((n for n in GRID if n in curve and curve[n]["s_hat"] >= BAR), None)
            entry["N_star"] = nstar if nstar is not None else ">80"
            report["clusters"][cl] = entry
            report["n_star"][cl] = entry["N_star"]

    # H2-L: 레벨별 N* 분포
    for lv, members in LEVELS.items():
        ns = [report["n_star"].get(f"{s}_task{t}") for s, t in members]
        ns = [n for n in ns if n is not None]
        report["levels"][lv] = {
            "N_star_values": ns,
            "median": sorted([80 * 2 if n == ">80" else n for n in ns])[len(ns) // 2] if ns else None,
            "n_censored": sum(1 for n in ns if n == ">80"),
        }

    # H2-T: ŝ(80) 천장 — T1 vs T3 (T2는 chained 곡선 존재 시 추가)
    def pool80(pairs, from_e2=False):
        k = n = 0
        for suite, task in pairs:
            cl = f"{suite}_task{task}"
            c = load_curve_from_e2(cl, suite, task) if (suite, task) in T1 else load_curve_e3(cl)
            if c and 80 in c:
                k += c[80]["n_success"]
                n += c[80]["n_eval"]
        return k, n

    k1, n1 = pool80(T1)
    k3, n3 = pool80(T3_LONG)
    if n1 and n3:
        p, method = one_sided_decrease(k1, n1, k3, n3)
        report["t_ceiling"]["T1_vs_T3"] = {
            "role": "동역학 천장 주 증거 (§4e 개정 — Long 생태 앵커)",
            "s80_T1": round(k1 / n1, 4), "s80_T3": round(k3 / n3, 4),
            "p_one_sided_decrease": p, "method": method,
        }

    # C-T2 주 검정 (§4e 개정 + held-out 50 확대): ŝ_chain(80, n=50) vs 곱 기준선 (ŝ_T1(80))²
    # — T1 기준 = **E2 50-ep 곡선 원본** (§5 이력: p₀ = 0.96² = 0.9216). 단측 이항 α=0.05 +
    # 모수 부트스트랩 병행 (B=10⁴, seed 0 — p₀ 추정 불확실성 보완, §4e 등재식).
    # 하회 = 장기 시퀀스 학습의 결합 비용 / 상회 = 시연 2배 효과 후보(사전 등재).
    # 원 기준(T1 vs T2 단측 감소)은 병행 보고 (secondary).
    def load_t1_ref(chain_cl):
        """체인별 T1 참조 곡선 (R1b): task0 = E2 50 / task6 = 싱글 h50 병합본."""
        sub, fname = T2_T1_REF[chain_cl]
        p = os.path.join(HABIT2, "results", sub, fname)
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        return {c["n"]: c for c in d["curve"] if c.get("s_hat") is not None}

    rng = __import__("numpy").random.default_rng(0)
    B = 10_000
    t2_tests = {}
    for cl_ch in T2_CHAINED:
        c1 = load_t1_ref(cl_ch)
        c2 = load_curve_e3(cl_ch)
        if not (c1 and 80 in c1 and c2 and 80 in c2):
            t2_tests[cl_ch] = {"status": "MISSING"}
            continue
        k_t1, n_t1 = c1[80]["n_success"], c1[80]["n_eval"]
        s1 = k_t1 / n_t1
        p0 = s1 * s1
        k, n = c2[80]["n_success"], c2[80]["n_eval"]
        sc = k / n
        dec_p, dec_m = one_sided_decrease(k_t1, n_t1, k, n)
        # 모수 부트스트랩: Δ* = ŝ_chain* − (ŝ_T1*)²
        s1_b = rng.binomial(n_t1, s1, B) / n_t1
        sc_b = rng.binomial(n, sc, B) / n
        delta = sc_b - s1_b**2
        t2_tests[cl_ch] = {
            "s80_T1_ref50": round(s1, 4),
            "t1_ref_source": "/".join(T2_T1_REF[cl_ch]),
            "n_T1": n_t1,
            "product_baseline_p0": round(p0, 4),
            "s80_chain": round(sc, 4),
            "n_chain": n,
            "wilson_chain_80": wilson(k, n),
            "p_below_product": round(float(binom.cdf(k, n, p0)), 4),
            "p_above_product": round(float(binom.sf(k - 1, n, p0)), 4),
            "bootstrap": {
                "B": B, "seed": 0,
                "P_delta_below_0": round(float((delta < 0).mean()), 4),
                "delta_mean": round(float(delta.mean()), 4),
                "delta_ci95": [round(float(v), 4) for v in
                               (sorted(delta)[int(0.025 * B)], sorted(delta)[int(0.975 * B) - 1])],
            },
            "secondary_p_decrease_vs_T1": round(dec_p, 4) if dec_p is not None else None,
            "secondary_method": dec_m,
        }
    report["t_ceiling"]["T2_product_baseline"] = t2_tests

    done = sum(1 for v in report["clusters"].values() if "curve" in v)
    report["n_clusters_reported"] = done
    # 완결성 (R1c) = 정확히 EXPECTED_CLUSTERS 27 집합 — 개수 일치가 아닌 구성원 검사
    have = {cl for cl, v in report["clusters"].items() if "curve" in v}
    missing = sorted(set(EXPECTED_CLUSTERS) - have)
    unexpected = sorted(have - set(EXPECTED_CLUSTERS))
    report["completeness"] = {"expected": len(EXPECTED_CLUSTERS), "missing": missing,
                              "unexpected": unexpected}
    report["status"] = "COMPLETE" if not missing and not unexpected else f"PARTIAL({done})"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}, indent=2, ensure_ascii=False))
    print(f"[E3-CURVES] status={report['status']} json={os.path.relpath(OUT, HABIT2)}")


if __name__ == "__main__":
    main()
