"""§6 산출 + §7 배치 통계 (RGB-only). 논문 문장·그림 없음 — 숫자와 원자료만.

추정량은 **기존 스크립트의 함수를 그대로 import**해 쓴다(중복 구현 금지):
  e3_collect: wilson · one_sided_decrease   /   e3_h2_analysis: censored_ranks ·
  rank_decomposition · regression · collinearity_appendix
→ 프로토콜 동일성이 코드 수준에서 보장된다. 변경은 입력 곡선이 RGB-only라는 것뿐.

산출 (01_batch_formation/ · 08_statistics/):
  batch_episode_results.csv · batch_summary.csv · NSTAR_RESULTS.csv · batch_statistics.json
  rgb_only_e3_curves.json (e3_curves.json과 동일 스키마 — 하류 재사용용)
실행: hv2_hab python -u experiments/rgb_only_rerun/analyze_batch.py
마커: [BATCH-STATS-DONE]
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import binom

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from envs.stream import heldout_specs  # noqa: E402
from experiments.e3_collect import (EXPECTED_CLUSTERS, T1, T2_CHAINED,  # noqa: E402
                                    T3_LONG, one_sided_decrease, wilson)
from experiments.e3_h2_analysis import (CENSOR_CAP, LEVEL_OF,  # noqa: E402
                                        collinearity_appendix, rank_decomposition, regression)
from experiments.rgb_only_rerun.run_batch import CKROOT, CURVES, N_HELDOUT, parse  # noqa: E402
from experiments.rgb_only_rerun.runner import ROOT  # noqa: E402

GRID = [10, 20, 40, 80]
BAR = 0.8
OUT_B = f"{ROOT}/01_batch_formation"
OUT_S = f"{ROOT}/08_statistics"
TRAIN_SEED = 0            # habits.train.HP["seed"] — 전 클러스터 동일 (동결)


def load_raw(cl):
    p = f"{CURVES}/{cl}_curve.json"
    return json.load(open(p)) if os.path.exists(p) else None


def e3_view(cl, raw):
    """E3 프로토콜 관점의 곡선 — n_heldout=50인 클러스터는 **앞 20 스펙으로 절단**한다
    (기존 e3_collect.load_curve_from_e2와 동일 규칙). chained는 절단하지 않는다(원래 50)."""
    if raw["n_heldout"] == 20 or cl in T2_CHAINED:
        return {c["n"]: c for c in raw["curve"] if c.get("s_hat") is not None}
    suite, task, _ = parse(cl)
    keep = {s.uid for s in heldout_specs(suite, task, 20)}
    out = {}
    for c in raw["curve"]:
        eps = [e for e in c.get("per_episode", []) if e["uid"] in keep
               and e["outcome"] != "infra_error"]
        if not eps:
            continue
        k = sum(1 for e in eps if e["outcome"] == "success")
        out[c["n"]] = {"n": c["n"], "n_eval": len(eps), "n_success": k,
                       "s_hat": round(k / len(eps), 4), "per_episode": eps}
    return out


def nstar_of(curve):
    for n in GRID:
        if n in curve and curve[n]["s_hat"] >= BAR:
            return n
    return ">80"


def main():
    os.makedirs(OUT_B, exist_ok=True)
    os.makedirs(OUT_S, exist_ok=True)

    raws, missing = {}, []
    for cl in EXPECTED_CLUSTERS:
        r = load_raw(cl)
        (raws.__setitem__(cl, r) if r else missing.append(cl))
    if missing:
        print(f"[BATCH-STATS-WARN] 곡선 누락 {len(missing)}: {missing}")

    # ---------------- §6 per-episode / summary / N*
    ep_rows, sum_rows, ns_rows = [], [], []
    curves_e3, curves_full = {}, {}
    for cl, raw in raws.items():
        suite, task, chained = parse(cl)
        c_e3 = e3_view(cl, raw)
        curves_e3[cl] = c_e3
        curves_full[cl] = {c["n"]: c for c in raw["curve"] if c.get("s_hat") is not None}
        for c in raw["curve"]:
            if c.get("s_hat") is None:
                continue
            ck = os.path.join(CKROOT, cl, f"act_n{c['n']}.pt")
            for e in c["per_episode"]:
                ep_rows.append({
                    "cluster_id": cl, "suite": suite, "task_id": task, "n": c["n"],
                    "training_seed": TRAIN_SEED, "eval_episode_id": e["uid"],
                    "initial_state_id": e["uid"],   # spec uid = 초기상태 명세 식별자
                    "success": int(e["outcome"] == "success"),
                    "outcome": e["outcome"], "steps": e.get("steps"),
                    "stage": e.get("stage"),
                    "checkpoint_path": os.path.relpath(ck, HABIT2),
                    "n_heldout_protocol": raw["n_heldout"],
                    "in_e3_view": int(e["uid"] in {x["uid"] for x in
                                                   c_e3.get(c["n"], {}).get("per_episode", [])})
                    if c["n"] in c_e3 else 0,
                })
            sum_rows.append({"cluster_id": cl, "suite": suite, "task_id": task, "n": c["n"],
                             "num_trials": c["n_eval"], "num_success": c["n_success"],
                             "success_rate": c["s_hat"], "n_infra_error": c["n_infra_error"],
                             "protocol_heldout": raw["n_heldout"]})
        ns = nstar_of(c_e3)
        n80 = c_e3.get(80, {})
        ns_rows.append({"cluster_id": cl, "suite": suite, "task_id": task,
                        "level": LEVEL_OF.get(suite, "T3" if not chained else "T2_chain"),
                        "N_star": ns, "right_censored": int(ns == ">80"),
                        "formable": int(ns != ">80"),
                        "s_hat_10": c_e3.get(10, {}).get("s_hat"),
                        "s_hat_20": c_e3.get(20, {}).get("s_hat"),
                        "s_hat_40": c_e3.get(40, {}).get("s_hat"),
                        "s_hat_80": n80.get("s_hat"),
                        "n_eval_80": n80.get("n_eval"),
                        "n_success_80": n80.get("n_success"),
                        "wilson_80_lo": wilson(n80.get("n_success", 0), n80.get("n_eval", 0))[0],
                        "wilson_80_hi": wilson(n80.get("n_success", 0), n80.get("n_eval", 0))[1]})

    def dump_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump_csv(f"{OUT_B}/batch_episode_results.csv", ep_rows)
    dump_csv(f"{OUT_B}/batch_summary.csv", sum_rows)
    dump_csv(f"{OUT_B}/NSTAR_RESULTS.csv", ns_rows)

    # ---------------- e3_curves.json 동일 스키마 (하류 재사용)
    curves_json = {"clusters": {}, "n_star": {}, "n_clusters_reported": len(raws),
                   "completeness": {"expected": len(EXPECTED_CLUSTERS),
                                    "missing": missing, "unexpected": []},
                   "status": "COMPLETE" if not missing else f"PARTIAL({len(raws)})",
                   "modality": "rgb_only"}
    for cl, c in curves_e3.items():
        e = {"curve": {str(n): c[n]["s_hat"] for n in GRID if n in c},
             "wilson": {str(n): wilson(c[n]["n_success"], c[n]["n_eval"])
                        for n in GRID if n in c},
             "N_star": nstar_of(c)}
        curves_json["clusters"][cl] = e
        curves_json["n_star"][cl] = e["N_star"]
    json.dump(curves_json, open(f"{OUT_S}/rgb_only_e3_curves.json", "w"),
              indent=1, ensure_ascii=False)

    # ---------------- §7 통계 (기존 추정량 재사용)
    cov = json.load(open("results/e3/covariates.json"))
    rows = []
    for cl, c in curves_e3.items():
        if cl.startswith("chained_"):
            continue
        cv = cov["clusters"].get(cl)
        if not cv:
            continue
        rows.append({"cluster": cl, "suite": cv["suite"], "level": LEVEL_OF.get(cv["suite"], "T3"),
                     "N_star": nstar_of(c), "free_joints": cv["free_joints"],
                     "S_V_cluster": cv["S_V_cluster"], "median_len": cv["median_len_success"]})
    formation = [r for r in rows if r["level"] in ("L1_2", "L3", "L4a")]

    stats = {
        "modality": "rgb_only",
        "note": ("추정량은 e3_collect·e3_h2_analysis의 함수를 그대로 import해 적용했다. "
                 "공변량(free_joints·S_V_cluster·median_len)은 teacher/환경 속성이라 §1 동결 "
                 "대상이며 results/e3/covariates.json을 재사용한다."),
        "covariate_source": "results/e3/covariates.json (frozen, teacher-side)",
        "n_clusters": len(raws), "n_formation_cells": len(formation),
        "censor_cap": CENSOR_CAP, "perm_B": 10000, "perm_seed": 0,
        "decomposition_L": rank_decomposition(rows),
        "regression_formation22": regression(formation, "formation cells (L1_2+L3+L4a)"),
        "regression_all_standard": regression(rows, "all standard incl. libero_10 (suite dummy)"),
        "collinearity_appendix": collinearity_appendix(formation),
    }

    # --- horizon: T1 vs T3 단측 (Fisher/two-prop 자동 선택 — e3_collect와 동일 함수)
    def pool80(pairs):
        k = n = 0
        for suite, task in pairs:
            c = curves_e3.get(f"{suite}_task{task}")
            if c and 80 in c:
                k += c[80]["n_success"]
                n += c[80]["n_eval"]
        return k, n

    k1, n1 = pool80(T1)
    k3, n3 = pool80(T3_LONG)
    if n1 and n3:
        p, method = one_sided_decrease(k1, n1, k3, n3)
        stats["horizon_T1_vs_T3"] = {"s80_T1": round(k1 / n1, 4), "k_T1": k1, "n_T1": n1,
                                     "s80_T3": round(k3 / n3, 4), "k_T3": k3, "n_T3": n3,
                                     "p_one_sided_decrease": p, "method": method}

    # --- controlled chain: 곱 기준선 단측 이항 + 모수 부트스트랩 (e3_collect §4e와 동일)
    rng = np.random.default_rng(0)
    B = 10_000
    chains = {}
    for ch in T2_CHAINED:
        t1_cl = ch.replace("chained_", "")           # 곱 기준선의 T1 참조 = 같은 태스크 단일
        c1 = curves_full.get(t1_cl)                  # E2 프로토콜 50-ep 곡선
        c2 = curves_full.get(ch)
        if not (c1 and 80 in c1 and c2 and 80 in c2):
            chains[ch] = {"status": "MISSING"}
            continue
        k_t1, n_t1 = c1[80]["n_success"], c1[80]["n_eval"]
        s1 = k_t1 / n_t1
        p0 = s1 * s1
        k, n = c2[80]["n_success"], c2[80]["n_eval"]
        sc = k / n
        dec_p, dec_m = one_sided_decrease(k_t1, n_t1, k, n)
        s1_b = rng.binomial(n_t1, s1, B) / n_t1
        sc_b = rng.binomial(n, sc, B) / n
        delta = sc_b - s1_b**2
        chains[ch] = {"t1_ref_cluster": t1_cl, "s80_T1_ref50": round(s1, 4), "n_T1": n_t1,
                      "product_baseline_p0": round(p0, 4), "s80_chain": round(sc, 4),
                      "n_chain": n, "k_chain": k,
                      "wilson_chain_80": wilson(k, n),
                      "p_below_product": round(float(binom.cdf(k, n, p0)), 4),
                      "p_above_product": round(float(binom.sf(k - 1, n, p0)), 4),
                      "bootstrap": {"B": B, "seed": 0,
                                    "P_delta_below_0": round(float((delta < 0).mean()), 4),
                                    "delta_mean": round(float(delta.mean()), 4),
                                    "delta_ci95": [round(float(np.percentile(delta, 2.5)), 4),
                                                   round(float(np.percentile(delta, 97.5)), 4)]},
                      "secondary_p_decrease_vs_T1": round(dec_p, 4) if dec_p is not None else None,
                      "secondary_method": dec_m}
    stats["controlled_chain_product_baseline"] = chains

    # --- 재계산용 중간 입력 테이블 (§7)
    stats["intermediate_inputs"] = {
        "rank_regression_rows": rows,
        "formation_cells": [r["cluster"] for r in formation],
        "nstar_table": {r["cluster_id"]: r["N_star"] for r in ns_rows},
        "n80_counts": {cl: {"k": c[80]["n_success"], "n": c[80]["n_eval"]}
                       for cl, c in curves_e3.items() if 80 in c},
    }
    json.dump(stats, open(f"{OUT_B}/batch_statistics.json", "w"), indent=1, ensure_ascii=False)

    d = stats["decomposition_L"]
    print(f"[BATCH-STATS-DONE] clusters={len(raws)}/{len(EXPECTED_CLUSTERS)} "
          f"episodes={len(ep_rows)} formation_cells={len(formation)} "
          f"between={d['between_share']} KW_H={d['kruskal_H']} KW_p={d['kruskal_p']} "
          f"censored={sum(r['right_censored'] for r in ns_rows)}")


if __name__ == "__main__":
    main()
