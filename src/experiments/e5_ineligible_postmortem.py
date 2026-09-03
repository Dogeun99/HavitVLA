"""부적격(X) 클러스터 사후 분석 — **탐색적**. 추가 rollout 없음(로그 분석만).

등재: configs/preregistration.md §5 2026-08-17 (seed 0 판독 3, 연구원 판정).
판독 규칙은 결과 산출 전에 고정됐고, 본 스크립트는 그 규칙만 집행한다.

  규칙 1  부적격 확정 후 BC 풀이 80을 크게 넘겨 축적된 클러스터 수
          → "재도전 규칙이 있었다면 성숙 가능성" 사후 추정 근거 (개수 명시)
  규칙 2  E3에서 N* ≤ 80이었는데 스트림에서 부적격이 된 클러스터
          → 배치·스트림의 형성 조건 차이를 별도 논점으로 등재
  규칙 3  전체를 "탐색적"으로 라벨해 사전등록 검정과 구분

산출: results/e5/ineligible_postmortem_{seed}.json
실행: hv2_hab python -u experiments/e5_ineligible_postmortem.py --seed-idx 0
"""
import argparse
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)

BC_SURPLUS_REF = 80   # 마지막 grid 지점 — "크게 넘겨 축적"의 기준선


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    args = ap.parse_args()
    rd = os.path.join(HABIT2, "results", "e5")
    rows = [json.loads(l) for l in open(os.path.join(rd, f"stream_{args.seed_idx}.jsonl"))]
    summary = json.load(open(os.path.join(rd, f"summary_{args.seed_idx}.json")))
    e3 = json.load(open(os.path.join(HABIT2, "results", "e3", "e3_curves.json")))
    n_star = e3["n_star"]

    inel = [c for c, s in summary["final_states"].items() if s == "X"]
    per = {}
    for cl in sorted(inel):
        cr = [r for r in rows if r["cluster"] == cl]
        # 부적격 확정 시점 = lifecycle_state가 처음 X가 된 에피소드
        t_x = next((r["t"] for r in cr if r["lifecycle_state"] == "X"), None)
        after = [r for r in cr if t_x is not None and r["t"] >= t_x]
        probes = [r["retrain_event"] for r in cr if r["retrain_event"]]
        bc_final = cr[-1]["bc_pool"]
        bc_at_x = next((r["bc_pool"] for r in cr if r["t"] == t_x), None)
        per[cl] = {
            "exposures": len(cr),
            "final_bc_pool": bc_final,
            "bc_pool_at_ineligible": bc_at_x,
            "t_ineligible": t_x,
            "probe_rounds": [{"round": p["probe_round"], "n": p["n"], "passed": p["passed"],
                              "formation_wall_s": p.get("formation_wall_s")} for p in probes],
            "teacher_successes_after_ineligible": sum(
                1 for r in after if r["outcome"] == "success" and r["executor"] == "vla"),
            "exposures_after_ineligible": len(after),
            "bc_growth_after_ineligible": (bc_final - bc_at_x) if bc_at_x is not None else None,
            "e3_N_star": n_star.get(cl),
            "e3_curve": e3["clusters"].get(cl, {}).get("curve"),
            "surplus_over_last_grid": bc_final - BC_SURPLUS_REF,
        }

    # ---- 규칙 1: 확정 후에도 BC가 마지막 grid 지점을 크게 넘겨 축적된 클러스터
    r1 = {c: v for c, v in per.items() if v["surplus_over_last_grid"] > 0}
    # ---- 규칙 2: E3에서 N* ≤ 80이었는데 스트림에서 부적격
    # N*는 미도달 시 ">80" 우측절단 문자열(CLAUDE.md §5) — 절단값은 "≤80"에 해당하지 않는다.
    def reached(v):
        return isinstance(v, (int, float))
    r2 = {c: {"e3_N_star": v["e3_N_star"], "e3_curve": v["e3_curve"],
              "stream_probe": v["probe_rounds"], "final_bc_pool": v["final_bc_pool"]}
          for c, v in per.items() if reached(v["e3_N_star"]) and v["e3_N_star"] <= 80}
    censored = {c: v["e3_N_star"] for c, v in per.items() if not reached(v["e3_N_star"])}

    # ---- 규칙 2의 원인 규명 (사후 진단): probe 성적 역산 + 라운드2 구제가능성
    # A_mat 장부는 probe·fire만 계상하므로 X 클러스터는 재학습 직전 σ=φ=0에서 출발한다.
    # 재학습 직후 σ = c·σ_before + probe_successes (c = 0.25 재초기화, §3.5).
    from scipy.stats import beta as _beta

    C_REINIT, P_PROBE, TAU, DELTA = 0.25, 20, 0.8, 0.1

    def p_ge_tau(s, f):
        return float(1 - _beta.cdf(TAU, 1 + s, 1 + f))

    # 라운드1(무이력) 통과에 필요한 최소 probe 성공 수
    r1_need = next(s for s in range(P_PROBE + 1) if p_ge_tau(s, P_PROBE - s) >= 1 - DELTA)
    # 라운드2에서 **만점(P/P)**을 받아도 통과하지 못하게 되는 라운드1 실패 수의 하한
    def r2_best_case(f1):
        s0, f0 = C_REINIT * (P_PROBE - f1), C_REINIT * f1
        return p_ge_tau(s0 + P_PROBE, f0)
    f1_hopeless = next((f1 for f1 in range(P_PROBE + 1) if r2_best_case(f1) < 1 - DELTA), None)

    probe_recon = {}
    for cl in sorted(inel):
        cr = [r for r in rows if r["cluster"] == cl]
        rec = []
        for i, r in enumerate(cr):
            if not r["retrain_event"]:
                continue
            prev = cr[i - 1] if i else None
            sb, fb = (prev["sigma_k"], prev["phi_k"]) if prev else (0.0, 0.0)
            succ = round(r["sigma_k"] - C_REINIT * sb)
            rec.append({"round": r["retrain_event"]["probe_round"], "n": r["retrain_event"]["n"],
                        "probe_successes": succ, "probe_failures": P_PROBE - succ,
                        "probe_rate": round(succ / P_PROBE, 3),
                        "p_ge_tau_after": r["p_ge_tau"], "passed": r["retrain_event"]["passed"]})
        e3c = per[cl]["e3_curve"]
        rd1 = next((x for x in rec if x["round"] == 1), None)
        probe_recon[cl] = {
            "rounds": rec,
            "e3_heldout_at_80": e3c.get("80") if e3c else None,
            "best_probe_rate": max((x["probe_rate"] for x in rec), default=None),
            # 라운드2가 시작 시점에 이미 결정돼 있었는가 (만점을 받아도 통과 불가)
            "round2_was_unwinnable": bool(rd1 and f1_hopeless is not None
                                          and rd1["probe_failures"] >= f1_hopeless)}

    unwinnable = [c for c, v in probe_recon.items() if v["round2_was_unwinnable"]]
    scored_max = [c for c, v in probe_recon.items()
                  if any(x["round"] == 2 and x["probe_successes"] == P_PROBE for x in v["rounds"])]

    # 잔여 VLA 호출에서 부적격이 차지하는 몫 (판정 지시의 동기 수치 재확인)
    tail = [r for r in rows[-1000:] if r["outcome"] != "infra_error"]
    tail_vla = [r for r in tail if r["executor"] == "vla"]
    x_share = sum(1 for r in tail_vla if r["lifecycle_state"] == "X") / max(len(tail_vla), 1)

    out = {
        "label": "탐색적 (post-hoc) — 사전등록 검정과 구분해 보고할 것 (§5 2026-08-17 규칙 3)",
        "provenance": "추가 rollout 0 — results/e5/stream_{}.jsonl + results/e3/e3_curves.json".format(args.seed_idx),
        "seed_idx": args.seed_idx,
        "n_ineligible": len(inel),
        "share_of_tail_vla_calls": round(x_share, 4),
        "per_cluster": per,
        "rule1_retry_would_have_helped": {
            "criterion": f"부적격 확정 후 최종 BC 풀 > {BC_SURPLUS_REF} (마지막 grid 지점)",
            "n_clusters": len(r1),
            "clusters": {c: {"final_bc_pool": per[c]["final_bc_pool"],
                             "surplus": per[c]["surplus_over_last_grid"],
                             "bc_growth_after_ineligible": per[c]["bc_growth_after_ineligible"]}
                         for c in r1},
            "median_surplus": int(np.median([v["surplus_over_last_grid"] for v in r1.values()])) if r1 else None},
        "rule2_batch_vs_stream_gap": {
            "criterion": "E3 배치에서 N* ≤ 80 (성숙 도달)인데 스트림에서 부적격",
            "n_clusters": len(r2),
            "clusters": r2,
            "e3_right_censored": censored,
            "cause_diagnosis": {
                "kind": "사후 진단 (규칙 2의 원인 규명) — 사전 고정 규칙 아님",
                "maturity_criterion_gap": {
                    "e3_definition": f"점추정 ŝ_k(n) ≥ {TAU}",
                    "e5_definition": f"Pr(s_k ≥ {TAU} | D_k) ≥ {1-DELTA}",
                    "e5_required_probe_successes_round1": f"{r1_need}/{P_PROBE}",
                    "e5_implied_success_rate_round1": round(r1_need / P_PROBE, 3),
                    "note": "동일한 τ=0.8을 쓰지만 E5는 사후확률 기준이라 표본 20에서 "
                            f"실질 요구 성공률이 {r1_need/P_PROBE:.2f}로 올라간다 — "
                            "E3 성숙(≥0.80)과 E5 성숙은 같은 이름의 다른 문턱이다."},
                "round2_carryover_trap": {
                    "c_reinit": C_REINIT,
                    "f1_threshold_for_unwinnable": f1_hopeless,
                    "meaning": f"라운드1 실패가 {f1_hopeless}회 이상이면 c={C_REINIT} 이월로 남은 φ 때문에 "
                               f"라운드2에서 {P_PROBE}/{P_PROBE} 만점을 받아도 Pr < {1-DELTA} — "
                               "라운드2 시작 시점에 X가 이미 확정돼 있다",
                    "n_unwinnable": len(unwinnable), "clusters": unwinnable,
                    "scored_full_marks_in_round2_but_failed":
                        [c for c in scored_max if c in unwinnable]},
                "per_cluster_probe_reconstruction": probe_recon},
            "n_star_distribution": {str(v): sum(1 for x in per.values() if x["e3_N_star"] == v)
                                    for v in sorted(({x["e3_N_star"] for x in per.values()
                                                      if reached(x["e3_N_star"])}))}},
    }
    op = os.path.join(rd, f"ineligible_postmortem_{args.seed_idx}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E5PM-DONE] {op}")
    print(json.dumps({k: out[k] for k in
                      ["n_ineligible", "share_of_tail_vla_calls",
                       "rule1_retry_would_have_helped", "rule2_batch_vs_stream_gap"]},
                     indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
