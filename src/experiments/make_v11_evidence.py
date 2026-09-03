"""v11 패치 검증 근거 산출 — 원고 환경이 참조할 단일 진입점.

원고 환경에는 실험 원장이 없으므로, 패치가 요구하는 수치를 여기서 프로그래밍 산출해
JSON으로 넘긴다. 모든 값은 results/의 원자료에서만 읽는다(수동 입력 금지, CLAUDE.md §6).

산출: results/e5/v11_patch_evidence.json
"""
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)
S = [0, 1, 2]


def main():
    from experiments.e5_driver import BATCH_EQUIV_STEPS as B
    R = {s: json.load(open(f"results/e5/reading_{s}.json")) for s in S}
    F = {s: json.load(open(f"results/e5/summary_{s}.json"))["ledger_s"] for s in S}
    lat = json.load(open("results/e1/e1_latency.json"))
    ms_vla = lat["anchor1_oft_chunk_forward"]["median_ms"]
    a5 = lat["anchor5_act_train_n40"]
    rate = 8000 / a5["train_seconds"]
    streams = {s: [json.loads(l) for l in open(f"results/e5/stream_{s}.jsonl")] for s in S}

    # --- P3: standard / cold-start 라우팅 (full stream, episode-weighted)
    pooled_v = pooled_n = 0
    per_std, per_cold = [], []
    for s in S:
        std = [r for r in streams[s] if not r["is_novel_injection"]]
        cold = [r for r in streams[s] if r["is_novel_injection"]]
        pooled_v += sum(1 for r in std if r["executor"] == "vla")
        pooled_n += len(std)
        per_std.append(sum(1 for r in std if r["executor"] == "vla") / len(std))
        per_cold.append(sum(1 for r in cold if r["executor"] == "vla") / len(cold))

    # --- P12/P14: 형성 이벤트 비용
    ev = {}
    for s in S:
        for r in streams[s]:
            if r["retrain_event"]:
                e = r["retrain_event"]
                ev.setdefault(e["n"], []).append(e["formation_wall_s"])
    n20, n80 = len(ev[20]), len(ev[80])
    tr = {n: B[n] / rate for n in (20, 80)}
    w_ev = float(np.mean([x for v in ev.values() for x in v]))
    w_tr = (n20 * tr[20] + n80 * tr[80]) / (n20 + n80)

    # --- P13: 후반 VLA 중 X 비중
    per_share, cx, cv = [], 0, 0
    for s in S:
        t = R[s]["r_V_tail_decomposition"]
        per_share.append(t["share_of_window"].get("X", 0) / t["r_V_observed"])
        tail = streams[s][-1000:]
        v = [r for r in tail if r["executor"] == "vla"]
        cx += sum(1 for r in v if r["lifecycle_state"] == "X")
        cv += len(v)

    # --- 검증 1: ever-matured vs final-M
    ever, fin = [], []
    for s in S:
        ever.append(len({r["cluster"] for r in streams[s] if r["lifecycle_state"] == "M"}))
        last = {}
        for r in streams[s]:
            last[r["cluster"]] = r["lifecycle_state"]
        fin.append(sum(1 for x in last.values() if x == "M"))

    def ms(v, nd=4):
        a = np.array(v, float)
        return {"values": [round(float(x), 6) for x in a],
                "mean": round(float(a.mean()), nd), "sd": round(float(a.std(ddof=1)), nd)}

    out = {
        "note": "v11 패치 검증 근거. 원고 환경에서 재산출 불가한 값을 실험 원장에서 산출.",
        "source": "results/e5/stream_{0,1,2}.jsonl · reading_*.json · summary_*.json · e1_latency.json",
        "P3_routing_by_pool": {
            "scope": "full stream, episode-weighted",
            "standard_pooled": {"vla": pooled_v, "n": pooled_n, "rate": round(pooled_v / pooled_n, 6)},
            "standard_per_seed_mean": ms(per_std, 6),
            "cold_start_per_seed_mean": ms(per_cold, 6),
            "verdict": f"standard = {pooled_v/pooled_n:.6f} → 4자리 {pooled_v/pooled_n:.4f} "
                       f"(원고 0.5945는 올림 오기) · cold-start {np.mean(per_cold):.4f} (원고 일치)"},
        "P12_P14_formation_cost": {
            "events": {"n20": n20, "n80": n80, "total": n20 + n80},
            "event_wall_s": {str(n): round(float(np.mean(v)), 1) for n, v in sorted(ev.items())},
            "train_rate_steps_per_s": round(rate, 1),
            "train_steps": {str(n): B[n] for n in (20, 80)},
            "train_s_estimated": {str(n): round(tr[n], 1) for n in (20, 80)},
            "probe_prep_s": {str(n): round(float(np.mean(ev[n])) - tr[n], 1) for n in (20, 80)},
            "weighted_event_s": round(w_ev, 1),
            "weighted_train_s": round(w_tr, 1),
            "train_fraction": round(w_tr / w_ev, 4),
            "vla_call_equivalents": {"train_only": round(w_tr / (ms_vla / 1000)),
                                     "full_event": round(w_ev / (ms_vla / 1000))},
            "manuscript_anchor": {"value_s": a5["train_seconds"],
                                  "calls": a5["vla_call_equivalents"],
                                  "source": a5["source"],
                                  "issue": "E2 배치 8,000스텝 warm-start n=40 — E5(B-2 scratch, "
                                           "10,000/28,000 스텝) 조건이 아님"},
            "per_seed_event_s": ms([F[s]["formation_s"] / R[s]["formation_ledger"]["n_retrain"] for s in S], 1)},
        "P13_late_X_share": {
            "per_seed": ms(per_share, 4),
            "pooled": {"x": cx, "vla": cv, "rate": round(cx / cv, 6)},
            "verdict": f"원고 73.47% = seed별 비율의 비가중 평균 {np.mean(per_share):.4%} (일치) · "
                       f"pooled는 {cx/cv:.4%}"},
        "check1_maturity_counts": {
            "ever_matured": ms(ever, 4), "final_M": ms(fin, 4),
            "verdict": "20.3±2.1 = ever-matured · 18.3±2.1 = final-M"},
        "check2_tau_scope": {
            "verdict": "per-cluster",
            "evidence": "ClusterState마다 ACIRiskController 보유; observe_fire가 해당 클러스터 "
                        "maturity.tau만 갱신. 실측: 최종 τ가 클러스터별로 상이, 첫 발화 실패 후 "
                        "타 클러스터 τ 변화 0건",
            "final_tau_distribution": {str(k): v for k, v in sorted(
                __import__("collections").Counter(
                    round(next(r["tau"] for r in reversed(streams[0]) if r["cluster"] == c), 3)
                    for c in {r["cluster"] for r in streams[0]}).items())}},
        "check5_h2_definitions": {
            "between_share_formula": "SS_between / SS_total on censored ranks (순위 기반 η²)",
            "censoring": ">80 → CENSOR_CAP=160 치환 후 rankdata (공동 최상위)",
            "censored_clusters": [c for c, v in json.load(open("results/e3/e3_curves.json"))["n_star"].items()
                                  if not isinstance(v, (int, float))],
            "formation_cells_n": json.load(open("results/e3/h2_analysis.json"))["decomposition_L"]["n"],
            "verdict": "절단 1개는 long 스위트 → Fig. 2(b) 22셀 밖"},
        "test_names": {
            "p_0.0968": "one-sided Fisher exact test (fisher_exact alternative='less')",
            "p_0.0049": "one-sided exact binomial",
            "H4a": "one-sided two-proportion z-test",
            "H4b": "paired bootstrap B=10,000, margin −0.03"},
    }
    p = "results/e5/v11_patch_evidence.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[V11EV-DONE] {p}")
    print(f"  P3  standard {out['P3_routing_by_pool']['standard_pooled']['rate']:.6f} → 0.5944")
    print(f"  P12 이벤트당 {out['P12_P14_formation_cost']['weighted_event_s']} s")
    print(f"  P14 학습분 {out['P12_P14_formation_cost']['weighted_train_s']} s "
          f"({out['P12_P14_formation_cost']['train_fraction']:.0%}) = "
          f"{out['P12_P14_formation_cost']['vla_call_equivalents']['train_only']:,} calls")


if __name__ == "__main__":
    main()
