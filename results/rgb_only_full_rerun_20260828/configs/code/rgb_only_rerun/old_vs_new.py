"""§15 기존 RGB-D 결과와의 수치 대조 — **내부 검증용**. 해석 문장 없음.

각 행에 old/new의 **출처 파일까지** 남긴다 (분석자가 직접 대조할 수 있게).
산출: 08_statistics/OLD_VS_NEW_NUMERIC.csv (+ .json)
실행: hv2_hab python -u experiments/rgb_only_rerun/old_vs_new.py
마커: [OLDVSNEW-DONE]
"""
import csv
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import ROOT  # noqa: E402

SEEDS = (0, 1, 2)
OUT = f"{ROOT}/08_statistics"
CENSOR_CAP = 160
rows = []


def add(metric, key, old, new, src_old, src_new, note=""):
    def num(x):
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    o, n = num(old), num(new)
    rows.append({
        "metric": metric, "seed_or_cluster": key,
        "old_rgbd": old, "new_rgb": new,
        "absolute_change": round(n - o, 6) if (o is not None and n is not None) else None,
        "relative_change": (round((n - o) / o, 6)
                            if (o not in (None, 0) and n is not None) else None),
        "source_old": src_old, "source_new": src_new, "note": note})


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------- BATCH
    o_c = jload("results/e3/e3_curves.json")
    n_c = jload(f"{OUT}/rgb_only_e3_curves.json")
    if o_c and n_c:
        so, sn = "results/e3/e3_curves.json", f"{OUT}/rgb_only_e3_curves.json"
        val = lambda v: CENSOR_CAP if v == ">80" else v          # noqa: E731
        for cl in sorted(set(o_c["n_star"]) | set(n_c["n_star"])):
            add("batch.N_star", cl, o_c["n_star"].get(cl), n_c["n_star"].get(cl), so, sn,
                f">80은 우측절단 (수치 비교 시 cap={CENSOR_CAP})")
        for tag, pred in (("all_standard", lambda c: not c.startswith("chained_")),
                          ("formation_cells", lambda c: c.split("_task")[0] in
                           ("libero_object", "libero_goal", "libero_spatial"))):
            ov = [val(o_c["n_star"][c]) for c in o_c["n_star"] if pred(c)]
            nv = [val(n_c["n_star"][c]) for c in n_c["n_star"] if pred(c)]
            add("batch.median_N_star", tag, float(np.median(ov)), float(np.median(nv)), so, sn)
            add("batch.formable_count", tag,
                sum(1 for c in o_c["n_star"] if pred(c) and o_c["n_star"][c] != ">80"),
                sum(1 for c in n_c["n_star"] if pred(c) and n_c["n_star"][c] != ">80"), so, sn)
        for cl in sorted(set(o_c["clusters"]) & set(n_c["clusters"])):
            for n_ in ("20", "80"):
                oc = o_c["clusters"][cl].get("curve", {}).get(n_)
                nc = n_c["clusters"][cl].get("curve", {}).get(n_)
                if oc is not None and nc is not None:
                    add(f"batch.s_hat_n{n_}", cl, oc, nc, so, sn)

    # ---------------- ONLINE
    for s in SEEDS:
        o_r = jload(f"results/e5/reading_{s}.json")
        o_s = jload(f"results/e5/summary_{s}.json")
        n_o = jload(f"{seed_dir(s)}/ONLINE_SUMMARY_seed{s}.json")
        if not (o_r and n_o):
            continue
        so, sn = f"results/e5/reading_{s}.json", f"{seed_dir(s)}/ONLINE_SUMMARY_seed{s}.json"
        add("online.routing_first_1000", s, o_r["H4a_call_rate_reduction"]["p_first"],
            n_o["vla_routing_rate"]["first_1000"], so, sn)
        add("online.routing_last_1000", s, o_r["H4a_call_rate_reduction"]["p_last"],
            n_o["vla_routing_rate"]["last_1000"], so, sn)
        add("online.routing_reduction", s, o_r["H4a_call_rate_reduction"]["diff"],
            round(n_o["vla_routing_rate"]["reduction_pp"] / 100, 4), so, sn)
        add("online.routing_full_stream", s, o_r["overview"]["r_V_overall"],
            n_o["vla_routing_rate"]["full_stream"], so, sn)
        add("online.system_success", s, o_r["overview"]["system_success_rate"],
            n_o["system_success"]["full_stream"], so, sn)
        add("online.fire_success_rate", s, o_r["overview"]["fire_success_rate"],
            n_o["system_success"]["fire_success_rate"], so, sn)
        add("online.habit_firing_count", s, o_r["overview"]["n_fire"],
            n_o["risk"]["n_fire"], so, sn)
        add("online.firing_risk", s, o_r["risk_control"]["pr_fail_given_fire"],
            n_o["risk"]["pr_fail_given_fire"], so, sn)
        add("online.demotion_count", s, o_r["demotions"]["n_demotions"],
            n_o["lifecycle"]["demotion_count"], so, sn)
        add("online.rematuration_count", s,
            sum(1 for e in o_r["demotions"]["events"] if e.get("regained_M")),
            n_o["lifecycle"]["rematuration_count"], so, sn,
            "old = demotions.events 중 regained_M")
        add("online.n_retrain_events", s, o_r["formation_ledger"]["n_retrain"],
            n_o["lifecycle"]["n_retrain_events"], so, sn)
        if o_s:
            fs = o_s["final_states"]
            for st in ("M", "I", "X", "U"):
                add(f"online.final_{st}", s, sum(1 for v in fs.values() if v == st),
                    n_o["final_lifecycle"][st], f"results/e5/summary_{s}.json", sn)
        add("compute.operational_s", s, o_r["ledgers"]["operational_s"],
            n_o["ledger_seconds"].get("operational_s"), so, sn)
        add("compute.formation_s", s, o_r["ledgers"]["formation_s"],
            n_o["ledger_seconds"].get("formation_s"), so, sn)

        # ---------------- PAIRED
        n_p = jload(f"{ROOT}/05_paired_replay/PAIRED_REPLAY_SUMMARY.json")
        if n_p and s in [int(k) for k in n_p.get("per_seed", {})]:
            np_s = n_p["per_seed"][str(s)]["full_stream_noninferiority"]
            oh = o_r["H4b_noninferiority"]
            sp = f"{ROOT}/05_paired_replay/PAIRED_REPLAY_SUMMARY.json"
            add("paired.system_success", s, oh["system_rate"], np_s["system_rate"], so, sp,
                "전체 스트림 합성 (논문 H4b 구성)")
            add("paired.full_vla_success", s, oh["full_vla_rate"], np_s["full_vla_rate"], so, sp)
            add("paired.difference", s, oh["diff"], np_s["diff"], so, sp)
            add("paired.ci_lower", s, oh["ci95"][0], np_s["ci95"][0], so, sp)
            add("paired.ci_upper", s, oh["ci95"][1], np_s["ci95"][1], so, sp)
            add("paired.n_fired", s, o_r["overview"]["n_fire"],
                n_p["per_seed"][str(s)]["n_paired_episodes"], so, sp,
                "new = CF 재현이 끝난 발화 에피소드 수")

    # ---------------- COMPUTE (레이턴시)
    o_l = jload("results/e1/e1_latency.json")
    n_l = jload(f"{ROOT}/07_latency_cost/COMPUTE_SUMMARY.json")
    if o_l and n_l:
        so, sn = "results/e1/e1_latency.json", f"{ROOT}/07_latency_cost/COMPUTE_SUMMARY.json"
        add("compute.act_latency_ms", "median", o_l["anchor2_act_forward"]["median_ms"],
            n_l["act_forward_rgb_only"]["median_ms"], so, sn,
            "old = RGB-D ACT(4ch) / new = RGB-only ACT(3ch)")
        add("compute.act_latency_p95_ms", "p95", o_l["anchor2_act_forward"]["p95_ms"],
            n_l["act_forward_rgb_only"]["p95_ms"], so, sn)
        add("compute.teacher_latency_ms", "median", o_l["anchor1_oft_chunk_forward"]["median_ms"],
            n_l["teacher_oft_chunk_forward"]["median_ms"], so, sn, "teacher는 §1 동결 자산")
        add("compute.gate_latency_ms", "median", o_l["anchor3_gate_path"]["median_ms"],
            n_l["gate_path"]["median_ms"], so, sn)
        ft = n_l.get("formation_timing", {}).get("formation_event_total")
        if ft:
            add("compute.formation_event_s", "mean", None, ft["mean_s"], so, sn,
                "old 대응값 없음 — 기존 run은 이벤트 단위 학습/probe 분리 계측을 하지 않았다")

    # ---------------- 저장
    if rows:
        with open(f"{OUT}/OLD_VS_NEW_NUMERIC.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    json.dump({"run_id": os.path.basename(ROOT),
               "purpose": "내부 검증용 수치 대조. 해석 문장은 작성하지 않는다 (§15).",
               "n_rows": len(rows), "rows": rows},
              open(f"{OUT}/OLD_VS_NEW_NUMERIC.json", "w"), indent=1, ensure_ascii=False)
    fam = sorted({r["metric"].split(".")[0] for r in rows})
    print(f"[OLDVSNEW-DONE] rows={len(rows)} families={fam}")


if __name__ == "__main__":
    main()
