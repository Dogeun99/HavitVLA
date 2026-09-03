"""§8·§9·§10 온라인 산출 (RGB-only). 그림 없음 — 원장 CSV와 숫자 JSON만.

지표 정의는 기존 e5_analyze.py와 동일하다:
  - 유효 에피소드 = outcome != infra_error
  - r_V = |executor==vla| / |유효|
  - Pr(fail|fire) = 발화 실패 / 발화
산출 (02~04_online_seedX/ · derived/):
  ONLINE_EPISODE_LEDGER_seedX.csv · LIFECYCLE_EVENTS_LONG.csv ·
  LIFECYCLE_CLUSTER_SUMMARY.csv · ONLINE_SUMMARY_seedX.json · ONLINE_SUMMARY_ALL_SEEDS.json
실행: hv2_hab python -u experiments/rgb_only_rerun/analyze_online.py
마커: [ONLINE-SUMMARY-DONE]
"""
import csv
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.e5_analyze import two_proportion_one_sided  # noqa: E402
from experiments.rgb_only_rerun.runner import ROOT  # noqa: E402

SEEDS = (0, 1, 2)
WIN = 200
EPS_RISK = 0.2
DERIVED = f"{ROOT}/derived"


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def load(s):
    p = f"{seed_dir(s)}/stream_{s}.jsonl"
    if not os.path.exists(p):
        return None, None, None
    rows = [json.loads(l) for l in open(p)]
    ep = f"{seed_dir(s)}/lifecycle_events_{s}.jsonl"
    events = [json.loads(l) for l in open(ep)] if os.path.exists(ep) else []
    sp = f"{seed_dir(s)}/summary_{s}.json"
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    return rows, events, summ


def flatten(r):
    """중첩 필드를 CSV 열로 편다. 원본 JSONL은 그대로 보존된다."""
    sj = r.get("shadow_jur") or {}
    aci = r.get("aci") or {}
    re_ = r.get("retrain_event") or {}
    out = {k: v for k, v in r.items()
           if k not in ("shadow_jur", "aci", "retrain_event", "ledger_update")}
    out.update({
        "shadow_jur_score": sj.get("score"), "shadow_jur_q": sj.get("q"),
        "shadow_jur_accept": sj.get("accept"),
        "aci_fired": aci.get("fired"), "aci_fired_fail": aci.get("fired_fail"),
        "aci_empirical_risk": aci.get("empirical_risk"),
        "retrain_n": re_.get("n"), "retrain_probe_round": re_.get("probe_round"),
        "retrain_passed": re_.get("passed"),
        "retrain_train_wall_s": re_.get("train_wall_s"),
        "retrain_formation_wall_s": re_.get("formation_wall_s"),
        "retrain_probe_habit_calls": re_.get("probe_habit_calls"),
        "retrain_bc_pool_at_trigger": re_.get("bc_pool_at_trigger"),
        "ledger_update_source": (r.get("ledger_update") or {}).get("source"),
        "ledger_update_success": (r.get("ledger_update") or {}).get("success"),
    })
    return out


def windows(rows, key):
    out = []
    for s in range(0, len(rows), WIN):
        w = [r for r in rows[s:s + WIN] if r["outcome"] != "infra_error"]
        if not w:
            continue
        out.append({"ep_start": s, "n_valid": len(w),
                    "r_V": round(sum(1 for r in w if r["executor"] == "vla") / len(w), 4),
                    "system_success": round(sum(1 for r in w if r["outcome"] == "success")
                                            / len(w), 4)})
    return out


def summarize(s, rows, events, summ):
    ok = [r for r in rows if r["outcome"] != "infra_error"]
    fire = [r for r in ok if r["executor"] == "habit"]
    vla = [r for r in ok if r["executor"] == "vla"]
    first, last = rows[:1000], rows[-1000:]
    fk = [r for r in first if r["outcome"] != "infra_error"]
    lk = [r for r in last if r["outcome"] != "infra_error"]
    clusters = sorted({r["cluster"] for r in rows})
    # 최종 상태는 **원장에서 도출**한다 — 드라이버 summary가 없거나(미완주) 어긋나도
    # 조용히 0이 되지 않게. summary는 아래에서 교차검증 용도로만 쓴다.
    final_states = {}
    for r in rows:
        final_states[r["cluster"]] = r["state_after"]
    driver_final = summ.get("final_states", {})
    cold = {r["cluster"] for r in rows if r.get("cold_start")}

    # --- lifecycle 집계 (이벤트 원장에서)
    ev_by = lambda t: [e for e in events if e["event_type"] == t]           # noqa: E731
    first_mat = ev_by("first_maturity")
    demotions = ev_by("demotion")
    remats = ev_by("rematuration")
    to_x = ev_by("transition_X")
    retrains = [r["retrain_event"] for r in rows if r.get("retrain_event")]

    # 첫 성숙까지의 노출 수 (해당 클러스터의 그 시점까지 등장 횟수)
    exposures_to_mat = []
    for e in first_mat:
        n_exp = sum(1 for r in rows[:e["episode"] + 1] if r["cluster"] == e["cluster_id"])
        exposures_to_mat.append(n_exp)

    def rate(sub, pred):
        return round(sum(1 for r in sub if pred(r)) / len(sub), 4) if sub else None

    out = {
        "seed": s, "modality": summ.get("modality", "rgb_only"),
        "n_episodes": len(rows), "n_valid": len(ok), "n_infra_error": len(rows) - len(ok),
        "vla_routing_rate": {
            "full_stream": round(len(vla) / len(ok), 4),
            "first_1000": rate(fk, lambda r: r["executor"] == "vla"),
            "last_1000": rate(lk, lambda r: r["executor"] == "vla"),
            "reduction_pp": round(100 * (rate(fk, lambda r: r["executor"] == "vla")
                                         - rate(lk, lambda r: r["executor"] == "vla")), 2),
            "one_sided_test_first_vs_last": two_proportion_one_sided(
                sum(1 for r in first if r["executor"] == "vla"), len(first),
                sum(1 for r in last if r["executor"] == "vla"), len(last)),
        },
        "system_success": {
            "full_stream": round(sum(1 for r in ok if r["outcome"] == "success") / len(ok), 4),
            "first_1000": rate(fk, lambda r: r["outcome"] == "success"),
            "last_1000": rate(lk, lambda r: r["outcome"] == "success"),
            "fire_success_rate": rate(fire, lambda r: r["outcome"] == "success"),
            "vla_success_rate": rate(vla, lambda r: r["outcome"] == "success"),
        },
        "windows_200": windows(rows, "all"),
        "final_lifecycle": {st: sum(1 for c in clusters if final_states.get(c) == st)
                            for st in ("M", "I", "X", "U")},
        "final_states_by_cluster": {c: final_states.get(c) for c in clusters},
        "final_states_source": "episode ledger (마지막 state_after)",
        "final_states_driver_crosscheck": {
            "driver_summary_present": bool(driver_final),
            "mismatch": {c: [final_states.get(c), driver_final.get(c)] for c in clusters
                         if driver_final and final_states.get(c) != driver_final.get(c)}},
        "lifecycle": {
            "n_clusters_observed": len(clusters),
            "ever_mature_count": len({e["cluster_id"] for e in first_mat}),
            "ever_mature_clusters": sorted({e["cluster_id"] for e in first_mat}),
            "demotion_count": len(demotions),
            "demotion_clusters": sorted({e["cluster_id"] for e in demotions}),
            "rematuration_count": len(remats),
            "rematuration_clusters": sorted({e["cluster_id"] for e in remats}),
            "transition_X_count": len(to_x),
            "transition_X_clusters": sorted({e["cluster_id"] for e in to_x}),
            "n_retrain_events": len(retrains),
            "retrain_by_grid_n": {str(n): {
                "attempts": sum(1 for e in retrains if e["n"] == n),
                "passed": sum(1 for e in retrains if e["n"] == n and e["passed"])}
                for n in sorted({e["n"] for e in retrains})} if retrains else {},
            "probe_pass_by_round": {str(r_): {
                "attempts": sum(1 for e in retrains if e["probe_round"] == r_),
                "passed": sum(1 for e in retrains if e["probe_round"] == r_ and e["passed"]),
                "pass_rate": round(sum(1 for e in retrains if e["probe_round"] == r_
                                       and e["passed"])
                                   / max(sum(1 for e in retrains if e["probe_round"] == r_), 1), 4)}
                for r_ in sorted({e["probe_round"] for e in retrains})} if retrains else {},
            "first_maturity_exposures": {
                "values": exposures_to_mat,
                "median": float(np.median(exposures_to_mat)) if exposures_to_mat else None,
                "mean": round(float(np.mean(exposures_to_mat)), 2) if exposures_to_mat else None,
                "min": min(exposures_to_mat) if exposures_to_mat else None,
                "max": max(exposures_to_mat) if exposures_to_mat else None},
            "first_maturity_episode": {e["cluster_id"]: e["episode"] for e in first_mat},
        },
        "risk": {
            "n_fire": len(fire),
            "n_fire_fail": sum(1 for r in fire if r["outcome"] == "fail"),
            "pr_fail_given_fire": round(sum(1 for r in fire if r["outcome"] == "fail")
                                        / max(len(fire), 1), 4),
            "epsilon": EPS_RISK,
            "within_bound": bool(sum(1 for r in fire if r["outcome"] == "fail")
                                 / max(len(fire), 1) <= EPS_RISK),
            "tau_final_max": max(r["tau"] for r in rows),
            "tau_raised_episodes": sum(1 for r in rows if r["tau"] > 0.8),
            "clusters_with_tau_raised": sorted({r["cluster"] for r in rows if r["tau"] > 0.8}),
        },
        "cold_start": {
            "clusters": sorted(cold),
            "n_episodes": sum(1 for r in ok if r.get("cold_start")),
            "routing_r_V": rate([r for r in ok if r.get("cold_start")],
                                lambda r: r["executor"] == "vla"),
            "system_success": rate([r for r in ok if r.get("cold_start")],
                                   lambda r: r["outcome"] == "success"),
            "n_matured": len({e["cluster_id"] for e in first_mat if e["cluster_id"] in cold}),
            "final_state_distribution": {
                st: sum(1 for c in cold if final_states.get(c) == st)
                for st in ("M", "I", "X", "U")},
            "standard_routing_r_V": rate([r for r in ok if not r.get("cold_start")],
                                         lambda r: r["executor"] == "vla"),
            "standard_system_success": rate([r for r in ok if not r.get("cold_start")],
                                            lambda r: r["outcome"] == "success"),
        },
    }

    # --- 후반 트래픽 기여 분해 (last 1000, 상태별)
    tail_vla = [r for r in lk if r["executor"] == "vla"]
    by_state = {}
    for r in tail_vla:
        by_state[r["state_before"]] = by_state.get(r["state_before"], 0) + 1
    out["late_traffic_last1000"] = {
        "n_valid": len(lk), "n_vla": len(tail_vla),
        "r_V": round(len(tail_vla) / len(lk), 4),
        "vla_by_state_before": by_state,
        "contribution_of_window": {st: round(n / len(lk), 4) for st, n in by_state.items()},
        "share_of_vla": {st: round(n / max(len(tail_vla), 1), 4) for st, n in by_state.items()},
    }

    # --- 호출 회계 (§8 VLA_calls / habit_calls)
    out["call_accounting"] = {
        "total_VLA_calls": int(sum(r.get("VLA_calls") or 0 for r in rows)),
        "total_habit_calls_stream": int(sum(r.get("habit_calls") or 0 for r in rows)),
        "total_probe_habit_calls": int(sum((r.get("retrain_event") or {}).get(
            "probe_habit_calls", 0) or 0 for r in rows)),
        "mean_VLA_calls_per_teacher_episode": round(float(np.mean(
            [r["VLA_calls"] for r in rows if r.get("teacher_used")])), 2)
        if any(r.get("teacher_used") for r in rows) else None,
        "mean_habit_calls_per_fire_episode": round(float(np.mean(
            [r["habit_calls"] for r in rows if r.get("habit_fired")])), 2)
        if any(r.get("habit_fired") for r in rows) else None,
    }
    out["ledger_seconds"] = summ.get("ledger_s", {})
    out["total_wall_s"] = summ.get("total_wall_s")
    return out


def cluster_summary_rows(s, rows, events, summ):
    final_states = summ.get("final_states", {})
    out = []
    for cl in sorted({r["cluster"] for r in rows}):
        cr = [r for r in rows if r["cluster"] == cl]
        ce = [e for e in events if e["cluster_id"] == cl]
        first = lambda t: next((e["episode"] for e in ce if e["event_type"] == t), None)  # noqa: E731
        trains = [e["episode"] for e in ce if e["event_type"] in ("first_training", "retraining")]
        fires = [r for r in cr if r["executor"] == "habit"]
        out.append({
            "seed": s, "cluster_id": cl, "suite": cr[0]["suite"], "task_id": cr[0]["task_id"],
            "cold_start": int(bool(cr[0].get("cold_start"))),
            "first_exposure_episode": cr[0]["t"],
            "first_training_episode": trains[0] if trains else None,
            "second_training_episode": trains[1] if len(trains) > 1 else None,
            "first_maturity_episode": first("first_maturity"),
            "first_X_episode": first("transition_X"),
            "num_exposures": len(cr),
            "num_firings": len(fires),
            "num_failures": sum(1 for r in fires if r["outcome"] == "fail"),
            "num_demotions": sum(1 for e in ce if e["event_type"] == "demotion"),
            "num_rematurations": sum(1 for e in ce if e["event_type"] == "rematuration"),
            "num_retrainings": len(trains),
            "final_state": final_states.get(cl),
            "final_B_k_size": cr[-1]["bc_pool"],
            "final_tau": cr[-1]["tau"],
            "final_sigma_k": cr[-1]["sigma_k"], "final_phi_k": cr[-1]["phi_k"],
        })
    return out


def dump_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def main():
    os.makedirs(DERIVED, exist_ok=True)
    all_events, all_cluster, summaries = [], [], {}
    for s in SEEDS:
        rows, events, summ = load(s)
        if rows is None:
            print(f"[ONLINE-SUMMARY-SKIP] seed {s} 스트림 없음")
            continue
        dump_csv(f"{seed_dir(s)}/ONLINE_EPISODE_LEDGER_seed{s}.csv", [flatten(r) for r in rows])
        all_events += events
        all_cluster += cluster_summary_rows(s, rows, events, summ)
        o = summarize(s, rows, events, summ)
        summaries[s] = o
        json.dump(o, open(f"{seed_dir(s)}/ONLINE_SUMMARY_seed{s}.json", "w"),
                  indent=1, ensure_ascii=False)
        print(f"  seed {s}: n={o['n_episodes']} r_V={o['vla_routing_rate']['full_stream']} "
              f"(first1000={o['vla_routing_rate']['first_1000']} → "
              f"last1000={o['vla_routing_rate']['last_1000']}) "
              f"success={o['system_success']['full_stream']} "
              f"risk={o['risk']['pr_fail_given_fire']} "
              f"M/I/X/U={list(o['final_lifecycle'].values())}", flush=True)

    dump_csv(f"{DERIVED}/LIFECYCLE_EVENTS_LONG.csv", all_events)
    dump_csv(f"{DERIVED}/LIFECYCLE_CLUSTER_SUMMARY.csv", all_cluster)

    # ---- 3 seed mean/sd
    def ms(getter):
        v = [getter(o) for o in summaries.values() if getter(o) is not None]
        if not v:
            return None
        return {"values": v, "mean": round(float(np.mean(v)), 4),
                "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0, "n_seeds": len(v)}

    allseeds = {
        "run_id": os.path.basename(ROOT), "modality": "rgb_only",
        "seeds_completed": sorted(summaries),
        "note": "mean/sd는 완료된 seed에 대해서만 계산한다 (ddof=1).",
        "vla_routing_full": ms(lambda o: o["vla_routing_rate"]["full_stream"]),
        "vla_routing_first_1000": ms(lambda o: o["vla_routing_rate"]["first_1000"]),
        "vla_routing_last_1000": ms(lambda o: o["vla_routing_rate"]["last_1000"]),
        "vla_routing_reduction_pp": ms(lambda o: o["vla_routing_rate"]["reduction_pp"]),
        "system_success_full": ms(lambda o: o["system_success"]["full_stream"]),
        "fire_success_rate": ms(lambda o: o["system_success"]["fire_success_rate"]),
        "vla_success_rate": ms(lambda o: o["system_success"]["vla_success_rate"]),
        "pr_fail_given_fire": ms(lambda o: o["risk"]["pr_fail_given_fire"]),
        "n_fire": ms(lambda o: o["risk"]["n_fire"]),
        "ever_mature_count": ms(lambda o: o["lifecycle"]["ever_mature_count"]),
        "demotion_count": ms(lambda o: o["lifecycle"]["demotion_count"]),
        "rematuration_count": ms(lambda o: o["lifecycle"]["rematuration_count"]),
        "transition_X_count": ms(lambda o: o["lifecycle"]["transition_X_count"]),
        "n_retrain_events": ms(lambda o: o["lifecycle"]["n_retrain_events"]),
        "final_M": ms(lambda o: o["final_lifecycle"]["M"]),
        "final_I": ms(lambda o: o["final_lifecycle"]["I"]),
        "final_X": ms(lambda o: o["final_lifecycle"]["X"]),
        "final_U": ms(lambda o: o["final_lifecycle"]["U"]),
        "first_maturity_exposures_median": ms(
            lambda o: o["lifecycle"]["first_maturity_exposures"]["median"]),
        "cold_start_routing": ms(lambda o: o["cold_start"]["routing_r_V"]),
        "cold_start_success": ms(lambda o: o["cold_start"]["system_success"]),
        "cold_start_matured": ms(lambda o: o["cold_start"]["n_matured"]),
        "standard_routing": ms(lambda o: o["cold_start"]["standard_routing_r_V"]),
        "operational_s": ms(lambda o: o["ledger_seconds"].get("operational_s")),
        "formation_s": ms(lambda o: o["ledger_seconds"].get("formation_s")),
        "total_VLA_calls": ms(lambda o: o["call_accounting"]["total_VLA_calls"]),
        "per_seed": summaries,
    }
    json.dump(allseeds, open(f"{DERIVED}/ONLINE_SUMMARY_ALL_SEEDS.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"[ONLINE-SUMMARY-DONE] seeds={sorted(summaries)} "
          f"events={len(all_events)} cluster_rows={len(all_cluster)}")


if __name__ == "__main__":
    main()
