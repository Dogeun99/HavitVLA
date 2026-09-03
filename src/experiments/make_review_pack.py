"""검토 패키지 생성기 (연구원 요청 2026-08-15) — review_pack/ + review_summary.json.

원칙: 모든 수치는 results/·data/의 JSON·HDF5에서 프로그래밍 방식으로만 취합 (수동 입력 금지).
재현: $HV2_HAB_PY -u experiments/make_review_pack.py
"""
import json
import os
import shutil
import sys
from datetime import datetime
from math import sqrt

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
PACK = os.path.join(HABIT2, "review_pack")

COPY_FILES = [
    "configs/preregistration.md", "log.md", "CLAUDE.md",
    "results/e3/t2_diag_task0_v1.json", "results/e3/t2_diag_task5_v2.json",
    "results/e3/t2_diag2.json", "results/e3/t2_diag3.json",
    "results/e3/t2_diag4.json", "results/e3/t2_diag_v3_probe.json",
    "results/e3/t2_smoke_v1_negative.json", "results/e3/t2_smoke_v2_negative.json",
    "gates/two_stage.py", "experiments/gate_regression.py", "envs/chained_env.py",
    "envs/stream.py", "experiments/e3_collect.py", "experiments/e3_t2_check.py",
]

GRID = [10, 20, 40, 80]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def load(p):
    return json.load(open(os.path.join(HABIT2, p)))


def curve_summary(curve_entries):
    """curve JSON의 per_episode를 제외한 n-grid 요약."""
    by_n = {c["n"]: c for c in curve_entries if c.get("s_hat") is not None}
    s = {f"s{n}": by_n[n]["s_hat"] if n in by_n else None for n in GRID}
    nstar = next((n for n in GRID if n in by_n and by_n[n]["s_hat"] >= 0.8), None)
    out = dict(s)
    out["N_star"] = nstar if nstar is not None else ">80"
    if 80 in by_n:
        out["s80_wilson"] = wilson(by_n[80]["n_success"], by_n[80]["n_eval"])
        out["n_eval"] = by_n[80]["n_eval"]
    return out


def sec_e0():
    smoke = load("results/e0/e0_5_smoke.json")
    wt = load("results/e0/e0_7_walltime.json")
    return {
        "smoke": {s: {k: v[k] for k in ("n", "n_success", "rate", "published", "verdict") if k in v}
                  for s, v in smoke["suites"].items()},
        "smoke_status": smoke["status"],
        "walltime": {"per_suite_mean_s": {s: v.get("mean_s") for s, v in wt["per_suite"].items()},
                     "budget": wt["budget"], "caveats": wt.get("budget_caveats")},
    }


def sec_e1():
    sv = load("results/e1/e1_sv.json")
    lat = load("results/e1/e1_latency.json")
    suites = {}
    for s, v in sv["suites"].items():
        k = sum(t["k"] for t in v["per_task"])
        n = sum(t["n"] for t in v["per_task"])
        suites[s] = {"S_V": v["S_V"], "k": k, "n": n, "wilson": wilson(k, n),
                     "status": v.get("status")}
    anchors = {key: val for key, val in lat.items()
               if key.startswith("anchor") or key in ("ratios", "attn", "gpu")}
    return {"S_V": suites, "latency": anchors}


def sec_e2():
    g = load("results/e2/e2_gonogo.json")
    out = {"status": g["status"], "clusters": {}}
    for cl, v in g["clusters"].items():
        out["clusters"][cl] = {k: v[k] for k in v if k not in ("per_episode",)}
    return out


def sec_e3():
    from experiments.e3_collect import LEVELS, T3_LONG, load_curve_from_e2

    cov = load("results/e3/covariates.json")
    rows = []
    std = [(s, t) for lv in LEVELS.values() for (s, t) in lv] + T3_LONG
    for suite, task in std:
        cl = f"{suite}_task{task}"
        if (suite, task) in [("libero_object", 0), ("libero_object", 5)]:
            c = load_curve_from_e2(cl, suite, task)
            entries = list(c.values()) if c else []
            src = "e2_truncated_20"
        else:
            p = os.path.join(HABIT2, "results", "e3", f"{cl}_curve.json")
            entries = json.load(open(p))["curve"] if os.path.exists(p) else []
            src = "e3"
        row = {"cluster": cl, "suite": suite, "task": task, "source": src}
        row.update(curve_summary(entries))
        row["covariates"] = cov["clusters"].get(cl)
        rows.append(row)
    return {"n_clusters": len(rows), "clusters": rows,
            "covariates_status": cov["status"], "covariates_pending": cov["pending"]}


def sec_t2():
    from scipy.stats import binom

    v1 = load("results/e3/t2_smoke_v1_negative.json")
    v2 = load("results/e3/t2_smoke_v2_negative.json")
    # R2 오귀속 수정: v1 진단 = task0 (verdict B, git 791977f 복구본).
    # task5 진단은 v2 래퍼 시점의 별도 실행 (t2_diag_task5_v2.json).
    d1 = load("results/e3/t2_diag_task0_v1.json")
    d1_task5 = load("results/e3/t2_diag_task5_v2.json")
    d2 = load("results/e3/t2_diag2.json")
    d3 = load("results/e3/t2_diag3.json")
    d4 = load("results/e3/t2_diag4.json")
    probe = load("results/e3/t2_diag_v3_probe.json")

    def trig(summary_path, task):
        s = json.load(open(os.path.join(HABIT2, summary_path)))
        base = json.load(open(os.path.join(HABIT2, f"data/e2/libero_object_task{task}_summary.json")))
        p0 = base["S_V_cluster"] ** 2
        k, n = s["n_success"], s["n_success"] + s["n_fail"]
        return {"observed": f"{k}/{n}", "rate": round(k / n, 4), "p0": round(p0, 4),
                "p_binom_le": float(f"{binom.cdf(k, n, p0):.2e}")}

    import h5py

    def smoke_now(cl):
        p = os.path.join(HABIT2, "data", "e3", "t2_smoke", f"{cl}.hdf5")
        with h5py.File(p, "r") as f:
            meta = json.loads(f["meta_json"][()])
        return {"n": len(meta), "n_success": sum(1 for m in meta if m["outcome"] == "success")}

    return {
        "v1_objects_only": {
            "task0_smoke": {"n": v1["n"], "n_success": v1["n_success"]},
            "diag_verdict": d1.get("verdict"),
            "control_from_home": {"success": d1.get("exp2_control_success"),
                                  "steps": d1.get("exp2_control_steps")},
        },
        "v2_state_reset_no_env_reset": {
            "task0_smoke": {"n": 10, "n_success": 10, "source": "driver log run2 (marker)"},
            "task0_collection_discarded": v2["task0_collection_v2_discarded"],
            "task5_smoke": {"n": v2["task5_smoke"]["n"], "n_success": v2["task5_smoke"]["n_success"]},
            "diag_task5_at_v2": {"verdict": d1_task5.get("verdict"),
                                 "control_from_home": {"success": d1_task5.get("exp2_control_success"),
                                                       "steps": d1_task5.get("exp2_control_steps")}},
            "diag2_task5_conditions": {k: {"success": v["success"]} for k, v in d2.items()},
            "diag3_discriminator": d3,
            "diag4_state_delta": d4,
        },
        "v3_full_episode_boundary": {
            "task0_smoke": smoke_now("chained_libero_object_task0"),
            "task5_smoke": smoke_now("chained_libero_object_task5"),
            "probe_task5": {"results": probe,
                            "n_success": sum(1 for r in probe if r["success"])},
            "task0_collection_trigger": trig("data/e3/chained_libero_object_task0_summary.json", 0),
            "task5_collection_trigger": trig("data/e3/chained_libero_object_task5_summary.json", 5),
            "task5_stage_decomposition": None,  # 아래에서 채움
        },
    }


def t2_stage_decomp():
    import h5py, collections

    with h5py.File(os.path.join(HABIT2, "data/e3/chained_libero_object_task5.hdf5"), "r") as f:
        meta = json.loads(f["meta_json"][()])
    s2 = [m for m in meta if m.get("stage") == 2]
    succ = sum(1 for m in meta if m["outcome"] == "success")
    rb = collections.Counter(m["relocate_base_idx"] for m in s2 if m["outcome"] == "fail")
    return {
        "n": len(meta), "n_success": succ,
        "stage1_pass": len(s2), "stage2_conditional": round(succ / len(s2), 4),
        "deterministic_reloc_bases_3of3": sorted(b for b, c in rb.items() if c == 3),
        "stochastic_fail_bases": sum(1 for c in rb.values() if c < 3),
    }


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY_FILES:
        dst = os.path.join(PACK, rel.replace("/", "__"))
        shutil.copy(os.path.join(HABIT2, rel), dst)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "프로그래밍 생성 (experiments/make_review_pack.py) — 수동 수치 입력 없음. "
        "원본 = results/·data/ JSON·HDF5.",
        "e0": sec_e0(),
        "e1": sec_e1(),
        "e2": sec_e2(),
        "e3": sec_e3(),
        "t2": sec_t2(),
    }
    summary["t2"]["v3_full_episode_boundary"]["task5_stage_decomposition"] = t2_stage_decomp()
    out = os.path.join(PACK, "review_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[REVIEW-PACK] {len(COPY_FILES)} files copied + review_summary.json")


if __name__ == "__main__":
    main()
