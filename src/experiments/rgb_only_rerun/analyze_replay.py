"""§11 PAIRED FULL-VLA REPLAY 집계 (RGB-only). 그림 없음 — 숫자·원자료·부트스트랩 분포만.

기존 RGB-D paired replay는 재사용하지 않는다. 본 run의 발화 집합에서 새로 추출된
cf_queue를 teacher가 동일 spec으로 재수행한 결과를 집계한다.
산출: 05_paired_replay/{PAIRED_REPLAY_EPISODES.csv, PAIRED_REPLAY_SUMMARY.json,
                       bootstrap_seed{S}.npy}
실행: hv2_hab python -u experiments/rgb_only_rerun/analyze_replay.py
마커: [PAIRED-DONE]
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
OUT = f"{ROOT}/05_paired_replay"
B_BOOT, BOOT_SEED = 10_000, 0        # §1 동결: bootstrap B = 10,000
MARGIN = -0.03                        # §1 동결: 사전 지정 −3 percentage-point margin


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def paired_bootstrap(diff, b=B_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(diff, float)
    idx = rng.integers(0, len(a), size=(b, len(a)))
    return a[idx].mean(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, per_seed = [], {}
    for s in SEEDS:
        cfp = f"{OUT}/cf_{s}.jsonl"
        qp = f"{seed_dir(s)}/cf_queue_{s}.jsonl"
        sp = f"{seed_dir(s)}/stream_{s}.jsonl"
        if not (os.path.exists(cfp) and os.path.exists(qp) and os.path.exists(sp)):
            print(f"[PAIRED-SKIP] seed {s} 입력 누락")
            continue
        queue = {d["uid"]: d for d in (json.loads(l) for l in open(qp))}
        stream = {}
        for l in open(sp):
            r = json.loads(l)
            if r["executor"] == "habit":
                stream[r["spec_uid"]] = r
        cf = [json.loads(l) for l in open(cfp)]

        srows, n_infra = [], 0
        for c in cf:
            if "teacher_success" not in c:
                n_infra += 1
                continue
            uid = c["uid"]
            q, st = queue.get(uid, {}), stream.get(uid, {})
            sysucc = int(st.get("outcome") == "success") if st else int(c["habit_success"])
            hab = int(bool(c["habit_success"]))
            vla = int(bool(c["teacher_success"]))
            srows.append({
                "seed": s, "episode": st.get("t"), "cluster_id": c["cluster"],
                "suite": q.get("suite"), "task_id": q.get("task_id"),
                "spec_uid": uid, "episode_seed": q.get("seed"),
                "initial_state_id": q.get("base_idx"), "perturbation_width": q.get("w"),
                "observation_noise_seed": q.get("noise_seed"),
                "cold_start": st.get("cold_start"),
                "system_success": sysucc, "habit_success": hab, "full_vla_success": vla,
                "difference": sysucc - vla,
                "habit_steps": st.get("steps"), "full_vla_steps": c.get("teacher_steps"),
                "state_before": st.get("state_before"), "tau_k": st.get("tau_k"),
            })
        rows += srows
        if not srows:
            continue

        diff = np.array([r["difference"] for r in srows], float)
        boot = paired_bootstrap(diff)
        np.save(f"{OUT}/bootstrap_seed{s}.npy", boot)
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        b_ = sum(1 for r in srows if r["system_success"] and not r["full_vla_success"])
        c_ = sum(1 for r in srows if not r["system_success"] and r["full_vla_success"])
        ent = {
            "seed": s, "n_paired_episodes": len(srows), "n_infra_error": n_infra,
            "system_success_rate": round(float(np.mean([r["system_success"] for r in srows])), 4),
            "full_vla_success_rate": round(float(np.mean([r["full_vla_success"] for r in srows])), 4),
            "habit_success_rate": round(float(np.mean([r["habit_success"] for r in srows])), 4),
            "paired_difference": round(float(diff.mean()), 4),
            "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                          "ci95": [round(lo, 4), round(hi, 4)],
                          "mean": round(float(boot.mean()), 4),
                          "sd": round(float(boot.std(ddof=1)), 4),
                          "file": f"bootstrap_seed{s}.npy"},
            "noninferiority": {
                "margin": MARGIN,
                "ci_lower": round(lo, 4),
                "passes_margin": bool(lo > MARGIN),
                "rule": "사전 지정 −3%p — CI 하한이 margin을 상회하면 비열등"},
            "discordant_system_only": b_, "discordant_vla_only": c_,
        }
        if b_ + c_ > 0:
            from scipy.stats import binomtest
            ent["mcnemar_exact_p"] = round(float(binomtest(b_, b_ + c_, 0.5).pvalue), 6)
        # --- 전체 스트림 합성 비열등 (논문 H4b와 동일 구성):
        #     발화분은 CF 재현 결과를, 비발화분은 VLA 실측을 기준선으로 쓴다 (§3).
        cfmap = {c["uid"]: bool(c["teacher_success"]) for c in cf if "teacher_success" in c}
        allrows = [json.loads(l) for l in open(sp)]
        sys_s, vla_s, miss = [], [], 0
        for r in allrows:
            if r["outcome"] == "infra_error":
                continue
            ok_ = r["outcome"] == "success"
            if r["executor"] == "habit":
                if r["spec_uid"] not in cfmap:
                    miss += 1
                    continue
                sys_s.append(ok_)
                vla_s.append(cfmap[r["spec_uid"]])
            else:
                sys_s.append(ok_)
                vla_s.append(ok_)
        fd = np.asarray(sys_s, float) - np.asarray(vla_s, float)
        fboot = paired_bootstrap(fd)
        np.save(f"{OUT}/bootstrap_fullstream_seed{s}.npy", fboot)
        flo, fhi = float(np.percentile(fboot, 2.5)), float(np.percentile(fboot, 97.5))
        ent["full_stream_noninferiority"] = {
            "construction": ("발화분 = CF 재현 teacher 결과 / 비발화분 = VLA 실측. "
                             "논문 H4b와 동일 구성."),
            "n_paired_episodes": len(sys_s), "n_cf_missing": miss,
            "cf_complete": miss == 0,
            "system_rate": round(float(np.mean(sys_s)), 4),
            "full_vla_rate": round(float(np.mean(vla_s)), 4),
            "diff": round(float(fd.mean()), 4),
            "ci95": [round(flo, 4), round(fhi, 4)],
            "margin": MARGIN, "noninferior": bool(flo > MARGIN),
            "bootstrap_file": f"bootstrap_fullstream_seed{s}.npy",
            "verdict": ("PASS" if flo > MARGIN else "FAIL") if miss == 0
                       else "PARTIAL — CF 미완, 판정 보류"}

        per_seed[s] = ent
        print(f"  seed {s}: n={len(srows)} sys={ent['system_success_rate']} "
              f"vla={ent['full_vla_success_rate']} Δ={ent['paired_difference']:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] margin_pass={ent['noninferiority']['passes_margin']} "
              f"| fullstream Δ={ent['full_stream_noninferiority']['diff']:+.4f} "
              f"CI{ent['full_stream_noninferiority']['ci95']} "
              f"{ent['full_stream_noninferiority']['verdict']}", flush=True)

    if rows:
        keys = list(rows[0].keys())
        with open(f"{OUT}/PAIRED_REPLAY_EPISODES.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # ---- 3 seed 통합 (pooled + seed 평균 둘 다 — 집계 방식 혼동 방지)
    allsum = {"run_id": os.path.basename(ROOT), "modality": "rgb_only",
              "seeds_completed": sorted(per_seed), "per_seed": per_seed,
              "margin": MARGIN, "bootstrap_B": B_BOOT}
    if per_seed:
        d = np.array([r["difference"] for r in rows], float)
        pooled = paired_bootstrap(d)
        np.save(f"{OUT}/bootstrap_pooled.npy", pooled)
        vals = [v["paired_difference"] for v in per_seed.values()]
        allsum["pooled"] = {
            "n_paired_episodes": len(rows),
            "system_success_rate": round(float(np.mean([r["system_success"] for r in rows])), 4),
            "full_vla_success_rate": round(float(np.mean([r["full_vla_success"] for r in rows])), 4),
            "paired_difference": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(pooled, 2.5)), 4),
                     round(float(np.percentile(pooled, 97.5)), 4)],
            "passes_margin": bool(float(np.percentile(pooled, 2.5)) > MARGIN),
            "bootstrap_file": "bootstrap_pooled.npy"}
        fs = [v["full_stream_noninferiority"] for v in per_seed.values()]
        allsum["full_stream_seed_mean"] = {
            "diff_mean": round(float(np.mean([x["diff"] for x in fs])), 4),
            "diff_sd": round(float(np.std([x["diff"] for x in fs], ddof=1)), 4)
            if len(fs) > 1 else 0.0,
            "system_rate_mean": round(float(np.mean([x["system_rate"] for x in fs])), 4),
            "full_vla_rate_mean": round(float(np.mean([x["full_vla_rate"] for x in fs])), 4),
            "all_noninferior": bool(all(x["noninferior"] for x in fs)),
            "verdicts": [x["verdict"] for x in fs]}
        allsum["seed_mean"] = {
            "paired_difference_mean": round(float(np.mean(vals)), 4),
            "paired_difference_sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals)}
    json.dump(allsum, open(f"{OUT}/PAIRED_REPLAY_SUMMARY.json", "w"), indent=1, ensure_ascii=False)
    print(f"[PAIRED-DONE] seeds={sorted(per_seed)} episodes={len(rows)}")


if __name__ == "__main__":
    main()
