"""§22 최종 검증 — **패키지만 읽어** 필요한 숫자를 복원할 수 있는지 확인한다.

원칙: 레포 코드를 import하지 않는다 (stdlib + numpy만). 요약 JSON을 신뢰하지 않고
원장 CSV에서 **다시 계산해** 요약값과 대조한다. 불일치가 있으면 FAIL.
그림은 그리지 않는다 — 숫자 재구성 가능성만 본다.
실행: hv2_hab python -u experiments/rgb_only_rerun/verify_package.py --package <dir>
마커: [PACKAGE-VERIFY-PASS|FAIL]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

TOL = 1e-6
checks = []


def chk(name, ok, detail):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return ok


def close(a, b, tol=TOL):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    a = ap.parse_args()
    P = a.package

    # ---------- 1. 배치 곡선 + N* 를 에피소드 원장에서 재구성
    ep_p = f"{P}/01_batch_formation/batch_episode_results.csv"
    ns_p = f"{P}/01_batch_formation/NSTAR_RESULTS.csv"
    if os.path.exists(ep_p) and os.path.exists(ns_p):
        eps = read_csv(ep_p)
        agg = {}
        for r in eps:
            if r["outcome"] == "infra_error" or r["in_e3_view"] != "1":
                continue
            k = (r["cluster_id"], int(r["n"]))
            agg.setdefault(k, [0, 0])
            agg[k][0] += int(r["success"])
            agg[k][1] += 1
        curves = {}
        for (cl, n), (s, t) in agg.items():
            curves.setdefault(cl, {})[n] = s / t
        recon_ns = {}
        for cl, c in curves.items():
            recon_ns[cl] = next((n for n in (10, 20, 40, 80)
                                 if n in c and c[n] >= 0.8), ">80")
        table = {r["cluster_id"]: r for r in read_csv(ns_p)}
        bad = {cl: (recon_ns[cl], table[cl]["N_star"]) for cl in recon_ns
               if cl in table and str(recon_ns[cl]) != table[cl]["N_star"]}
        chk("batch.N_star_reconstructed", not bad,
            {"n_clusters": len(recon_ns), "mismatch": bad})
        badc = {}
        for cl, c in curves.items():
            if cl not in table:
                continue
            for n in (10, 20, 40, 80):
                v = table[cl].get(f"s_hat_{n}")
                if n in c and v not in (None, "") and not close(c[n], float(v), 1e-4):
                    badc[f"{cl}|n{n}"] = (round(c[n], 4), v)
        chk("batch.success_curve_reconstructed", not badc,
            {"n_points": sum(len(c) for c in curves.values()), "mismatch": badc})
    else:
        chk("batch.files_present", False, {"missing": [ep_p, ns_p]})

    # ---------- 2. 온라인: 원장 CSV에서 §10 지표 재계산 → 요약과 대조
    for s in (0, 1, 2):
        led = f"{P}/0{s + 2}_online_seed{s}/ONLINE_EPISODE_LEDGER_seed{s}.csv"
        summ = f"{P}/0{s + 2}_online_seed{s}/ONLINE_SUMMARY_seed{s}.json"
        if not (os.path.exists(led) and os.path.exists(summ)):
            continue
        rows = read_csv(led)
        o = json.load(open(summ))
        ok = [r for r in rows if r["outcome"] != "infra_error"]
        rv = sum(1 for r in ok if r["executor"] == "vla") / len(ok)
        chk(f"online.routing_full_seed{s}",
            close(rv, o["vla_routing_rate"]["full_stream"], 1e-4),
            {"recomputed": round(rv, 4), "summary": o["vla_routing_rate"]["full_stream"]})
        for tag, sub in (("first_1000", rows[:1000]), ("last_1000", rows[-1000:])):
            k = [r for r in sub if r["outcome"] != "infra_error"]
            v = sum(1 for r in k if r["executor"] == "vla") / len(k)
            chk(f"online.routing_{tag}_seed{s}",
                close(v, o["vla_routing_rate"][tag], 1e-4),
                {"recomputed": round(v, 4), "summary": o["vla_routing_rate"][tag]})
        ss = sum(1 for r in ok if r["outcome"] == "success") / len(ok)
        chk(f"online.system_success_seed{s}",
            close(ss, o["system_success"]["full_stream"], 1e-4),
            {"recomputed": round(ss, 4), "summary": o["system_success"]["full_stream"]})
        fire = [r for r in ok if r["executor"] == "habit"]
        risk = sum(1 for r in fire if r["outcome"] == "fail") / max(len(fire), 1)
        chk(f"online.firing_risk_seed{s}",
            close(risk, o["risk"]["pr_fail_given_fire"], 1e-4),
            {"recomputed": round(risk, 4), "summary": o["risk"]["pr_fail_given_fire"],
             "n_fire": len(fire)})
        # 200-ep 창
        wins = []
        for st in range(0, len(rows), 200):
            w = [r for r in rows[st:st + 200] if r["outcome"] != "infra_error"]
            if w:
                wins.append(round(sum(1 for r in w if r["executor"] == "vla") / len(w), 4))
        sw = [x["r_V"] for x in o["windows_200"]]
        chk(f"online.windows200_seed{s}", wins == sw,
            {"n_windows": len(wins), "first3_recomputed": wins[:3], "first3_summary": sw[:3]})
        # 최종 M/I/X/U
        last = {}
        for r in rows:
            last[r["cluster"]] = r["state_after"]
        fin = {st: sum(1 for v in last.values() if v == st) for st in ("M", "I", "X", "U")}
        chk(f"online.final_MIXU_seed{s}", fin == o["final_lifecycle"],
            {"recomputed": fin, "summary": o["final_lifecycle"]})
        # 강등 / 재성숙
        dem = sum(1 for r in rows if r["demotion"] in ("True", "true", "1"))
        rem = sum(1 for r in rows if r["rematuration"] in ("True", "true", "1"))
        chk(f"online.demotion_count_seed{s}", dem == o["lifecycle"]["demotion_count"],
            {"recomputed": dem, "summary": o["lifecycle"]["demotion_count"]})
        chk(f"online.rematuration_count_seed{s}", rem == o["lifecycle"]["rematuration_count"],
            {"recomputed": rem, "summary": o["lifecycle"]["rematuration_count"]})

    # ---------- 3. 전이 이벤트 원장 ↔ 에피소드 원장
    evp = f"{P}/derived/LIFECYCLE_EVENTS_LONG.csv"
    if os.path.exists(evp):
        ev = read_csv(evp)
        for s in (0, 1, 2):
            led = f"{P}/0{s + 2}_online_seed{s}/ONLINE_EPISODE_LEDGER_seed{s}.csv"
            if not os.path.exists(led):
                continue
            rows = read_csv(led)
            for fld, et in (("demotion", "demotion"), ("rematuration", "rematuration"),
                            ("transition_to_X", "transition_X")):
                a = sum(1 for r in rows if r[fld] in ("True", "true", "1"))
                b = sum(1 for e in ev if e["event_type"] == et and int(e["seed"]) == s)
                chk(f"events_vs_ledger_{et}_seed{s}", a == b,
                    {"ledger": a, "events": b})

    # ---------- 4. paired replay: 재계산 + 부트스트랩 CI 복원
    pep = f"{P}/05_paired_replay/PAIRED_REPLAY_EPISODES.csv"
    psp = f"{P}/05_paired_replay/PAIRED_REPLAY_SUMMARY.json"
    if os.path.exists(pep) and os.path.exists(psp):
        pr = read_csv(pep)
        ps = json.load(open(psp))
        for s in (0, 1, 2):
            k = str(s)
            if k not in ps.get("per_seed", {}):
                continue
            sub = [r for r in pr if int(r["seed"]) == s]
            e = ps["per_seed"][k]
            chk(f"paired.fire_count_seed{s}", len(sub) == e["n_paired_episodes"],
                {"recomputed": len(sub), "summary": e["n_paired_episodes"]})
            d = np.array([int(r["system_success"]) - int(r["full_vla_success"]) for r in sub],
                         float)
            chk(f"paired.difference_seed{s}", close(d.mean(), e["paired_difference"], 1e-4),
                {"recomputed": round(float(d.mean()), 4), "summary": e["paired_difference"]})
            bp = f"{P}/05_paired_replay/bootstrap_seed{s}.npy"
            if os.path.exists(bp):
                b = np.load(bp)
                lo, hi = np.percentile(b, [2.5, 97.5])
                chk(f"paired.bootstrap_ci_seed{s}",
                    close(lo, e["bootstrap"]["ci95"][0], 1e-4)
                    and close(hi, e["bootstrap"]["ci95"][1], 1e-4),
                    {"recomputed": [round(float(lo), 4), round(float(hi), 4)],
                     "summary": e["bootstrap"]["ci95"], "n_draws": int(b.size)})

    # ---------- 5. compute timings
    cs = f"{P}/07_latency_cost/COMPUTE_SUMMARY.json"
    lr = f"{P}/07_latency_cost/LATENCY_RAW.csv"
    if os.path.exists(cs) and os.path.exists(lr):
        c = json.load(open(cs))
        raw = read_csv(lr)
        for anchor in ("act_forward_rgb_only", "act_forward_rgbd", "gate_path",
                       "teacher_oft_chunk_forward"):
            v = [float(r["ms"]) for r in raw if r["anchor"] == anchor]
            if not v or anchor not in c:
                continue
            chk(f"compute.median_{anchor}", close(np.median(v), c[anchor]["median_ms"], 1e-2),
                {"recomputed": round(float(np.median(v)), 3), "summary": c[anchor]["median_ms"],
                 "n_samples": len(v)})

    # ---------- 6. 필수 파일 존재
    need = ["README_RESULTS.md", "DATA_DICTIONARY.md", "CHECKPOINT_MANIFEST.csv",
            "00_preflight/CONFIG_DIFF.json", "00_preflight/RGB_ONLY_INPUT_AUDIT.json",
            "00_preflight/ENVIRONMENT.json", "FAILED_JOBS.json"]
    miss = [f for f in need if not os.path.exists(f"{P}/{f}")]
    chk("package.required_files", not miss, {"missing": miss})

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    rep = {"package": os.path.basename(P), "n_checks": len(checks), "n_fail": n_fail,
           "verdict": "PASS" if n_fail == 0 else "FAIL",
           "principle": "패키지만 읽어 원장에서 재계산 → 요약과 대조. 레포 코드 import 없음.",
           "checks": checks}
    json.dump(rep, open(f"{P}/PACKAGE_VERIFICATION.json", "w"), indent=1, ensure_ascii=False)
    for c in checks:
        if c["status"] == "FAIL":
            print(f"  [FAIL] {c['check']}: {json.dumps(c['detail'], ensure_ascii=False)[:220]}")
    print(f"[PACKAGE-VERIFY-{rep['verdict']}] checks={len(checks)} fail={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
