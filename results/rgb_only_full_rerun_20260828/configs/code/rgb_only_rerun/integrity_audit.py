"""§14 데이터 무결성 검사 — 독립 validation. FAIL이 있으면 해당 결과를 VALID로 표시하지 않는다.

검사는 산출물을 다시 읽어 **처음부터 재구성**한다 (드라이버의 카운터를 믿지 않는다).
산출: 09_integrity/DATA_INTEGRITY_AUDIT.json
실행: hv2_hab python -u experiments/rgb_only_rerun/integrity_audit.py
마커: [INTEGRITY-DONE]
"""
import glob
import json
import os
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import ROOT  # noqa: E402

SEEDS = (0, 1, 2)
OUT = f"{ROOT}/09_integrity"
CK_BATCH = "checkpoints/rgb_only_rerun/batch"
CK_ONLINE = "checkpoints/rgb_only_rerun/online"


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def chk(name, ok, detail, expect="0"):
    return {"check": name, "status": "PASS" if ok else "FAIL", "expected": expect,
            "detail": detail}


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = []

    # ---------- 1. seed 대역 disjoint (수집/held-out/novel/probe/chained/e5_stream)
    from envs.stream import assert_six_bands_disjoint
    for s in SEEDS:
        try:
            b = assert_six_bands_disjoint(s)
            checks.append(chk(f"seed_band_overlap_seed{s}", True,
                              {"band_sizes": b, "overlaps": 0}))
        except AssertionError as e:
            checks.append(chk(f"seed_band_overlap_seed{s}", False, str(e)))

    # ---------- 2. train/eval overlap (배치): 학습 HDF5 uid ∩ held-out 평가 uid
    from experiments.e3_collect import EXPECTED_CLUSTERS
    from experiments.rgb_only_rerun.run_batch import E2_CLUSTERS
    import h5py
    tr_ev_overlap, checked = 0, 0
    for cl in EXPECTED_CLUSTERS:
        cp = f"{ROOT}/01_batch_formation/curves/{cl}_curve.json"
        ddir = "e2" if cl in E2_CLUSTERS else "e3"
        hp = f"data/{ddir}/{cl}.hdf5"
        if not (os.path.exists(cp) and os.path.exists(hp)):
            continue
        with h5py.File(hp, "r") as f:
            train_uids = {m["uid"] for m in json.loads(f["meta_json"][()])}
        eval_uids = set()
        for c in json.load(open(cp))["curve"]:
            eval_uids |= {e["uid"] for e in c.get("per_episode", [])}
        tr_ev_overlap += len(train_uids & eval_uids)
        checked += 1
    checks.append(chk("train_eval_overlap", tr_ev_overlap == 0,
                      {"n_clusters_checked": checked, "overlapping_uids": tr_ev_overlap}))

    # ---------- 3~11. 온라인 원장 기반 검사
    from envs.stream import probe_specs
    for s in SEEDS:
        sp = f"{seed_dir(s)}/stream_{s}.jsonl"
        if not os.path.exists(sp):
            continue
        rows = [json.loads(l) for l in open(sp)]
        ep = f"{seed_dir(s)}/lifecycle_events_{s}.jsonl"
        events = [json.loads(l) for l in open(ep)] if os.path.exists(ep) else []
        uids = [r["spec_uid"] for r in rows]

        # 3. 중복 / 누락 에피소드 ID
        checks.append(chk(f"duplicate_episode_id_seed{s}", len(uids) == len(set(uids)),
                          {"n_rows": len(uids), "n_unique": len(set(uids))}))
        idx = [r["t"] for r in rows]
        missing = sorted(set(range(len(rows))) - set(idx))
        checks.append(chk(f"missing_episode_id_seed{s}", not missing,
                          {"n_rows": len(rows), "n_missing_indices": len(missing),
                           "first_missing": missing[:5]}))

        # 4. probe / stream 대역 겹침
        probe_uids = set()
        for cl in {(r["suite"], r["task_id"]) for r in rows}:
            for rd in (0, 1):
                probe_uids |= {x.uid for x in probe_specs(cl[0], cl[1], rd)}
        checks.append(chk(f"probe_stream_overlap_seed{s}", not (probe_uids & set(uids)),
                          {"n_probe_specs": len(probe_uids),
                           "overlap": len(probe_uids & set(uids))}))

        # 5. probe가 스트림 지표에 유입되지 않았는가 (probe는 off-stream)
        bad_reason = sum(1 for r in rows if r["decision_reason"] not in
                         ("fire", "unknown_cluster", "habit_ineligible", "immature", "infra"))
        checks.append(chk(f"probe_in_stream_metric_seed{s}", bad_reason == 0,
                          {"unexpected_decision_reasons": bad_reason,
                           "reasons": sorted({r["decision_reason"] for r in rows})}))

        # 6. 비성숙 상태에서 발화했는가
        bad_fire = [r["t"] for r in rows if r["executor"] == "habit" and r["state_before"] != "M"]
        checks.append(chk(f"habit_fire_while_non_mature_seed{s}", not bad_fire,
                          {"n_bad": len(bad_fire), "episodes": bad_fire[:10]}))

        # 7. teacher 결과가 A_mat 원장(sigma/phi)에 산입됐는가
        #    재학습이 없는 teacher 에피소드에서는 sigma·phi가 변하면 안 된다.
        bad_ledger, prev = [], {}
        for r in rows:
            cl = r["cluster"]
            cur = (r["sigma_k"], r["phi_k"])
            if cl in prev and r["executor"] == "vla" and not r.get("training_triggered"):
                if cur != prev[cl]:
                    bad_ledger.append({"episode": r["t"], "cluster": cl,
                                       "before": prev[cl], "after": cur})
            prev[cl] = cur
        checks.append(chk(f"teacher_into_maturity_ledger_seed{s}", not bad_ledger,
                          {"n_violations": len(bad_ledger), "examples": bad_ledger[:5]}))

        # 8. lifecycle 전이 합법성
        illegal = [{"episode": r["t"], "cluster": r["cluster"],
                    "from": r["state_before"], "to": r["state_after"]}
                   for r in rows
                   if (r["state_before"] == "X" and r["state_after"] != "X")
                   or (r["state_after"] == "U" and r["state_before"] != "U")]
        checks.append(chk(f"invalid_lifecycle_transition_seed{s}", not illegal,
                          {"n_illegal": len(illegal), "examples": illegal[:5],
                           "rule": "X는 흡수 상태 · U로의 복귀 불가"}))

        # 9. 전이 로그 ↔ 에피소드 원장 일치
        led = {"demotion": sum(1 for r in rows if r.get("demotion")),
               "rematuration": sum(1 for r in rows if r.get("rematuration")),
               "transition_X": sum(1 for r in rows if r.get("transition_to_X")),
               "firing": sum(1 for r in rows if r["executor"] == "habit"),
               "routing_vla": sum(1 for r in rows if r["executor"] == "vla")}
        evc = {t: sum(1 for e in events if e["event_type"] == t)
               for t in ("demotion", "rematuration", "transition_X")}
        for k in ("demotion", "rematuration", "transition_X"):
            checks.append(chk(f"event_vs_ledger_{k}_seed{s}", led[k] == evc.get(k, 0),
                              {"episode_ledger": led[k], "event_log": evc.get(k, 0)},
                              expect="일치"))
        checks.append(chk(f"routing_partition_seed{s}",
                          led["firing"] + led["routing_vla"] == len(rows),
                          {"fire": led["firing"], "vla": led["routing_vla"],
                           "total_rows": len(rows)}, expect="합 = 총 행수"))

        # 10. 최종 M/I/X 계수 일치 (summary vs 마지막 state_after)
        smp = f"{seed_dir(s)}/summary_{s}.json"
        if os.path.exists(smp):
            fs = json.load(open(smp))["final_states"]
            last = {}
            for r in rows:
                last[r["cluster"]] = r["state_after"]
            mism = {c: (last[c], fs.get(c)) for c in last if last[c] != fs.get(c)}
            checks.append(chk(f"final_state_consistency_seed{s}", not mism,
                              {"n_mismatch": len(mism), "examples": dict(list(mism.items())[:5])},
                              expect="일치"))

        # 11. paired replay 정합
        cfp = f"{ROOT}/05_paired_replay/cf_{s}.jsonl"
        qp = f"{seed_dir(s)}/cf_queue_{s}.jsonl"
        if os.path.exists(cfp) and os.path.exists(qp):
            qu = {json.loads(l)["uid"] for l in open(qp)}
            cu = [json.loads(l)["uid"] for l in open(cfp)]
            fired = {r["spec_uid"] for r in rows if r["executor"] == "habit"}
            checks.append(chk(f"paired_replay_queue_match_seed{s}", qu == fired,
                              {"queue": len(qu), "fired_in_stream": len(fired),
                               "queue_minus_fired": len(qu - fired),
                               "fired_minus_queue": len(fired - qu)}, expect="집합 동일"))
            checks.append(chk(f"paired_replay_episode_mismatch_seed{s}",
                              not (set(cu) - qu),
                              {"n_replayed": len(cu), "not_in_queue": len(set(cu) - qu),
                               "duplicates": len(cu) - len(set(cu))}))

    # ---------- 12. RGB-only habit이 depth를 썼는가 (체크포인트 전수)
    import torch
    bad_ck, n_ck = [], 0
    for p in sorted(glob.glob(f"{CK_BATCH}/*/act_n*.pt") + glob.glob(f"{CK_ONLINE}/*/*/act_n*.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=False)
        n_ck += 1
        if sd.get("use_depth") is not False or sd.get("in_ch") != 3:
            bad_ck.append({"path": p, "use_depth": sd.get("use_depth"), "in_ch": sd.get("in_ch")})
    checks.append(chk("depth_used_by_rgb_only_habit", not bad_ck,
                      {"n_checkpoints": n_ck, "n_violations": len(bad_ck),
                       "examples": bad_ck[:5]}))

    # ---------- 결과
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    rep = {"run_id": os.path.basename(ROOT), "modality": "rgb_only",
           "n_checks": len(checks), "n_fail": n_fail,
           "overall": "VALID" if n_fail == 0 else "INVALID",
           "note": ("FAIL이 하나라도 있으면 overall=INVALID이며 해당 결과를 VALID로 표시하지 "
                    "않는다 (§14). 검사는 드라이버 카운터가 아니라 산출물에서 재구성한다."),
           "checks": checks}
    json.dump(rep, open(f"{OUT}/DATA_INTEGRITY_AUDIT.json", "w"), indent=1, ensure_ascii=False)
    for c in checks:
        if c["status"] == "FAIL":
            print(f"  [FAIL] {c['check']}: {c['detail']}")
    print(f"[INTEGRITY-DONE] checks={len(checks)} fail={n_fail} overall={rep['overall']}")


if __name__ == "__main__":
    main()
