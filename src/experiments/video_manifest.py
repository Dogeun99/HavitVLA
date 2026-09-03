"""영상 지시서 v2 — 매니페스트 생성 (CPU).

클러스터별로:
  V1: habit(n=80) 성공 스펙 후보(순서대로, 최대 5) — teacher 검증은 rollout 단계에서.
  V2: habit(n=80) 실패 스펙 1개 (실패 기록 있는 클러스터만).
  V3: teacher 수집 실패 스펙 1개 (task5 = base17 맹점 필수 + MEMO 플래그).
chained 2클러스터는 체크포인트 부재(트리거 정지)로 deferred 명시.
스펙 파라미터는 heldout_specs 재생성(uid 대조) / 수집 meta에서 프로그래밍 취득.

실행: hv2_hab python -u experiments/video_manifest.py → results/videos/manifest.json
"""
import json
import os
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)

from envs.stream import heldout_specs  # noqa: E402

STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}
CHAINED = ["chained_libero_object_task0", "chained_libero_object_task5"]
V1_MAX_CANDIDATES = 5


def curve80(cl, suite, task):
    """n=80 per_episode + uid→spec 매핑."""
    if (suite, task) in E2_REUSE:
        p, n_heldout = os.path.join(HABIT2, "results", "e2", f"{cl}_curve.json"), 50
    else:
        p, n_heldout = os.path.join(HABIT2, "results", "e3", f"{cl}_curve.json"), 20
    d = json.load(open(p))
    entry = next(c for c in d["curve"] if c["n"] == 80)
    spec_by_uid = {s.uid: s for s in heldout_specs(suite, task, n_heldout)}
    return entry["per_episode"], spec_by_uid


def teacher_fail_spec(cl, suite, task):
    import h5py

    ddir = "e2" if (suite, task) in E2_REUSE else "e3"
    p = os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5")
    with h5py.File(p, "r") as f:
        meta = json.loads(f["meta_json"][()])
    fails = [m for m in meta if m["outcome"] == "fail"]
    if not fails:
        return None
    # task5: base17 결정적 맹점 필수 (지시서 V3)
    if (suite, task) == ("libero_object", 5):
        b17 = [m for m in fails if m["base_idx"] == 17]
        if b17:
            m = b17[0]
            m["_memo"] = "MEMO: deterministic blindspot (E1/E2/diag triple-confirmed; bases {17,28})"
            return m
    return fails[0]


def main():
    registry = json.load(open(os.path.join(HABIT2, "configs", "task_registry.json")))
    man = {"clusters": {}, "deferred": {}}
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        lang = registry[suite][str(task)]
        eps, spec_by_uid = curve80(cl, suite, task)
        succ = [e for e in eps if e["outcome"] == "success"]
        fail = [e for e in eps if e["outcome"] == "fail"]
        entry = {
            "suite": suite, "task": task, "language": lang,
            "v1_candidates": [spec_by_uid[e["uid"]].to_dict() for e in succ[:V1_MAX_CANDIDATES]],
            "v2_fail_spec": spec_by_uid[fail[0]["uid"]].to_dict() if fail else None,
        }
        tf = teacher_fail_spec(cl, suite, task)
        if tf:
            entry["v3_fail_spec"] = {k: tf[k] for k in ("uid", "suite", "task_id", "seed",
                                                        "base_idx", "w", "noise_seed")}
            if "_memo" in tf:
                entry["v3_memo"] = tf["_memo"]
        else:
            entry["v3_fail_spec"] = None
        man["clusters"][cl] = entry
    # chained: 재수집본(α, chunk-break 실행기) 산출물이 있으면 27 클러스터로 확장, 없으면 보류
    import h5py

    from envs.chained_env import chained_heldout_specs

    for cl in CHAINED:
        task = int(cl.rsplit("task", 1)[1])
        curve_p = os.path.join(HABIT2, "results", "e3", f"{cl}_curve.json")
        ckpt_p = os.path.join(HABIT2, "checkpoints", cl, "act_n80.pt")
        h5_p = os.path.join(HABIT2, "data", "e3", f"{cl}.hdf5")
        if not (os.path.exists(curve_p) and os.path.exists(ckpt_p) and os.path.exists(h5_p)):
            man["deferred"][cl] = ("α 판정(§5 2026-08-15) 재수집본 기준 갱신 예약 — chunk-break "
                                   "실행기 재수집·학습 완료 후 §6 재개 시 생성")
            continue
        d = json.load(open(curve_p))
        entry80 = next(c for c in d["curve"] if c["n"] == 80)
        n_h = d.get("n_heldout", 20)
        spec_by_uid = {s.uid: s for s in chained_heldout_specs("libero_object", task, n_h)}
        succ = [e for e in entry80["per_episode"] if e["outcome"] == "success"]
        fail = [e for e in entry80["per_episode"] if e["outcome"] == "fail"]
        with h5py.File(h5_p, "r") as f:
            meta = json.loads(f["meta_json"][()])
        t_fails = [m for m in meta if m["outcome"] == "fail"]
        entry = {
            "suite": "libero_object", "task": task, "chained": True,
            "language": registry["libero_object"][str(task)],
            "v1_candidates": [spec_by_uid[e["uid"]].to_dict() for e in succ[:V1_MAX_CANDIDATES]],
            "v2_fail_spec": spec_by_uid[fail[0]["uid"]].to_dict() if fail else None,
            "v3_fail_spec": ({k: t_fails[0][k] for k in ("uid", "suite", "task_id", "seed",
                                                         "base_idx", "w", "noise_seed",
                                                         "relocate_base_idx", "relocate_noise_seed")}
                             | {"chained": True}) if t_fails else None,
        }
        man["clusters"][cl] = entry

    out = os.path.join(HABIT2, "results", "videos", "manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(man, open(out, "w"), indent=2, ensure_ascii=False)
    n = len(man["clusters"])
    n_v2 = sum(1 for v in man["clusters"].values() if v["v2_fail_spec"])
    n_v3 = sum(1 for v in man["clusters"].values() if v["v3_fail_spec"])
    print(f"[VIDEO-MANIFEST] {n} clusters (deferred {len(man['deferred'])}): "
          f"V1 후보 {n}, V2 대상 {n_v2}, V3 대상 {n_v3} -> {out}")


if __name__ == "__main__":
    main()
