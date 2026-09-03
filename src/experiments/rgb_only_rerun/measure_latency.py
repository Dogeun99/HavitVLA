"""§13 LATENCY / COMPUTE 측정 (RGB-only). publication figure/table 없음 — 숫자·raw timing만.

측정 프로토콜은 e1_latency.py와 동일: warmup 10회 후 100회, torch.cuda.synchronize로 경계 고정,
attn=sdpa (flash-attn은 sm_120 미빌드).
  - RGB-only ACT forward (in_ch=3, 3채널 입력) — 본 run의 습관 실행 비용
  - RGB-D ACT forward (in_ch=4) — 대조 (기존 run의 비용 단위)
  - gate path (DINOv2 + PCA + 2단 판정)
  - teacher OFT chunk forward — 동결 자산이지만 재측정해 동일 환경 값을 남긴다
  - 재학습/probe/형성 이벤트 시간은 온라인 원장에서 산출
산출: 07_latency_cost/{LATENCY_RAW.csv, FORMATION_TIMING_RAW.csv, COMPUTE_SUMMARY.json}
실행: hv2_oft python -u experiments/rgb_only_rerun/measure_latency.py
마커: [LATENCY-DONE]
"""
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from experiments.rgb_only_rerun.runner import ROOT  # noqa: E402

OUT = f"{ROOT}/07_latency_cost"
SEEDS = (0, 1, 2)
N_WARM, N_MEAS = 10, 100
RAW = []            # (anchor, idx, ms)


def timed(anchor, fn, n_warm=N_WARM, n_meas=N_MEAS):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(n_meas):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        ts.append(ms)
        RAW.append({"anchor": anchor, "sample_idx": i, "ms": round(ms, 4)})
    a = np.array(ts)
    return {"median_ms": round(float(np.median(a)), 3), "mean_ms": round(float(a.mean()), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "min_ms": round(float(a.min()), 3), "max_ms": round(float(a.max()), 3),
            "sd_ms": round(float(a.std(ddof=1)), 3), "n": n_meas}


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {"run_id": os.path.basename(ROOT), "modality": "rgb_only",
           "attn": "sdpa (flash-attn 미빌드, sm_120)",
           "gpu": torch.cuda.get_device_name(),
           "protocol": f"warmup {N_WARM} + measure {N_MEAS}, cuda.synchronize 경계"}

    # ---- ACT forward: RGB-only(주) + RGB-D(대조)
    from habits.act import ACTPolicy
    for tag, ch in (("act_forward_rgb_only", 3), ("act_forward_rgbd", 4)):
        m = ACTPolicy(pretrained=False, in_ch=ch).cuda().eval()
        img = [torch.randn(1, ch, 128, 128, device="cuda"),
               torch.randn(1, ch, 128, 128, device="cuda")]
        p = torch.randn(1, 8, device="cuda")
        rep[tag] = timed(tag, lambda: m.act(img, p))
        rep[tag]["in_ch"] = ch
        rep[tag]["n_params"] = sum(x.numel() for x in m.parameters())
        del m, img, p
        torch.cuda.empty_cache()

    # ---- gate path
    from gates.features import DinoFeatureExtractor, SharedPCA, prep_gate_rgb
    from gates.two_stage import JurisdictionGate, MaturityGate
    dino = DinoFeatureExtractor()
    rng = np.random.default_rng(0)
    fake = prep_gate_rgb(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8))
    pca = SharedPCA().fit(rng.normal(size=(200, 384)))
    jur = JurisdictionGate().fit(rng.normal(size=(40, 32)))
    mat = MaturityGate()
    for _ in range(30):
        mat.update(True, source="fire")

    def gate_path():
        z = pca.transform(dino.embed([fake]))
        _ = jur.accepts(z[0]) and mat.accepts()

    rep["gate_path"] = timed("gate_path", gate_path)
    del dino
    torch.cuda.empty_cache()

    # ---- teacher OFT chunk forward (동결 자산 — 동일 환경 재측정)
    from experiments.robot.openvla_utils import get_vla_action
    from teacher.collector import load_teacher
    cfg, model, ah, pp, nap, proc = load_teacher("libero_spatial")
    with open(os.path.join(HABIT2, "third_party", "openvla-oft", "experiments", "robot",
                           "libero", "sample_libero_spatial_observation.pkl"), "rb") as f:
        obs = pickle.load(f)
    rep["teacher_oft_chunk_forward"] = timed(
        "teacher_oft_chunk_forward",
        lambda: get_vla_action(cfg, model, proc, obs, obs["task_description"],
                               action_head=ah, proprio_projector=pp, noisy_action_projector=nap))
    rep["teacher_source"] = {
        "note": "teacher는 §1 동결 자산이나, 동일 환경에서 재측정해 본 run의 값을 남긴다.",
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "previous_run_value_ms": None}
    prev = "results/e1/e1_latency.json"
    if os.path.exists(prev):
        p = json.load(open(prev))
        rep["teacher_source"]["previous_run_value_ms"] = \
            p["anchor1_oft_chunk_forward"]["median_ms"]
        rep["teacher_source"]["previous_source_file"] = prev
    del model, ah, pp, nap, proc
    torch.cuda.empty_cache()

    # ---- 비율 (VLA-호출 등가의 분모 = teacher per-chunk)
    tv = rep["teacher_oft_chunk_forward"]["median_ms"]
    rep["ratios"] = {
        "act_rgb_only_over_teacher": round(rep["act_forward_rgb_only"]["median_ms"] / tv, 5),
        "act_rgbd_over_teacher": round(rep["act_forward_rgbd"]["median_ms"] / tv, 5),
        "gate_over_teacher": round(rep["gate_path"]["median_ms"] / tv, 5),
        "denominator_ms": tv,
        "note": "VLA-호출 등가 환산의 분모 = teacher per-chunk forward (8 steps)"}

    # ---- 형성 타이밍 (온라인 원장에서 — 하드코딩 없음)
    ftrows = []
    for s in SEEDS:
        p = f"{seed_dir(s)}/stream_{s}.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            ev = r.get("retrain_event")
            if not ev:
                continue
            train_s = ev.get("train_wall_s")
            total_s = ev.get("formation_wall_s")
            ftrows.append({
                "seed": s, "episode": r.get("t"), "cluster_id": r.get("cluster"),
                "suite": r.get("suite"), "task_id": r.get("task_id"),
                "n": ev.get("n"), "probe_round": ev.get("probe_round"),
                "policy_version": ev.get("version"),
                "bc_pool_at_trigger": ev.get("bc_pool_at_trigger"),
                "train_wall_s": train_s,
                "probe_and_prep_wall_s": (round(total_s - train_s, 2)
                                          if (train_s is not None and total_s is not None)
                                          else None),
                "formation_event_wall_s": total_s,
                "probe_episodes": ev.get("formation_episodes"),
                "probe_habit_calls": ev.get("probe_habit_calls"),
                "probe_success_count": ev.get("probe_success_count"),
                "probe_failure_count": ev.get("probe_failure_count"),
                "passed": ev.get("passed")})
    if ftrows:
        with open(f"{OUT}/FORMATION_TIMING_RAW.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ftrows[0].keys()))
            w.writeheader()
            w.writerows(ftrows)

    def agg(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return None
        return {"n": len(v), "mean_s": round(float(np.mean(v)), 2),
                "median_s": round(float(np.median(v)), 2),
                "sd_s": round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0,
                "min_s": round(float(min(v)), 2), "max_s": round(float(max(v)), 2),
                "total_s": round(float(np.sum(v)), 1)}

    rep["formation_timing"] = {
        "n_events": len(ftrows),
        "training_by_n": {str(n): agg([r["train_wall_s"] for r in ftrows if r["n"] == n])
                          for n in sorted({r["n"] for r in ftrows})} if ftrows else {},
        "formation_event_total": agg([r["formation_event_wall_s"] for r in ftrows]),
        "training_only": agg([r["train_wall_s"] for r in ftrows]),
        "probe_and_prep": agg([r["probe_and_prep_wall_s"] for r in ftrows]),
        "note": "재학습 이벤트 = 학습 + P=20 probe + 준비. 세 성분을 분리 계측했다."}

    # ---- 운영 시간 (seed별 장부)
    oper = {}
    for s in SEEDS:
        p = f"{seed_dir(s)}/summary_{s}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            oper[str(s)] = {"operational_s": d["ledger_s"]["operational_s"],
                            "formation_s": d["ledger_s"]["formation_s"],
                            "formation_episodes": d["ledger_s"]["formation_episodes"],
                            "total_wall_s": d["total_wall_s"]}
    rep["operational_time_per_seed"] = oper
    if oper:
        rep["operational_time_summary"] = {
            "operational_h": agg([v["operational_s"] / 3600 for v in oper.values()]),
            "formation_h": agg([v["formation_s"] / 3600 for v in oper.values()])}

    # ---- VLA-호출 등가 환산 (형성 비용)
    if ftrows and tv:
        per_call_s = tv / 1000.0
        rep["vla_call_equivalents"] = {
            "denominator_s_per_call": round(per_call_s, 6),
            "per_formation_event": round(
                float(np.mean([r["formation_event_wall_s"] for r in ftrows
                               if r["formation_event_wall_s"] is not None])) / per_call_s, 1),
            "per_training_only": round(
                float(np.mean([r["train_wall_s"] for r in ftrows
                               if r["train_wall_s"] is not None])) / per_call_s, 1),
            "by_n": {str(n): round(float(np.mean(
                [r["train_wall_s"] for r in ftrows if r["n"] == n
                 and r["train_wall_s"] is not None])) / per_call_s, 1)
                for n in sorted({r["n"] for r in ftrows})}}

    with open(f"{OUT}/LATENCY_RAW.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anchor", "sample_idx", "ms"])
        w.writeheader()
        w.writerows(RAW)
    json.dump(rep, open(f"{OUT}/COMPUTE_SUMMARY.json", "w"), indent=1, ensure_ascii=False)
    print(f"[LATENCY-DONE] act_rgb_only={rep['act_forward_rgb_only']['median_ms']}ms "
          f"(p95={rep['act_forward_rgb_only']['p95_ms']}) "
          f"act_rgbd={rep['act_forward_rgbd']['median_ms']}ms "
          f"gate={rep['gate_path']['median_ms']}ms teacher={tv}ms "
          f"formation_events={len(ftrows)} raw_samples={len(RAW)}")


if __name__ == "__main__":
    main()
