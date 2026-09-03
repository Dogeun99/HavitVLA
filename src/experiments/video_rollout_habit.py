"""영상 지시서 v2 — habit 롤아웃 녹화 (hv2_hab, GPU).

V1: teacher 단계가 채택한 스펙(chosen.json)으로 habit(n=80) 실행 — **성공 재현 단언**
    (기록상 성공 스펙이므로 실패 시 결정성 위반 FAIL 보고).
V2: 기록된 habit(n=80) 실패 스펙 재실행 — **실패 재현 단언**.
산출: results/videos/_raw/{cluster}_{V1|V2}_habit.npz

실행: hv2_hab python -u experiments/video_rollout_habit.py
"""
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))

from envs.chained_env import ChainedEpisodeEnv, ChainedEpisodeSpec, execute_chunk_with_boundary  # noqa: E402
from envs.libero_env import EpisodeSpec, LiberoEpisodeEnv  # noqa: E402
from habits.policy import HabitPolicy  # noqa: E402

RAW = os.path.join(HABIT2, "results", "videos", "_raw")
CHUNK = 8


def spec_from_dict(d):
    if d.get("chained"):
        return ChainedEpisodeSpec(d["suite"], d["task_id"], d["seed"], d["base_idx"], d["w"],
                                  d["noise_seed"], d["relocate_base_idx"], d["relocate_noise_seed"])
    return EpisodeSpec(d["suite"], d["task_id"], seed=d["seed"], base_idx=d["base_idx"],
                       w=d["w"], noise_seed=d["noise_seed"])


def rollout_capture(env, spec, policy):
    """정본 실행기 사용 — 평가기와 동형 (§4e α: 전환 시 stale 폐기·재질의)."""
    obs = spec.realize(env)
    frames, queries_at = [obs["agentview_image"][::-1, ::-1].copy()], [0]
    state = {"n_q": 0}

    def on_step(o):
        frames.append(o["agentview_image"][::-1, ::-1].copy())
        queries_at.append(state["n_q"])

    t, success = 0, False
    while t < env.max_steps:
        chunk = policy.act_chunk(obs)
        state["n_q"] += 1
        obs, t, _n_exec, _stale = execute_chunk_with_boundary(
            env, list(chunk[:CHUNK]), t, env.max_steps, on_step=on_step)
        if env.check_success():
            success = True
            break
    stage1_end = env.stage_steps.get(1) if hasattr(env, "stage_steps") else None
    return np.stack(frames), np.array(queries_at, np.int16), success, t, state["n_q"], stage1_end


def main():
    man = json.load(open(os.path.join(HABIT2, "results", "videos", "manifest.json")))
    chosen = json.load(open(os.path.join(RAW, "chosen.json")))["chosen_v1"]
    failures = []

    for cl, v in man["clusters"].items():
        ckpt = os.path.join(HABIT2, "checkpoints", cl, "act_n80.pt")
        policy = HabitPolicy(ckpt)
        env = (ChainedEpisodeEnv(v["suite"], v["task"]) if v.get("chained")
               else LiberoEpisodeEnv(v["suite"], v["task"]))
        # --- V1 (채택 스펙, 성공 단언)
        if cl in chosen:
            uid = chosen[cl]
            cand = next(c for c in v["v1_candidates"] if c["uid"] == uid)
            spec = spec_from_dict(cand)
            frames, queries, succ, t, n_q, s1e = rollout_capture(env, spec, policy)
            if succ:
                np.savez_compressed(os.path.join(RAW, f"{cl}_V1_habit.npz"), frames=frames,
                                    queries=queries, meta=json.dumps(
                                        {"uid": uid, "success": True, "steps": t,
                                         "n_queries": n_q, "stage1_end": s1e}))
                print(f"[V1-habit] {cl}: uid={uid} steps={t} q={n_q}", flush=True)
            else:
                failures.append(f"{cl} V1: 기록상 성공 스펙 {uid}이 재현에서 실패 — 결정성 위반")
                print(f"[V1-REPRO-FAIL] {cl}", flush=True)
        # --- V2 (실패 스펙, 실패 단언)
        if v.get("v2_fail_spec"):
            spec = spec_from_dict(v["v2_fail_spec"])
            frames, queries, succ, t, n_q, s1e = rollout_capture(env, spec, policy)
            if succ:
                failures.append(f"{cl} V2: 기록상 실패 스펙 {spec.uid}이 재현에서 성공 — 결정성 위반")
                print(f"[V2-REPRO-FAIL] {cl}: {spec.uid}", flush=True)
            else:
                np.savez_compressed(os.path.join(RAW, f"{cl}_V2_habit.npz"), frames=frames,
                                    queries=queries, meta=json.dumps(
                                        {"uid": spec.uid, "success": False, "steps": t,
                                         "n_queries": n_q, "stage1_end": s1e}))
                print(f"[V2-habit] {cl}: uid={spec.uid} 실패 재현 OK steps={t}", flush=True)
        env.close()
        del policy
        import torch

        torch.cuda.empty_cache()

    prev = json.load(open(os.path.join(RAW, "chosen.json")))
    prev["habit_failures"] = failures
    json.dump(prev, open(os.path.join(RAW, "chosen.json"), "w"), indent=2, ensure_ascii=False)
    marker = "PASS" if not failures else f"WITH-FAILURES({len(failures)})"
    print(f"[VIDEO-HABIT-{marker}]")
    for f_ in failures:
        print("  FAIL:", f_)


if __name__ == "__main__":
    main()
