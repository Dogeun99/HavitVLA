"""영상 지시서 v2 — teacher 롤아웃 녹화 (hv2_oft, GPU).

V1: 클러스터별 habit-성공 후보 스펙을 순서대로 teacher fresh 실행 — 성공하는 첫 스펙 채택
    (녹화 + 성공 단언). 채택 uid는 chosen.json에 기록 → habit 녹화 단계가 사용.
V3: teacher 수집 실패 스펙 fresh 재실행 — **실패 재현 단언** (성공해버리면 결정성 위반 FAIL
    보고, 영상 미생성).
산출: results/videos/_raw/{cluster}_{V1|V3}_teacher.npz (frames u8, queries/step, meta)

실행: hv2_oft python -u experiments/video_rollout_teacher.py
"""
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.chained_env import ChainedEpisodeEnv, ChainedEpisodeSpec, execute_chunk_with_boundary  # noqa: E402
from envs.libero_env import EpisodeSpec, LiberoEpisodeEnv  # noqa: E402
from teacher.collector import load_teacher  # noqa: E402
from experiments.e3_t2_diag import teacher_chunk  # noqa: E402

RAW = os.path.join(HABIT2, "results", "videos", "_raw")
CHUNK = 8


def spec_from_dict(d):
    if d.get("chained"):
        return ChainedEpisodeSpec(d["suite"], d["task_id"], d["seed"], d["base_idx"], d["w"],
                                  d["noise_seed"], d["relocate_base_idx"], d["relocate_noise_seed"])
    return EpisodeSpec(d["suite"], d["task_id"], seed=d["seed"], base_idx=d["base_idx"],
                       w=d["w"], noise_seed=d["noise_seed"])


def make_env(v, suite, task):
    return ChainedEpisodeEnv(suite, task) if v.get("chained") else LiberoEpisodeEnv(suite, task)


def rollout_capture(env, spec, teacher, resize_size):
    """정본 실행기(execute_chunk_with_boundary)로 rollout — 프레임은 on_step 캡처.
    chained면 stage 전환 시 stale 폐기·재질의가 수집·평가와 동형으로 재현된다 (§4e α)."""
    obs = spec.realize(env)
    frames, queries_at = [obs["agentview_image"][::-1, ::-1].copy()], [0]
    state = {"n_q": 0}

    def on_step(o):
        frames.append(o["agentview_image"][::-1, ::-1].copy())
        queries_at.append(state["n_q"])

    t, success = 0, False
    while t < env.max_steps:
        actions = teacher_chunk(teacher, resize_size, obs, env.language)
        state["n_q"] += 1
        obs, t, _n_exec, _stale = execute_chunk_with_boundary(
            env, actions, t, env.max_steps, on_step=on_step)
        if env.check_success():
            success = True
            break
    stage1_end = env.stage_steps.get(1) if hasattr(env, "stage_steps") else None
    return np.stack(frames), np.array(queries_at, np.int16), success, t, state["n_q"], stage1_end


def save_npz(path, frames, queries, meta):
    np.savez_compressed(path, frames=frames, queries=queries, meta=json.dumps(meta))


def main():
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    man = json.load(open(os.path.join(HABIT2, "results", "videos", "manifest.json")))
    os.makedirs(RAW, exist_ok=True)
    chosen, failures = {}, []

    by_suite = {}
    for cl, v in man["clusters"].items():
        by_suite.setdefault(v["suite"], []).append((cl, v))

    set_seed_everywhere(7)
    for suite, items in by_suite.items():
        teacher = load_teacher(suite)
        resize_size = get_image_resize_size(teacher[0])
        for cl, v in items:
            env = make_env(v, suite, v["task"])
            # --- V1: 후보 순회, teacher 성공 첫 스펙 채택
            picked = None
            for cand in v["v1_candidates"]:
                spec = spec_from_dict(cand)
                frames, queries, succ, t, n_q, s1e = rollout_capture(env, spec, teacher, resize_size)
                if succ:
                    save_npz(os.path.join(RAW, f"{cl}_V1_teacher.npz"), frames, queries,
                             {"uid": spec.uid, "success": True, "steps": t, "n_queries": n_q,
                              "stage1_end": s1e})
                    picked = spec.uid
                    break
            if picked:
                chosen[cl] = picked
                print(f"[V1-teacher] {cl}: uid={picked} steps={t} q={n_q}", flush=True)
            else:
                failures.append(f"{cl} V1: teacher가 habit-성공 후보 {len(v['v1_candidates'])}개 전부 실패")
                print(f"[V1-teacher-FAIL] {cl}", flush=True)
            # --- V3: 실패 재현 단언
            if v.get("v3_fail_spec"):
                spec = spec_from_dict(v["v3_fail_spec"])
                frames, queries, succ, t, n_q, s1e = rollout_capture(env, spec, teacher, resize_size)
                if succ:
                    failures.append(f"{cl} V3: 실패 스펙 {spec.uid}이 재현 실행에서 성공 — 결정성 위반")
                    print(f"[V3-REPRO-FAIL] {cl}: {spec.uid} 성공해버림", flush=True)
                else:
                    save_npz(os.path.join(RAW, f"{cl}_V3_teacher.npz"), frames, queries,
                             {"uid": spec.uid, "success": False, "steps": t, "n_queries": n_q,
                              "memo": v.get("v3_memo"), "stage1_end": s1e})
                    print(f"[V3-teacher] {cl}: uid={spec.uid} 실패 재현 OK steps={t}", flush=True)
            env.close()
        del teacher
        import torch

        torch.cuda.empty_cache()

    json.dump({"chosen_v1": chosen, "failures": failures},
              open(os.path.join(RAW, "chosen.json"), "w"), indent=2, ensure_ascii=False)
    marker = "PASS" if not failures else f"WITH-FAILURES({len(failures)})"
    print(f"[VIDEO-TEACHER-{marker}] V1 채택 {len(chosen)}/{len(man['clusters'])}")
    for f_ in failures:
        print("  FAIL:", f_)


if __name__ == "__main__":
    main()
