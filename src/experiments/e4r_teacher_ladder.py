"""E4-R 후속 통제군 — teacher w-사다리 (§5 등재, 판독 규칙 사전 고정).

왜: 게이트의 이득 = "더 나은 실행자로 라우팅". teacher가 넓은 w에서 잘하면 미탐은 실질
손실(H3 현행 유지), teacher도 무너지면 기각해도 갈 곳이 없어 미탐이 손실이 아니다(H3 재작성).
설계: 3 클러스터 × w{0.01,0.02,0.04,0.06,0.08} × 15 ep, teacher(OFT) rollout,
      **E4-R과 동일 스펙 uid(paired)** + 동일 물리 유효성 필터, 동일 실행기 경로.
산출: results/e4/e4r_teacher_ladder.json + [E4R-TEACHER-PASS]
실행: hv2_oft python -u experiments/e4r_teacher_ladder.py
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

from envs.chained_env import execute_chunk_with_boundary  # noqa: E402
from envs.libero_env import EpisodeSpec, InfraError, LiberoEpisodeEnv, USABLE_W_MAX  # noqa: E402
from teacher.collector import load_teacher, teacher_observation  # noqa: E402

E4 = os.path.join(HABIT2, "results", "e4")
OUT = os.path.join(E4, "e4r_teacher_ladder.json")
CLUSTERS = [("libero_object", 0), ("libero_object", 5), ("libero_goal", 2)]
W_GRID = [0.01, 0.02, 0.04, 0.06, 0.08]
N_PER_W, CHUNK = 15, 8
SEED_BASE, NOISE_BASE = 30_000, 2_000_000  # E4-R과 동일 대역식 (paired)
BASES = list(range(40, 50))
_ref = {}


def settle_ref(env, key, state):
    if key in _ref:
        return _ref[key]
    env.begin_episode(90_000 + (hash(key) % 1000), state)
    sim = env._env.env.sim
    _ref[key] = {adr: np.array(sim.data.qpos[adr:adr + 3]) for adr in env._free_adrs}
    return _ref[key]


def settled_validity(env, target_state, ref_pos):
    sim = env._env.env.sim
    off = env._time_offset
    for adr in env._free_adrs:
        cur = sim.data.qpos[adr:adr + 3]
        tgt = target_state[off + adr:off + adr + 3]
        if float(ref_pos[adr][2]) - float(cur[2]) > 0.05:
            return False, "z_drop"
        if float(np.linalg.norm(np.asarray(cur[:2]) - np.asarray(tgt[:2]))) > 0.10:
            return False, "lateral"
    return True, None


def main():
    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import get_vla_action
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    habit = json.load(open(os.path.join(E4, "e4r_competence_map.json")))
    rep = {"note": "teacher w-사다리 통제군 (§5). E4-R과 동일 스펙 uid·필터·실행기.",
           "w_grid": W_GRID, "clusters": {}}
    set_seed_everywhere(7)
    by_suite = {}
    for s, t in CLUSTERS:
        by_suite.setdefault(s, []).append(t)

    for suite, tasks in by_suite.items():
        teacher = load_teacher(suite)
        cfg, resize = teacher[0], get_image_resize_size(teacher[0])

        def act(obs, lang):
            t_obs = teacher_observation(obs, resize)
            chunk = get_vla_action(cfg, teacher[1], teacher[5], t_obs, lang,
                                   action_head=teacher[2], proprio_projector=teacher[3],
                                   noisy_action_projector=teacher[4])
            return [process_action(np.asarray(a, dtype=np.float32), cfg.model_family)
                    for a in chunk[:CHUNK]]

        for task in tasks:
            cl = f"{suite}_task{task}"
            env = LiberoEpisodeEnv(suite, task)
            entry = {"usable_w_max": USABLE_W_MAX[suite], "by_w": {}}
            for wi, w in enumerate(W_GRID):
                for b in sorted(set(BASES[j % 10] for j in range(N_PER_W))):
                    settle_ref(env, f"{cl}|{b}", env.init_states[b])
                n_valid = n_succ = n_rej = n_infra = 0
                eps = []
                for j in range(N_PER_W):
                    base = BASES[j % 10]
                    seed = SEED_BASE + 1000 * (wi + 1) + j
                    noise = NOISE_BASE + 1000 * (wi + 1) + j
                    spec = EpisodeSpec(suite, task, seed, base, w, noise)
                    try:
                        state = env.perturbed_init_state(base, w, np.random.default_rng(noise),
                                                         allow_beyond_usable=True)
                        obs = env.begin_episode(seed, state)
                        ok, why = settled_validity(env, state, _ref[f"{cl}|{base}"])
                        if not ok:
                            n_rej += 1
                            eps.append({"uid": spec.uid, "valid": False, "why": why})
                            print(f"[E4RT] {cl} w={w} {j+1}/{N_PER_W} invalid({why})", flush=True)
                            continue
                        t, success = 0, False
                        while t < env.max_steps:
                            obs, t, _n, _s = execute_chunk_with_boundary(
                                env, act(obs, env.language), t, env.max_steps)
                            if env.check_success():
                                success = True
                                break
                    except InfraError as e:
                        n_infra += 1
                        eps.append({"uid": spec.uid, "outcome": "infra_error", "error": str(e)})
                        continue
                    n_valid += 1
                    n_succ += int(success)
                    eps.append({"uid": spec.uid, "valid": True, "success": bool(success), "steps": t})
                    print(f"[E4RT] {cl} w={w} {j+1}/{N_PER_W} teacher_succ={success}", flush=True)
                hb = habit["clusters"][cl]["by_w"][str(w)]
                tr = round(n_succ / n_valid, 4) if n_valid else None
                entry["by_w"][str(w)] = {
                    "n_valid": n_valid, "n_reject_physical": n_rej, "n_infra": n_infra,
                    "teacher_success_rate": tr, "habit_success_rate": hb["habit_success_rate"],
                    "gap_teacher_minus_habit": (round(tr - hb["habit_success_rate"], 4)
                                                if tr is not None and hb["habit_success_rate"] is not None
                                                else None),
                    "episodes": eps}
                print(f"[E4RT-W] {cl} w={w}: teacher={tr} habit={hb['habit_success_rate']} "
                      f"격차={entry['by_w'][str(w)]['gap_teacher_minus_habit']} 물리탈락={n_rej}/{N_PER_W}",
                      flush=True)
            rep["clusters"][cl] = entry
            env.close()
        del teacher
        import torch

        torch.cuda.empty_cache()

    gaps = {str(w): [rep["clusters"][f"{s}_task{t}"]["by_w"][str(w)]["gap_teacher_minus_habit"]
                     for s, t in CLUSTERS] for w in W_GRID}
    rep["summary"] = {
        "teacher_by_w": {str(w): round(float(np.mean(
            [rep["clusters"][f"{s}_task{t}"]["by_w"][str(w)]["teacher_success_rate"]
             for s, t in CLUSTERS])), 4) for w in W_GRID},
        "habit_by_w": {str(w): round(float(np.mean(
            [rep["clusters"][f"{s}_task{t}"]["by_w"][str(w)]["habit_success_rate"]
             for s, t in CLUSTERS])), 4) for w in W_GRID},
        "gap_by_w": {w: round(float(np.mean([g for g in v if g is not None])), 4)
                     for w, v in gaps.items()},
    }
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    s = rep["summary"]
    print(f"[E4R-TEACHER-PASS] teacher/w={s['teacher_by_w']} | habit/w={s['habit_by_w']} | "
          f"격차/w={s['gap_by_w']}")


if __name__ == "__main__":
    main()
