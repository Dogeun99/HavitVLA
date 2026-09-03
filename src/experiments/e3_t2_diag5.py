"""diag5 (연구원 승인 조건부 판정 블록) — task5 stage-2 비맹점 실패의 fresh 동등성 전수 검증.

대상: chained task5 수집 meta의 stage-2 실패 중 결정적 맹점 base {17, 28} 제외 (~20건).
각 건을 **표준 fresh 경로**(LiberoEpisodeEnv.begin_episode — v3 전환과 동일 구성:
seed = 에피소드 seed, state = perturbed_init_state(reloc_base, w, rng(reloc_noise)))로
동일 입력 재실행:
  - 세계 A: 전건 실패 재현 → v3 stage-2 ≡ fresh 확증, draw 민감성 설명 유지 → task6 진행.
  - 세계 B: 1건이라도 성공 → 비동등 신호 → 교체 중단·원인 격리·연구원 회부.
부가: --save-frames(시작·종료), 실패 양상 분류(엉뚱한 물체 투입 / 미완수 / predicate 이상),
대역 구성 동일성 검증(perturbed_init_state vs chained 인라인 섭동 — 함수 수준 수치 대조).

실행: hv2_oft python -u experiments/e3_t2_diag5.py --save-frames
출력: results/e3/t2_diag5.json (경로 유일성 — suffix 규칙 준수), 프레임 diag5_frames/.
"""
import argparse
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

from envs.libero_env import LiberoEpisodeEnv  # noqa: E402
from teacher.collector import load_teacher  # noqa: E402
from experiments.e3_t2_diag import teacher_chunk  # noqa: E402

BLINDSPOT = {17, 28}
FRAME_DIR = os.path.join(HABIT2, "data", "e3", "t2_smoke", "diag5_frames")
OUT = os.path.join(HABIT2, "results", "e3", "t2_diag5.json")


def body_xy(env, keyword):
    sim = env._env.env.sim
    for name in sim.model.body_names:
        if keyword in name and "main" not in name:
            return np.array(sim.data.get_body_xpos(name))[:2], name
    return None, None


def classify(env):
    """종료 상태 분류: 바스켓 xy ±10cm 내 물체 집합 기준."""
    sim = env._env.env.sim
    basket_xy, _ = body_xy(env, "basket")
    in_basket, tomato_in = [], False
    for name in sim.model.body_names:
        if not name.endswith("_main"):
            continue
        short = name[:-5]
        if "basket" in short:
            continue
        xy = np.array(sim.data.get_body_xpos(name))[:2]
        if basket_xy is not None and np.linalg.norm(xy - basket_xy) < 0.10:
            in_basket.append(short)
            if "tomato_sauce" in short:
                tomato_in = True
    if tomato_in and not env.check_success():
        return "target_in_basket_predicate_false", in_basket
    if in_basket:
        return "wrong_object_in_basket", in_basket
    return "no_object_in_basket", in_basket


def band_identity_check(env):
    """대역 구성 동일성: perturbed_init_state(공식 경로) vs chained 인라인 섭동 수치 대조."""
    checks = []
    for base, noise, w in [(3, 500_003, 0.01), (25, 500_008, 0.01), (30, 123, 0.01)]:
        s_official = env.perturbed_init_state(base, w, np.random.default_rng(noise))
        rng = np.random.default_rng(noise)
        s_inline = env.init_states[base].copy()
        for adr in env._free_adrs:
            s_inline[env._time_offset + adr: env._time_offset + adr + 2] += rng.uniform(-w, w, 2)
        checks.append({"base": base, "noise": noise,
                       "max_abs_diff": float(np.max(np.abs(s_official - s_inline)))})
    identical = all(c["max_abs_diff"] == 0.0 for c in checks)
    return {"identical": identical, "samples": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-frames", action="store_true")
    args = ap.parse_args()

    import cv2
    import h5py

    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    with h5py.File(os.path.join(HABIT2, "data", "e3", "chained_libero_object_task5.hdf5"), "r") as f:
        meta = json.loads(f["meta_json"][()])
    targets = [m for m in meta if m["outcome"] == "fail" and m.get("stage") == 2
               and m["relocate_base_idx"] not in BLINDSPOT]
    print(f"[diag5] 대상 {len(targets)}건 (stage-2 실패, 맹점 {sorted(BLINDSPOT)} 제외)", flush=True)

    set_seed_everywhere(7)
    teacher = load_teacher("libero_object")
    resize_size = get_image_resize_size(teacher[0])
    env = LiberoEpisodeEnv("libero_object", 5)
    os.makedirs(FRAME_DIR, exist_ok=True)

    band = band_identity_check(env)
    print(f"[diag5] 대역 구성 동일성: identical={band['identical']}", flush=True)

    results, n_repro_fail = [], 0
    for i, m in enumerate(targets):
        rng = np.random.default_rng(m["relocate_noise_seed"])
        state = env.perturbed_init_state(m["relocate_base_idx"], m["w"], rng)
        obs = env.begin_episode(m["seed"], state)
        if args.save_frames:
            cv2.imwrite(os.path.join(FRAME_DIR, f"{m['uid']}_start.png"),
                        cv2.cvtColor(obs["agentview_image"][::-1, ::-1], cv2.COLOR_RGB2BGR))
        t, success = 0, False
        while t < env.max_steps:
            for a in teacher_chunk(teacher, resize_size, obs, env.language):
                obs, _, done, _ = env.step(a.tolist())
                t += 1
                if done or t >= env.max_steps:
                    break
            if env.check_success():
                success = True
                break
        mode, in_basket = (None, None) if success else classify(env)
        if args.save_frames:
            cv2.imwrite(os.path.join(FRAME_DIR, f"{m['uid']}_end.png"),
                        cv2.cvtColor(obs["agentview_image"][::-1, ::-1], cv2.COLOR_RGB2BGR))
        if not success:
            n_repro_fail += 1
        results.append({"uid": m["uid"], "reloc_base": m["relocate_base_idx"],
                        "reloc_noise": m["relocate_noise_seed"], "seed": m["seed"],
                        "fresh_success": success, "steps": t,
                        "failure_mode": mode, "objects_in_basket": in_basket})
        print(f"  [{i+1}/{len(targets)}] base={m['relocate_base_idx']} fresh_success={success} "
              f"mode={mode}", flush=True)

    n = len(targets)
    modes = {}
    for r in results:
        if r["failure_mode"]:
            modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
    world = "A" if n_repro_fail == n else "B"
    report = {
        "n_targets": n, "n_reproduced_fail": n_repro_fail,
        "band_identity": band, "failure_mode_counts": modes,
        "world": world,
        "verdict": ("A: 전건 실패 재현 — v3 stage-2 ≡ fresh 확증, task6 진행" if world == "A"
                    else "B: fresh 성공 존재 — 비동등 신호, 교체 중단·연구원 회부"),
        "per_episode": results,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[DIAG5-WORLD-{world}] repro_fail={n_repro_fail}/{n} band_identical={band['identical']} "
          f"modes={modes} -> {os.path.relpath(OUT, HABIT2)}")
    env.close()


if __name__ == "__main__":
    main()
