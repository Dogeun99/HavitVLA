"""E0-3: depth 노출 확인 (설계서 §2.1 / docs/E0_INSTRUCTIONS.md E0-3).

OffScreenRenderEnv에 camera_depths=True를 전달해 depth 관측이 실제로 나오는지,
(H,W)·dtype·값 범위·에피소드 진행에 따른 변화를 검증한다.

실행: LIBERO_CONFIG_PATH=$HABIT2/.libero MUJOCO_GL=egl \
      conda run -n hv2_oft python -u experiments/e0_depth_check.py
"""
import json
import os
import sys

import numpy as np

# LIBERO_CONFIG_PATH 미지정 실행이 공용 ~/.libero를 오염시키는 사고 방지 (ISSUE-13 가드)
_HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(_HABIT2, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")

RESULT = os.path.join(os.path.dirname(__file__), "..", "results", "e0", "e0_3_depth.json")


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl,
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_depths": True,  # ★ 검증 대상
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = suite.get_task_init_states(0)
    obs = env.set_init_state(init_states[0])

    keys = sorted(obs.keys())
    depth_keys = [k for k in keys if "depth" in k]
    rgb_keys = [k for k in keys if "image" in k]

    report = {
        "task": task.name,
        "language": task.language,
        "all_obs_keys": keys,
        "rgb_keys": rgb_keys,
        "depth_keys": depth_keys,
        "depth": {},
        "status": "FAIL",
    }

    if not depth_keys:
        report["reason"] = "no depth key in obs — wrapper may filter or kwarg not passed"
    else:
        # 물체 정착 후 초기 프레임
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        first = {k: np.array(obs[k], dtype=np.float64) for k in depth_keys}

        # 팔을 움직여 depth 변화 유도 (전진 + 하강)
        for _ in range(30):
            obs, _, _, _ = env.step([0.5, 0.0, -0.5, 0, 0, 0, -1])
        second = {k: np.array(obs[k], dtype=np.float64) for k in depth_keys}

        ok = True
        for k in depth_keys:
            d0, d1 = first[k], second[k]
            rgb_match = any(obs[r].shape[:2] == d0.shape[:2] for r in rgb_keys)
            changed = float(np.abs(d1 - d0).mean())
            entry = {
                "shape": list(d0.shape),
                "dtype": str(obs[k].dtype),
                "min": float(d0.min()),
                "max": float(d0.max()),
                "mean": float(d0.mean()),
                "matches_rgb_hw": bool(rgb_match),
                "mean_abs_change_after_motion": changed,
                "finite": bool(np.isfinite(d0).all()),
                # robosuite depth는 [0,1] 정규화(OpenGL non-linear). 미터 변환은
                # camera_utils.get_real_depth_map 참조 — 판정 근거로 범위 기록.
                "looks_normalized_01": bool(0.0 <= d0.min() and d0.max() <= 1.0),
            }
            entry_ok = (
                entry["matches_rgb_hw"]
                and entry["finite"]
                and d0.std() > 1e-6
                and changed > 1e-6
            )
            entry["ok"] = bool(entry_ok)
            report["depth"][k] = entry
            ok &= entry_ok
        report["status"] = "PASS" if ok else "FAIL"

    env.close()

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[E0-PASS] item=E0-3 status={report['status']} json=results/e0/e0_3_depth.json")
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
