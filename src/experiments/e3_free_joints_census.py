"""§4f 공변량: 40 태스크 free-joint 전수조사 (CPU 전용 — 렌더러 없음).

`e0_6_variation.json`은 스위트당 task0 1개만 측정 → H2-L′ 회귀(preregistration §4f)에는
클러스터별 free_joints가 필요해 전수조사를 신설한다. 기존 e0_6 파일은 불변.
ControlEnv(카메라·렌더러 비활성)로 MuJoCo 모델만 구축 — GPU/EGL 미사용이라 실행 중인
학습 배치와 안전하게 병행 가능 (실측 태스크당 ~2.4s).

실행: $HV2_HAB_PY -u experiments/e3_free_joints_census.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "e3", "free_joints_census.json")

# env var 누락 실행이 공용 ~/.libero·전역 캐시를 오염시키는 경로 차단 (ISSUE-13 가드)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(ROOT, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# e0_6 실측(스위트당 task0)과의 정합 검사 기준
E0_6_TASK0 = {"libero_spatial": 5, "libero_object": 7, "libero_goal": 4, "libero_10": 8}


def main():
    from libero.libero import benchmark, get_libero_path
    # ControlEnv는 패키지 __init__에서 재수출되지 않음 — 모듈에서 직접 import
    from libero.libero.envs.env_wrapper import ControlEnv

    out = {
        "note": "free joint = sim.model.jnt_type == 0 (mjJNT_FREE). "
        "H2-L' 회귀 공변량 단일 진입점 (preregistration §4f).",
        "suites": {},
    }
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        rows = {}
        for tid in range(suite.n_tasks):
            task = suite.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = ControlEnv(
                bddl_file_name=bddl,
                use_camera_obs=False,
                has_offscreen_renderer=False,
                has_renderer=False,
            )
            sim = env.env.sim
            n_free = sum(1 for j in range(sim.model.njnt) if sim.model.jnt_type[j] == 0)
            nq = int(sim.model.nq)  # close() 전에 캡처 — close 후 sim.model 접근 불가
            rows[str(tid)] = {
                "task": task.name,
                "language": task.language,
                "n_free_joints": n_free,
                "nq": nq,
            }
            env.close()
            print(f"[{suite_name} task{tid}] free={n_free} nq={nq}", flush=True)
        # e0_6 task0 실측과 정합 검사 — 불일치는 카운팅 회귀(regression)이므로 즉시 FAIL
        got = rows["0"]["n_free_joints"]
        if got != E0_6_TASK0[suite_name]:
            raise RuntimeError(
                f"{suite_name} task0 free_joints={got} != e0_6 실측 {E0_6_TASK0[suite_name]}"
            )
        out["suites"][suite_name] = rows

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[CENSUS-PASS] 40 tasks -> {OUT}")


if __name__ == "__main__":
    main()
