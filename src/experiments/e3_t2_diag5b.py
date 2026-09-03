"""diag5b — 세계 B 원인 확증: "stale chunk tail" 가설.

가설: v3 전환이 chunk 중간에 발생하면 실행기가 전환 **전** 관측으로 계산된 잔여 행동
(stale tail, 0–7개)을 전환 **후** 홈 포즈에서 계속 실행 → 첫 fresh 질의 전에 팔이 교란.
task5의 취약 판별에서 실패 유발 (task0은 강건 — 118/120).

검증: diag5에서 fresh 성공으로 뒤집힌 18건을 **chunk-break 실행기**(전환 감지 시 잔여
행동 폐기·즉시 재질의)로 chained 전 구간 재실행 — 가설이 맞으면 대부분 성공으로 전환.
잔여 stale 수(원 실행에서 폐기되는 행동 수)도 기록.

실행: hv2_oft python -u experiments/e3_t2_diag5b.py
출력: results/e3/t2_diag5b.json
"""
import json
import os
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.chained_env import ChainedEpisodeEnv, ChainedEpisodeSpec  # noqa: E402
from teacher.collector import load_teacher  # noqa: E402
from experiments.e3_t2_diag import teacher_chunk  # noqa: E402

OUT = os.path.join(HABIT2, "results", "e3", "t2_diag5b.json")


def rollout_chunk_break(env, spec, teacher, resize_size):
    obs = spec.realize(env)
    t, success, discarded = 0, False, None
    while t < env.max_steps:
        actions = teacher_chunk(teacher, resize_size, obs, env.language)
        for j, a in enumerate(actions):
            pre = env.stage()
            obs, _, done, _ = env.step(a.tolist())
            t += 1
            if pre == 1 and env.stage() == 2:
                discarded = len(actions) - (j + 1)  # stale tail 폐기 수
                break
            if done or t >= env.max_steps:
                break
        if env.check_success():
            success = True
            break
    return success, t, discarded, env.stage_steps


def main():
    import h5py

    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    d5 = json.load(open(os.path.join(HABIT2, "results", "e3", "t2_diag5.json")))
    flipped_uids = {r["uid"] for r in d5["per_episode"] if r["fresh_success"]}
    with h5py.File(os.path.join(HABIT2, "data", "e3", "chained_libero_object_task5.hdf5"), "r") as f:
        meta = json.loads(f["meta_json"][()])
    targets = [m for m in meta if m["uid"] in flipped_uids]
    print(f"[diag5b] 대상 {len(targets)}건 (diag5에서 fresh 성공으로 뒤집힌 stage-2 실패)", flush=True)

    set_seed_everywhere(7)
    teacher = load_teacher("libero_object")
    resize_size = get_image_resize_size(teacher[0])

    results, n_succ = [], 0
    for i, m in enumerate(targets):
        env = ChainedEpisodeEnv("libero_object", 5)
        spec = ChainedEpisodeSpec(m["suite"], m["task_id"], m["seed"], m["base_idx"], m["w"],
                                  m["noise_seed"], m["relocate_base_idx"], m["relocate_noise_seed"])
        assert spec.uid == m["uid"], "spec 재구성 uid 불일치"
        success, t, discarded, ss = rollout_chunk_break(env, spec, teacher, resize_size)
        n_succ += success
        results.append({"uid": m["uid"], "reloc_base": m["relocate_base_idx"],
                        "chunkbreak_success": success, "steps": t,
                        "stale_discarded": discarded, "stage_steps": ss})
        print(f"  [{i+1}/{len(targets)}] reloc={m['relocate_base_idx']} success={success} "
              f"stale폐기={discarded}", flush=True)
        env.close()

    n = len(targets)
    confirmed = n_succ >= round(0.8 * n)  # fresh 재현율(18/20=0.9)과 정합하면 확증
    report = {
        "hypothesis": "stale chunk tail (전환 후 잔여 행동 실행이 stage-2 시작 교란)",
        "n_targets": n, "n_success_with_chunk_break": n_succ,
        "confirmed": bool(confirmed),
        "per_episode": results,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[DIAG5B-{'CONFIRMED' if confirmed else 'NOT-CONFIRMED'}] {n_succ}/{n} "
          f"-> {os.path.relpath(OUT, HABIT2)}")


if __name__ == "__main__":
    main()
