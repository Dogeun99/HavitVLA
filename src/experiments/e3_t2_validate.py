"""C-T2 래퍼 검증 (수집 전 필수 — E0-6급).

--mechanics : 모델 불필요 (sim만) — §4e 개정(옵션 A: 전환 = 전체 상태 재설정) 반영판
  ① 전환 유효성: 강제 전환 후 물체가 목표 ±10cm, z-낙하 없음, **로봇 qpos = 재배치 init의
     홈 포즈로 재설정** (구 기준 "로봇 불변"은 옵션 A로 폐기 — §5 이력 2026-08-15)
  ② 결정성: 동일 spec 2회 → 강제 전환 후 sim 상태 해시 일치
--teacher : OFT 필요 (GPU 16GB — E3 수집과 동시 실행 금지)
  ③ teacher 2연쇄 스모크 10 ep: stage별 성공 분해. stage2 수행 불가 시
     설계서 §3 대체(Long 길이 층화) 발동 보고.

실행: conda run -n hv2_oft python -u experiments/e3_t2_validate.py --mechanics
      (teacher는 E3 phase 1 종료 후) ... --teacher
"""
import argparse
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

OUT = os.path.join(HABIT2, "results", "e3", "e3_t2_validation.json")
DUMMY = [0, 0, 0, 0, 0, 0, -1]


def state_hash(env):
    import hashlib

    sim = env._env.env.sim
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sim.data.qpos).tobytes())
    h.update(np.ascontiguousarray(sim.data.qvel).tobytes())
    return h.hexdigest()[:16]


def object_positions(env):
    sim = env._env.env.sim
    out = {}
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] == 0:
            adr = sim.model.jnt_qposadr[j]
            out[sim.model.joint_id2name(j)] = sim.data.qpos[adr : adr + 3].copy()
    return out


def robot_qpos(env):
    sim = env._env.env.sim
    free = set()
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] == 0:
            a = sim.model.jnt_qposadr[j]
            free.update(range(a, a + 7))
    return np.array([sim.data.qpos[i] for i in range(sim.model.nq) if i not in free])


def mechanics():
    from envs.chained_env import ChainedEpisodeEnv, chained_collection_specs

    env = ChainedEpisodeEnv("libero_object", 0, resolution=128)
    spec = chained_collection_specs("libero_object", 0, 1)[0]
    rep = {}

    def run_to_forced_relocation():
        spec.realize(env)
        for _ in range(30):
            env.step(DUMMY)
        env._stage = 2  # 강제 전환 경로로 전환만 검증
        env._relocate_objects()

    def robot_qpos_from_state(state):
        """상태 벡터의 로봇(비-free-joint) qpos 부분 — 전환 후 기대 홈 포즈."""
        sim = env._env.env.sim
        off = env._time_offset
        free = set()
        for j in range(sim.model.njnt):
            if sim.model.jnt_type[j] == 0:
                a = sim.model.jnt_qposadr[j]
                free.update(range(a, a + 7))
        return np.array([state[off + i] for i in range(sim.model.nq) if i not in free])

    # ① 전환 유효성 (옵션 A: 물체 = 재배치 포즈, 로봇 = 홈 재설정)
    run_to_forced_relocation()
    pos = object_positions(env)
    sim = env._env.env.sim
    targets = {}
    off = env._time_offset
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] == 0:
            adr = sim.model.jnt_qposadr[j]
            targets[sim.model.joint_id2name(j)] = env._relocate_state[off + adr : off + adr + 3]
    max_dev = max(float(np.linalg.norm(pos[k][:2] - targets[k][:2])) for k in pos)
    z_ok = all(float(pos[k][2]) > float(targets[k][2]) - 0.05 for k in pos)
    rq_after = robot_qpos(env)
    rq_home = robot_qpos_from_state(env._relocate_state)
    robot_home_ok = bool(np.allclose(rq_home, rq_after, atol=0.05))  # settle 중 미세 이동 허용
    rep["relocation"] = {
        "max_xy_deviation_m": round(max_dev, 4),
        "xy_ok": bool(max_dev < 0.10),
        "z_ok": bool(z_ok),
        "robot_reset_to_home": robot_home_ok,
        "max_robot_qpos_dev": round(float(np.max(np.abs(rq_home - rq_after))), 4),
    }

    # ② 결정성
    h1 = state_hash(env)
    run_to_forced_relocation()
    h2 = state_hash(env)
    rep["determinism"] = {"hash1": h1, "hash2": h2, "match": h1 == h2}

    ok = rep["relocation"]["xy_ok"] and rep["relocation"]["z_ok"] and rep["relocation"]["robot_reset_to_home"] and rep["determinism"]["match"]
    rep["status"] = "PASS" if ok else "FAIL"
    env.close()
    return rep


def teacher_smoke(n=10):
    from envs.chained_env import ChainedEpisodeEnv, chained_collection_specs
    from envs.libero_env import InfraError
    from teacher.collector import load_teacher, rollout_episode
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    set_seed_everywhere(7)
    teacher = load_teacher("libero_object")
    resize = get_image_resize_size(teacher[0])
    env = ChainedEpisodeEnv("libero_object", 0)
    specs = chained_collection_specs("libero_object", 0, n)
    results = []
    for spec in specs:
        try:
            success, frames, actions, steps, _stale = rollout_episode(spec, env, teacher, resize)
            results.append({"uid": spec.uid, "success": success, "steps": steps,
                            "stage_steps": env.stage_steps, "reached_stage2": env.stage() == 2})
        except InfraError as e:
            results.append({"uid": spec.uid, "infra_error": str(e)})
        r = results[-1]
        print(f"  ep: success={r.get('success')} stages={r.get('stage_steps')}", flush=True)
    n_s2 = sum(1 for r in results if r.get("reached_stage2"))
    n_full = sum(1 for r in results if r.get("success"))
    rep = {
        "n": n, "reached_stage2": n_s2, "full_success": n_full,
        "per_episode": results,
        "status": "PASS" if n_full >= 7 else ("STAGE2_WEAK" if n_s2 >= 7 else "FAIL"),  # 게이트와 정렬(≥7/10)
        "note": "FAIL/STAGE2_WEAK 시 설계서 §3 대체(Long 길이 층화) 발동 — 연구원 보고",
    }
    env.close()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanics", action="store_true")
    ap.add_argument("--teacher", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    report = {}
    if os.path.exists(OUT):
        report = json.load(open(OUT))
    if args.mechanics:
        report["mechanics"] = mechanics()
        print(json.dumps(report["mechanics"], indent=2))
    if args.teacher:
        report["teacher_smoke"] = teacher_smoke()
        print(json.dumps({k: v for k, v in report["teacher_smoke"].items() if k != "per_episode"}, indent=2))
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    parts = [v.get("status") for k, v in report.items() if isinstance(v, dict)]
    status = "PASS" if parts and all(s == "PASS" for s in parts) else "PENDING/FAIL:" + ",".join(str(s) for s in parts)
    print(f"[E3-T2-VALIDATE] status={status} json=results/e3/e3_t2_validation.json")


if __name__ == "__main__":
    main()
