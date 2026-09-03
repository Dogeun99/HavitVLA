"""E0-6: 초기상태 변이 폭 파라미터화 가능 여부 확인 (docs/E0_INSTRUCTIONS.md E0-6).

검증 워크플로우 발견 반영판 (v2) + 가용 폭 정밀화 (v3):
  - 범위: 4 스위트 각 task 0 (단일 태스크 과대 일반화 방지 — spatial만이 아님)
  - **가용 변이 폭 usable_w_max**: 유효율 ≥ 7/8을 유지하는 최대 w. 실측: 혼잡 씬(Long task0,
    free joint 8개)은 w=0.04에서 유효율 3/8로 붕괴 → 가용 폭은 씬 밀도 종속. go 기준 =
    스위트별 usable_w_max ≥ 0.02 (스트림 생성기는 클러스터별 w ≤ usable_w_max 준수).
  - 단조성: 가용 sub-grid에서 비퇴화(w_max 분산 > 1e-8, w=0 대비 증가) + 단조 증가
  - 유효성: (a) 낙하 없음(z − 5cm) (b) 수평 이탈 없음(정착 후 목표 섭동 위치에서 10cm 이내)
    를 함께 검사. 관통·전도·BDDL 초기 술어 재검증은 **미구현 한계**로 JSON에 명시
    (E4 novel 생성기 구현 시 승격 예정 — 문제 축소가 아니라 한계의 명시적 기록).
  - 무효 표본은 분산 계산에서 제외 (두 검정의 상호 오염 방지).

두 경로:
  1) 섭동 경로: 공식 init state 벡터의 물체 free-joint (x,y)에 uniform(-w, w) 노이즈.
  2) 재샘플링 경로: env.seed(s) + reset()의 BDDL placement 재샘플 (seed 재현·상이 확인).

재현 프로토콜(실측 확정, log.md 2026-08-15): 매 trial `env.seed(seed) → reset() → set_init_state → settle`.
주의: 이 프로토콜은 공식 eval(get_libero_env의 생성 직후 1회 seed)과 다르다 — 공식 eval은
run 수준 재현(전체 시퀀스 동일)이고, 본 프로토콜은 episode 수준 재현(개별 에피소드 독립 재생)이다.
teacher S_V 측정(E1)은 공식 프로토콜을, 우리 스트림 생성기·held-out 평가(E2+)는 본 프로토콜을 쓴다.

실행: conda run -n hv2_oft python -u experiments/e0_variation_check.py
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "e0", "e0_6_variation.json")

# LIBERO_CONFIG_PATH 미지정 실행이 공용 ~/.libero를 오염시키는 사고 방지 (ISSUE-13 가드)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(ROOT, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")

SETTLE_STEPS = 10
DUMMY = [0, 0, 0, 0, 0, 0, -1]
W_GRID = [0.0, 0.01, 0.02, 0.04]
N_PER_W = 8
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def obs_hash(obs):
    h = hashlib.sha256()
    for k in sorted(obs.keys()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(obs[k]).tobytes())
    return h.hexdigest()[:16]


def state_hash(env):
    sim = env.env.sim
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sim.data.qpos).tobytes())
    h.update(np.ascontiguousarray(sim.data.qvel).tobytes())
    return h.hexdigest()[:16]


def object_positions(env):
    """free joint를 가진 물체들의 (x,y,z) — env.sim에서 직접 읽음."""
    sim = env.env.sim
    out = {}
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] == 0:  # mjJNT_FREE
            adr = sim.model.jnt_qposadr[j]
            name = sim.model.joint_id2name(j)
            out[name] = sim.data.qpos[adr : adr + 3].copy()
    return out


def settle(env):
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY)
    return obs


def perturb_state(state, free_qpos_adrs, w, rng, time_offset):
    s = state.copy()
    deltas = {}
    for adr in free_qpos_adrs:
        d = rng.uniform(-w, w, size=2)
        s[time_offset + adr : time_offset + adr + 2] += d
        deltas[adr] = d
    return s, deltas


def check_suite(suite_name, benchmark, get_libero_path, OffScreenRenderEnv):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(0)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)
    env.reset()

    sim = env.env.sim
    nq, nv = sim.model.nq, sim.model.nv
    dim = init_states.shape[1]
    time_offset = dim - nq - nv  # robosuite SimState flatten = [time?] + qpos + qvel
    assert time_offset in (0, 1), f"unexpected state layout: dim={dim}, nq={nq}, nv={nv}"

    # 주의: env.reset()이 MjSim을 재생성하므로 model 상수(조인트 주소)는 reset 전에 뽑아 둔다
    free_adrs = [
        sim.model.jnt_qposadr[j] for j in range(sim.model.njnt) if sim.model.jnt_type[j] == 0
    ]
    adr_by_name = {
        sim.model.joint_id2name(j): sim.model.jnt_qposadr[j]
        for j in range(sim.model.njnt)
        if sim.model.jnt_type[j] == 0
    }

    rep = {
        "task": task.name,
        "state_dim": dim,
        "n_free_joints": len(free_adrs),
        "perturbation": {"w_grid": W_GRID, "n_per_w": N_PER_W, "results": {}},
        "resampling": {},
        "validity_checks": ["z_drop_5cm", "xy_settle_within_10cm_of_perturbed_target"],
        "validity_not_checked": ["contact_penetration", "tip_over", "bddl_init_predicates"],
    }

    def run_trial(state, seed=0):
        env.seed(seed)
        env.reset()
        env.set_init_state(state)
        return settle(env)

    # 기준(무섭동) — 분산은 단일 base state(init_states[0])에서 섭동만으로 유도
    run_trial(init_states[0])
    base_pos = object_positions(env)
    base_z = {k: float(v[2]) for k, v in base_pos.items()}
    base_xy = {k: v[:2].copy() for k, v in base_pos.items()}

    variances, valid_rates, per_w = [], [], {}
    for w in W_GRID:
        finals_valid, n_valid = [], 0
        for i in range(N_PER_W):
            rng = np.random.default_rng(1000 + i)  # w와 무관하게 동일 노이즈 방향 (paired)
            s, deltas = perturb_state(init_states[0], free_adrs, w, rng, time_offset)
            run_trial(s)
            pos = object_positions(env)
            # 유효성 (a) 낙하 없음 (b) 정착 위치가 섭동 목표(base_xy + delta)에서 10cm 이내
            ok_z = all(float(p[2]) > base_z[k] - 0.05 for k, p in pos.items())
            ok_xy = all(
                np.linalg.norm(p[:2] - (base_xy[k] + deltas[adr_by_name[k]])) < 0.10
                for k, p in pos.items()
            )
            if ok_z and ok_xy:
                n_valid += 1
                finals_valid.append(np.concatenate([p[:2] for p in pos.values()]))
        var = float(np.stack(finals_valid).var(axis=0).mean()) if len(finals_valid) >= 2 else None
        variances.append(var)
        valid_rates.append(n_valid / N_PER_W)
        per_w[str(w)] = {"mean_xy_variance": var, "valid_rate": n_valid / N_PER_W}

    # 재현성: 동일 seed·동일 w 2회 → sim 상태 해시 일치 (obs 해시 병기)
    hashes, obs_hashes = [], []
    for _ in range(2):
        rng = np.random.default_rng(42)
        s, _ = perturb_state(init_states[0], free_adrs, 0.02, rng, time_offset)
        obs = run_trial(s)
        hashes.append(state_hash(env))
        obs_hashes.append(obs_hash(obs))
    repro = hashes[0] == hashes[1]

    # 가용 변이 폭: 유효율 ≥ 7/8을 유지하는 최대 w (씬 밀도 종속 — Long 혼잡 씬은 w=0.04에서 붕괴 실측).
    # 판정은 가용 sub-grid에 대해서만: 비퇴화 + 단조. 스트림 생성기는 클러스터별 w ≤ usable_w_max 준수.
    usable_idx = [i for i, v in enumerate(valid_rates) if v >= 7 / 8]
    # 가용 격자는 0부터 연속이어야 의미 있음
    k = 0
    while k < len(W_GRID) and k in usable_idx:
        k += 1
    usable = list(range(k))  # [0..k-1]
    usable_w_max = W_GRID[usable[-1]] if usable else None
    sub_vars = [variances[i] for i in usable]
    nondegenerate = (
        len(sub_vars) >= 2
        and all(v is not None for v in sub_vars)
        and sub_vars[-1] > 1e-8
        and sub_vars[-1] > sub_vars[0] + 1e-8
    )
    monotone = nondegenerate and all(
        sub_vars[i] <= sub_vars[i + 1] + 1e-12 for i in range(len(sub_vars) - 1)
    )
    # go 기준: 가용 폭이 실용 최소치(0.02 m) 이상
    valid_ok = usable_w_max is not None and usable_w_max >= 0.02

    rep["perturbation"].update(
        {
            "results": per_w,
            "usable_w_max": usable_w_max,
            "variance_monotone": bool(monotone),
            "variance_nondegenerate": bool(nondegenerate),
            "valid_ok": bool(valid_ok),
            "repro_state_hashes": hashes,
            "repro_obs_hashes": obs_hashes,
            "obs_reproducible_info": bool(obs_hashes[0] == obs_hashes[1]),
            "reproducible": bool(repro),
        }
    )

    # ---- 경로 2: 재샘플링 (BDDL placement) ----
    def reset_positions(seed):
        env.seed(seed)
        env.reset()
        settle(env)
        return np.concatenate([p[:2] for p in object_positions(env).values()])

    p0a, p0b, p1 = reset_positions(0), reset_positions(0), reset_positions(1)
    resample_repro = bool(np.allclose(p0a, p0b, atol=1e-10))
    resample_differs = bool(np.abs(p0a - p1).max() > 1e-4)
    rep["resampling"] = {
        "same_seed_reproducible": resample_repro,
        "diff_seed_differs": resample_differs,
        "seed0_vs_seed1_max_delta_m": float(np.abs(p0a - p1).max()),
    }

    ok = monotone and valid_ok and repro and resample_repro and resample_differs
    rep["status"] = "PASS" if ok else "FAIL"
    env.close()
    return rep


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    report = {"suites": {}, "status": "FAIL"}
    for s in SUITES:
        print(f"--- checking {s} ---", flush=True)
        report["suites"][s] = check_suite(s, benchmark, get_libero_path, OffScreenRenderEnv)
        print(f"    {s}: {report['suites'][s]['status']}", flush=True)

    ok = all(r["status"] == "PASS" for r in report["suites"].values())
    report["status"] = "PASS" if ok else "FAIL"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v["status"] for k, v in report["suites"].items()}, indent=2))
    print(f"[E0-PASS] item=E0-6 status={report['status']} json=results/e0/e0_6_variation.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
