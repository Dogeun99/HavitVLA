"""E4-R 역량 지도 — "작업 반경 내 기하 불변성" 직접 시험 (§5 등재, 판독 규칙 사전 고정).

설계: 대표 6 클러스터 × w ∈ {0.01, 0.02, 0.04, 0.06, 0.08} × 15 ep, habit(n=80) rollout only.
  - 물리 유효성(E0-6 settled 참조 기준: 낙하 z>5cm·수평 이탈 >10cm) 통과분만 계수, 폭별 탈락률 보고.
  - 각 에피소드의 게이트 점수·기각 판정을 **불개입 기록**.
  - 스펙 대역: novel 대역의 폭별 서브대역 (seed 30000+1000(w_idx+1)+j / noise 2e6+1000(w_idx+1)+j)
    — 기존 novel(j<20, 오프셋 0)과 분리, 6대역 disjoint 유지.
  - 실행기: execute_chunk_with_boundary 단일 경로.
산출: results/e4/e4r_competence_map.json + [E4R-MAP-PASS]
진행: [E4R] <cluster> w=<w> j/15 (heartbeat 파싱용)
실행: hv2_hab python -u experiments/e4r_competence_map.py
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

from envs.chained_env import execute_chunk_with_boundary  # noqa: E402
from envs.libero_env import (DUMMY_ACTION, SETTLE_STEPS, EpisodeSpec, InfraError,  # noqa: E402
                             LiberoEpisodeEnv, USABLE_W_MAX)
from gates.features import DinoFeatureExtractor, SharedPCA, prep_gate_rgb  # noqa: E402
from gates.two_stage import JurisdictionGate  # noqa: E402
from habits.policy import HabitPolicy  # noqa: E402

E4 = os.path.join(HABIT2, "results", "e4")
OUT = os.path.join(E4, "e4r_competence_map.json")
# 대표 6 클러스터: 스위트 균형 + object task0·task5 필수 + 형성 속도 다양성(N* 10~>80)
CLUSTERS = [("libero_object", 0), ("libero_object", 5), ("libero_goal", 0),
            ("libero_goal", 2), ("libero_spatial", 0), ("libero_10", 0)]
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}
W_GRID = [0.01, 0.02, 0.04, 0.06, 0.08]
N_PER_W = 15
CHUNK = 8
SEED_BASE, NOISE_BASE = 30_000, 2_000_000
BASES = list(range(40, 50))  # known과 base 정합 (E4 주 설계)
_ref_cache = {}


def settle_ref(env, key, state):
    """무섭동 settled 참조 (E0-6 정합, v2 검사기와 동일 의미론)."""
    if key in _ref_cache:
        return _ref_cache[key]
    env.begin_episode(90_000 + (hash(key) % 1000), state)
    sim = env._env.env.sim
    _ref_cache[key] = {adr: np.array(sim.data.qpos[adr:adr + 3]) for adr in env._free_adrs}
    return _ref_cache[key]


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
    import argparse
    import h5py

    ap = argparse.ArgumentParser()
    # RGB-only full rerun (2026-08-28): 경로만 파라미터화. 설계·판독 규칙은 불변.
    ap.add_argument("--ckpt-root", default=os.path.join(HABIT2, "checkpoints"),
                    help="<root>/<cluster>/act_n80.pt")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--frag-dir", default=os.path.join(E4, "e4r_frag"))
    args = ap.parse_args()

    ext = DinoFeatureExtractor()
    pca = SharedPCA.load(os.path.join(E4, "shared_pca_e4.joblib"))

    def embed1(frame):
        return pca.transform(ext.embed([frame]))[0]

    rep = {"habit_ckpt_root": None,   # main에서 채움 (modality 추적)
           "note": "E4-R 역량 지도 (§5 판독 규칙 사전 등재). habit(n=80) rollout only; "
                   "유효성 통과분만 계수(폭별 탈락률 병기); 게이트는 불개입 기록.",
           "w_grid": W_GRID, "n_per_w": N_PER_W, "clusters": {}}
    rep["habit_ckpt_root"] = os.path.relpath(args.ckpt_root, HABIT2)
    os.makedirs(args.frag_dir, exist_ok=True)
    for suite, task in CLUSTERS:
        cl = f"{suite}_task{task}"
        frag_p = os.path.join(args.frag_dir, f"{cl}.json")
        if os.path.exists(frag_p):  # 재개 가능 (경로 유일성 렌즈 — 덮어쓰기 금지)
            rep["clusters"][cl] = json.load(open(frag_p))
            print(f"[E4R-SKIP] {cl}: 기존 조각 재사용", flush=True)
            continue
        ddir = "e2" if (suite, task) in E2_REUSE else "e3"
        with h5py.File(os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5"), "r") as f:
            col = pca.transform(np.concatenate([ext.embed([f[f"episodes/{k}/agentview_rgb"][0]])
                                                for k in f["episodes"]]))
        gate = JurisdictionGate().fit(col)
        env = LiberoEpisodeEnv(suite, task)
        policy = HabitPolicy(os.path.join(args.ckpt_root, cl, "act_n80.pt"))
        entry = {"usable_w_max": USABLE_W_MAX[suite], "q": round(gate.q, 3), "by_w": {}}
        for wi, w in enumerate(W_GRID):
            for b in sorted(set(BASES[j % 10] for j in range(N_PER_W))):
                settle_ref(env, f"{cl}|{b}", env.init_states[b])
            n_valid = n_succ = n_reject_phys = n_gate_reject = n_infra = 0
            eps = []
            for j in range(N_PER_W):
                base = BASES[j % 10]
                seed = SEED_BASE + 1000 * (wi + 1) + j
                noise = NOISE_BASE + 1000 * (wi + 1) + j
                spec = EpisodeSpec(suite, task, seed, base, w, noise)  # uid 기록용
                try:
                    # 역량 경계 탐색 = usable_w_max 밖 폭도 생성 (명시적 예외, §5 등재).
                    # 물리 유효성은 아래 E0-6 검사로 걸러내고 탈락률을 보고한다.
                    state = env.perturbed_init_state(base, w, np.random.default_rng(noise),
                                                     allow_beyond_usable=True)
                    obs = env.begin_episode(seed, state)  # 1회 실현 (검사·rollout 공용)
                    ok, why = settled_validity(env, state, _ref_cache[f"{cl}|{base}"])
                    if not ok:
                        n_reject_phys += 1
                        eps.append({"uid": spec.uid, "valid": False, "why": why})
                        print(f"[E4R] {cl} w={w} {j+1}/{N_PER_W} invalid({why})", flush=True)
                        continue
                    feat = embed1(prep_gate_rgb(obs["agentview_image"]))
                    score = gate.score(feat)
                    rejected = bool(score > gate.q)
                    t, success = 0, False
                    while t < env.max_steps:
                        chunk = policy.act_chunk(obs)
                        obs, t, _n, _s = execute_chunk_with_boundary(
                            env, list(chunk[:CHUNK]), t, env.max_steps)
                        if env.check_success():
                            success = True
                            break
                except InfraError as e:
                    n_infra += 1
                    eps.append({"uid": spec.uid, "outcome": "infra_error", "error": str(e)})
                    continue
                n_valid += 1
                n_succ += int(success)
                n_gate_reject += int(rejected)
                eps.append({"uid": spec.uid, "valid": True, "success": bool(success),
                            "steps": t, "gate_score": round(float(score), 3),
                            "gate_rejected": rejected,
                            "w": w, "base_idx": base, "seed": seed, "noise_seed": noise,
                            "pca32": [round(float(x), 5) for x in feat]})
                print(f"[E4R] {cl} w={w} {j+1}/{N_PER_W} succ={success} gate_rej={rejected}",
                      flush=True)
            entry["by_w"][str(w)] = {
                "n_valid": n_valid, "n_reject_physical": n_reject_phys, "n_infra": n_infra,
                "physical_reject_rate": round(n_reject_phys / N_PER_W, 4),
                "habit_success_rate": round(n_succ / n_valid, 4) if n_valid else None,
                "gate_reject_rate": round(n_gate_reject / n_valid, 4) if n_valid else None,
                "episodes": eps,
            }
            print(f"[E4R-W] {cl} w={w}: 습관 성공률="
                  f"{entry['by_w'][str(w)]['habit_success_rate']} "
                  f"게이트 기각률={entry['by_w'][str(w)]['gate_reject_rate']} "
                  f"물리탈락={n_reject_phys}/{N_PER_W}", flush=True)
        # 역량 경계 w* = 성공률이 0.8 아래로 내려가는 첫 폭 (판독 규칙 1)
        wstar = None
        for w in W_GRID:
            r = entry["by_w"][str(w)]["habit_success_rate"]
            if r is not None and r < 0.8:
                wstar = w
                break
        entry["w_star"] = wstar if wstar is not None else f">{W_GRID[-1]}"
        entry["w_star_exceeds_usable"] = (wstar is None or wstar > USABLE_W_MAX[suite])
        rep["clusters"][cl] = entry
        with open(frag_p, "w") as f:  # 클러스터 단위 중간 저장 (장애 시 재개)
            json.dump(entry, f, ensure_ascii=False)
        env.close()
        del policy
        import torch

        torch.cuda.empty_cache()
        print(f"[E4R-CL] {cl}: w*={entry['w_star']} (usable_w_max={USABLE_W_MAX[suite]})", flush=True)

    ws = [v["w_star"] for v in rep["clusters"].values()]
    rep["summary"] = {
        "w_star_by_cluster": {k: v["w_star"] for k, v in rep["clusters"].items()},
        "n_exceeding_usable_w_max": sum(1 for v in rep["clusters"].values()
                                        if v["w_star_exceeds_usable"]),
        "success_by_w": {str(w): round(float(np.mean(
            [v["by_w"][str(w)]["habit_success_rate"] for v in rep["clusters"].values()
             if v["by_w"][str(w)]["habit_success_rate"] is not None])), 4) for w in W_GRID},
        "gate_reject_by_w": {str(w): round(float(np.mean(
            [v["by_w"][str(w)]["gate_reject_rate"] for v in rep["clusters"].values()
             if v["by_w"][str(w)]["gate_reject_rate"] is not None])), 4) for w in W_GRID},
        "physical_reject_by_w": {str(w): round(float(np.mean(
            [v["by_w"][str(w)]["physical_reject_rate"] for v in rep["clusters"].values()])), 4)
            for w in W_GRID},
    }
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    s = rep["summary"]
    print(f"[E4R-MAP-PASS] 성공률/w={s['success_by_w']} | 게이트기각/w={s['gate_reject_by_w']} | "
          f"w*={s['w_star_by_cluster']} | usable 초과 {s['n_exceeding_usable_w_max']}/6")


if __name__ == "__main__":
    main()
