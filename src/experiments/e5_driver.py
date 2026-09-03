"""E5 온라인 스트림 드라이버 (설계서 v0.3 구현 승인본, hv2_oft 단일 프로세스).

발화 = **성숙도 단독**(A_mat) — 관할은 그림자 로깅(불개입, §4h REDUCE 반영).
lazy 재학습 {20,80} · probe P=20 · R_max=2 · 부적격 · 3장부 시간 회계 · CF 큐(종료 후 배치).
실행기는 `execute_chunk_with_boundary` 단일 경로(렌즈 3), 에피소드 실현은 E0-6 3단(렌즈 1·2).

실행:
  본실행: hv2_oft python -u experiments/e5_driver.py --seed-idx 0 --n 4000
  스모크: hv2_oft python -u experiments/e5_driver.py --smoke   (|B_k|≥10·P=10·부적격 강제 주입)
마커: [E5-EP] 진행 / [E5-RETRAIN] / [E5-PROBE] / [E5-DONE] / [E5-FAIL]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
# release: hv2_hab interpreter for the retrain subprocess (override with HV2_HAB_PY)
PY_HAB = os.environ.get("HV2_HAB_PY", os.path.expanduser("~/miniconda3/envs/hv2_hab/bin/python"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.chained_env import execute_chunk_with_boundary  # noqa: E402
from envs.libero_env import InfraError, LiberoEpisodeEnv  # noqa: E402
from envs.stream import (assert_six_bands_disjoint, e5_stream_specs,  # noqa: E402
                         probe_specs)
from gates.features import DinoFeatureExtractor, SharedPCA, prep_gate_rgb  # noqa: E402
from gates.two_stage import ACIRiskController, JurisdictionGate, MaturityGate  # noqa: E402

CHUNK = 8
GRID_FULL, PROBE_FULL = (20, 80), 20
# 스모크 |B_k| 트리거 = 10: n=3·300스텝으로는 유효 정책이 나오지 않아(E3 실측: goal_task2는
# n=10에서 ŝ=0.95가 최소 유효 지점) 성숙 전이(관문 조건 3)가 원천 불가였다.
GRID_SMOKE = (10,)
# 스모크 P=10: P=5는 5/5 전승해도 Pr(s≥0.8|Beta(6,1))=0.738 < 0.9라 **성숙 전이가 원천 불가**
# → 관문 조건 3(I→M)·4(CF 큐)를 검증할 수 없다. P=10이면 10/10에서 0.914 ≥ 0.9로 가능.
PROBE_SMOKE = 10
# 부적격(X) 경로 검증용: 이 클러스터의 **모든 probe 라운드**에 실패를 주입한다
# (첫 라운드만 주입하면 2라운드 미달이 보장되지 않아 X 전이가 관측되지 않을 수 있음)
SMOKE_FORCE_FAIL_CLUSTER = "libero_object_task0"

# --- B-2 배치 등가 학습 스텝 (연구원 최종 판정 2026-08-17, §5 HP 개정) ---
# E3 배치는 warm-start 체인(10→20→40→80)이라 각 체크포인트가 그 지점까지의 **누적** 스텝을
# 본다. E5 재학습은 scratch이므로 같은 총량을 단독 지정해야 학습량이 배치와 정합한다.
# 값은 HP["steps_per_n"]에서 프로그래밍 산출 — 수동 입력 금지(사전등록 §6).
def _batch_equiv_steps():
    from habits.train import HP
    sp, g = HP["steps_per_n"], [10, 20, 40, 80]
    return {n: sum(sp[k] for k in g[:g.index(n) + 1]) for n in g}


BATCH_EQUIV_STEPS = _batch_equiv_steps()      # {10: 4000, 20: 10000, 40: 18000, 80: 28000}
SMOKE_STEPS = 2500                            # 스모크 전용 축소 (본실행 상수 불변)


class ClusterState:
    """클러스터별 lifecycle 상태 + 장부 (설계서 §1.1)."""

    def __init__(self, cluster, suite, task):
        self.cluster, self.suite, self.task = cluster, suite, task
        self.maturity = MaturityGate()          # A_mat = probe+fire만 (§4h)
        self.aci = ACIRiskController()
        self.jur = None                          # 그림자 관할 (수집 프레임으로 fit)
        self.bc_pool = []                        # teacher 성공 궤적 uid (|B_k| 트리거)
        self.version = 0                         # 정책 버전
        self.ckpt = None
        self.next_grid_idx = 0                   # lazy {20, 80} 진행 위치
        self.ever_mature = False                 # 재성숙(rematuration) 판정용 — 로깅 전용

    def state(self):
        if self.maturity.ineligible:
            return "X"
        if self.version == 0 and not self.bc_pool:
            return "U"
        return "M" if self.maturity.accepts() else "I"


def assert_retrain_contract(ckpt_path, n, st, h5_path, smoke=False):
    """재학습 계약 3단언 (연구원 판정 2026-08-17 §2). 위반 시 RuntimeError로 즉시 정지.

    (a) 정규화 공간 횡단 0 — scratch 학습이므로 체크포인트의 stats는 **자기 학습 데이터**
        (episodes[:n])에서 산출된 값과 일치해야 한다. warm-start 흔적이 있으면 위반.
        ※ 판정문 원안은 "재학습 간 l2 상대차 == 0"이나 이는 A안(트리거 {80} 단일, 항상
          같은 80개 풀) 전제다. B-2에서는 n=20이 20개 풀, n=80이 80개 풀을 쓰므로 재학습
          간 stats가 다른 것이 정상이며, 원 취지인 '횡단 차단'은 아래 형태로 검증한다.
    (b) 학습 스텝 == 지정값 (본실행 = 배치 등가, 스모크 = 축소값)
    (c) |B_k| == 참조 HDF5의 성공 에피소드 수
    """
    import h5py
    import numpy as np
    import torch

    from habits.dataset import compute_stats, load_cluster

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    expect_steps = SMOKE_STEPS if smoke else BATCH_EQUIV_STEPS[n]

    # (c) 먼저 — (a)의 재산출이 같은 풀을 봐야 하므로 데이터 정합이 선행 조건이다.
    with h5py.File(h5_path, "r") as f:
        meta = json.loads(f["meta_json"][()])
        n_h5_succ = sum(1 for m in meta if m["outcome"] == "success")
        n_h5_group = len(f["episodes"])
    if not (len(st.bc_pool) == n_h5_succ == n_h5_group):
        raise RuntimeError(f"[GATE-C-FAIL] {st.cluster}: |B_k|={len(st.bc_pool)} vs "
                           f"HDF5 meta 성공={n_h5_succ} vs episodes={n_h5_group}")

    # (b)
    if sd["steps"] != expect_steps:
        raise RuntimeError(f"[GATE-B-FAIL] {st.cluster} n={n}: steps={sd['steps']} "
                           f"≠ 지정 {expect_steps}")

    # (a)
    recomputed = compute_stats(load_cluster(h5_path)[:n])
    for k, v in sd["stats"].items():
        a, b = np.asarray(v, float).ravel(), np.asarray(recomputed[k], float).ravel()
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))
        if rel > 1e-6:
            raise RuntimeError(f"[GATE-A-FAIL] {st.cluster} n={n}: stats['{k}']가 자기 학습 "
                               f"데이터(episodes[:{n}])에서 산출된 값과 불일치 (l2 상대차 {rel:.3e}) "
                               f"— 정규화 공간 횡단 의심")
    print(f"[GATE-PASS] {st.cluster} n={n} | steps={sd['steps']} | |B_k|={len(st.bc_pool)} "
          f"| stats 자기풀 일치 | scratch", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    # --- RGB-only full rerun (2026-08-28). 아래 3개는 **순수 가산**이며 기본값은 기존 동작을
    # 그대로 재현한다. depth 제거 외 실험 파라미터는 하나도 바뀌지 않는다 (§1 ABSOLUTE FREEZE).
    ap.add_argument("--no-depth", action="store_true",
                    help="RGB-only — 재학습 subprocess에 --no-depth 전파")
    ap.add_argument("--out-root", default=None, help="결과 루트 override (기본 results/e5)")
    ap.add_argument("--ck-root", default=None, help="체크포인트 루트 override")
    ap.add_argument("--data-root", default=None, help="스트림 궤적 루트 override")
    args = ap.parse_args()
    grid = GRID_SMOKE if args.smoke else GRID_FULL
    n_probe = PROBE_SMOKE if args.smoke else PROBE_FULL
    outdir = (args.out_root if args.out_root else
              os.path.join(HABIT2, "results", "e5", "smoke" if args.smoke else ""))
    ckroot = (os.path.join(args.ck_root, "smoke" if args.smoke else f"e5_s{args.seed_idx}")
              if args.ck_root else
              os.path.join(HABIT2, "checkpoints",
                           "e5_smoke" if args.smoke else f"e5_s{args.seed_idx}"))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ckroot, exist_ok=True)
    logp = os.path.join(outdir, f"stream_{args.seed_idx}.jsonl")
    for _p in (logp, os.path.join(outdir, f"lifecycle_events_{args.seed_idx}.jsonl")):
        if os.path.exists(_p):
            raise SystemExit(f"[E5-FAIL] 출력 경로 이미 존재 (덮어쓰기 금지, 렌즈 4): {_p}")

    bands = assert_six_bands_disjoint(args.seed_idx)      # 렌즈 6 — 실패 시 기동 거부
    print(f"[E5-INIT] 6대역 disjoint 통과 {bands}", flush=True)

    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import get_vla_action
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    from habits.policy import HabitPolicy
    from teacher.collector import load_teacher, teacher_observation, store_frame

    set_seed_everywhere(7)
    # 스모크 전용: 3 클러스터로 좁혀 |B_k| 트리거가 실제로 발화하게 한다 (본실행 25종 불변)
    # 2종으로 좁혀 클러스터당 ~22 ep 확보 (재학습→probe→M→발화→CF까지 여유)
    smoke_clusters = [("libero_object", 0), ("libero_goal", 2)]
    specs = e5_stream_specs(args.seed_idx, args.n,
                            clusters=smoke_clusters if args.smoke else None)
    ext = DinoFeatureExtractor()
    pca = SharedPCA.load(os.path.join(HABIT2, "results", "e4", "shared_pca_e4.joblib"))

    # 클러스터 상태 + 그림자 관할 fit (수집 성공 프레임 — §4c 규율)
    import h5py

    states, envs = {}, {}
    for spec, cl, _ in specs:
        if cl in states:
            continue
        st = ClusterState(cl, spec.suite_name, spec.task_id)
        ddir = "e2" if cl in ("libero_object_task0", "libero_object_task5") else "e3"
        h5 = os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5")
        if os.path.exists(h5):
            with h5py.File(h5, "r") as f:
                frames = [f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]
            feats = pca.transform(np.concatenate([ext.embed(frames[i:i + 64])
                                                  for i in range(0, len(frames), 64)]))
            st.jur = JurisdictionGate().fit(feats)
        states[cl] = st
    print(f"[E5-INIT] 클러스터 {len(states)}종 (그림자 관할 fit "
          f"{sum(1 for s in states.values() if s.jur)}종)", flush=True)

    teachers, resize = {}, {}
    stream_data_dir = (
        os.path.join(args.data_root, "smoke" if args.smoke else f"e5_s{args.seed_idx}")
        if args.data_root else
        os.path.join(HABIT2, "data", "e5_smoke" if args.smoke else f"e5_s{args.seed_idx}"))
    os.makedirs(stream_data_dir, exist_ok=True)
    stream_meta = defaultdict(list)
    ledger = {"operational_s": 0.0, "formation_s": 0.0, "formation_episodes": 0}
    cf_queue = []
    t_start = time.time()
    logf = open(logp, "a")

    def teacher_of(suite):
        if suite not in teachers:
            for s in list(teachers):            # VRAM: 스위트 전환 시 이전 모델 해제
                del teachers[s]
            import torch

            torch.cuda.empty_cache()
            teachers[suite] = load_teacher(suite)
            resize[suite] = get_image_resize_size(teachers[suite][0])
        return teachers[suite], resize[suite]

    def env_of(suite, task):
        key = (suite, task)
        if key not in envs:
            if len(envs) > 6:                    # env 캐시 상한
                k, e = envs.popitem()
                e.close()
            envs[key] = LiberoEpisodeEnv(suite, task)
        return envs[key]

    def run_teacher(env, obs):
        """스트림 teacher rollout. **궤적(프레임·행동)을 반환해 BC 풀에 적재한다** —
        E5의 BC 풀은 배치 수집분이 아니라 **스트림에서 축적된 teacher 성공 궤적**이어야 한다
        (§4h·설계서 §4 T1). 저장 규격은 collector와 동일(schema v2 호환)."""
        (cfg, model, ah, pp, nap, proc), rs = teacher_of(env.suite_name)
        t, success, frames, actions = 0, False, [], []
        n_calls = 0                                   # §8 VLA_calls — chunk 질의 횟수
        while t < env.max_steps:
            t_obs = teacher_observation(obs, rs)
            frames.append(store_frame(obs))
            chunk = get_vla_action(cfg, model, proc, t_obs, env.language, action_head=ah,
                                   proprio_projector=pp, noisy_action_projector=nap)
            n_calls += 1
            acts = [process_action(np.asarray(a, np.float32), cfg.model_family)
                    for a in chunk[:CHUNK]]
            obs, t, n_exec, _s = execute_chunk_with_boundary(env, acts, t, env.max_steps)
            actions.append(np.stack([np.asarray(a, np.float32) for a in acts[:max(n_exec, 1)]]))
            if env.check_success():
                success = True
                break
        return success, t, frames, actions, n_calls

    def store_stream_trajectory(cluster, spec, frames, actions, outcome, steps):
        """스트림 궤적을 클러스터별 HDF5에 증분 저장 (성공만 BC 풀 — 이중 장부 §2.5).
        meta_json은 매번 갱신 (load_cluster가 성공 uid 순서를 여기서 읽는다)."""
        from teacher.collector import write_episode, write_meta

        p = os.path.join(stream_data_dir, f"{cluster}.hdf5")
        stream_meta[cluster].append({**spec.to_dict(), "outcome": outcome, "steps": steps})
        with h5py.File(p, "a") as f:
            if "schema" not in f.attrs:
                f.attrs["schema"] = "habitvla2-e5-stream-v1"
            if outcome == "success":
                write_episode(f, spec.uid, frames, actions)
            write_meta(f, stream_meta[cluster])

    def run_habit(env, obs, ckpt):
        policy = HabitPolicy(ckpt)
        t, success = 0, False
        n_calls = 0                                   # §8 habit_calls — chunk 질의 횟수
        while t < env.max_steps:
            obs, t, _n, _s = execute_chunk_with_boundary(
                env, list(policy.act_chunk(obs)[:CHUNK]), t, env.max_steps)
            n_calls += 1
            if env.check_success():
                success = True
                break
        del policy
        return success, t, n_calls

    def retrain_and_probe(st, force_fail=False):
        """일시정지 → 재학습 → probe → 성숙 판정 (설계서 §4). 형성 장부에 계상."""
        import subprocess

        t0 = time.time()
        n = grid[st.next_grid_idx]
        st.version += 1
        # BC 풀 = **스트림 축적 teacher 성공 궤적** (배치 수집분 사용 금지 — §4h·§4 T1)
        # B-2 (연구원 최종 판정 2026-08-17): scratch 학습 + 배치 등가 스텝.
        #  - **scratch**: warm-start를 쓰지 않는다. lazy 재학습은 재학습마다 정규화 통계를
        #    자기 풀(episodes[:n])에서 산출하므로, warm-start를 유지하면 가중치가 정규화
        #    공간을 가로지른다(seed 0 무효화 사유). scratch면 각 체크포인트가 자기 공간에서
        #    완결되고 `HabitPolicy`도 체크포인트의 stats를 함께 로드하므로 추론까지 정합적이다.
        #  - **배치 등가 스텝**: E3는 warm-start 체인이라 n=80 체크포인트가 28,000 스텝을 본다.
        #    scratch는 단독 학습이므로 같은 총량을 명시 지정해야 배치와 학습량이 정합한다.
        cmd = [PY_HAB, "-u",
               os.path.join(HABIT2, "habits", "train.py"),
               "--h5", os.path.join(stream_data_dir, f"{st.cluster}.hdf5"),
               "--cluster", st.cluster, "--n-grid", str(n), "--out", ckroot,
               "--no-warm-start", "--steps", str(SMOKE_STEPS if args.smoke
                                                 else BATCH_EQUIV_STEPS[n])]
        if args.no_depth:
            cmd.append("--no-depth")     # RGB-only rerun — modality만 변경
        t_tr0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HABIT2)
        t_train = time.time() - t_tr0                 # §13 재학습 학습분 (probe 제외)
        ckpt = os.path.join(ckroot, st.cluster, f"act_n{n}.pt")
        if r.returncode != 0 or not os.path.exists(ckpt):
            raise RuntimeError(f"재학습 실패 {st.cluster}: {r.stderr[-300:]}")
        # --- 재실행 관문 3단언 (연구원 판정 2026-08-17). 스모크는 축소 스텝이라 본실행 값을
        # 검증할 수 없으므로 **런타임에서도** 단언한다 — 직전 두 결함이 모두 "카운터는 맞는데
        # 실제 값이 다른" 유형이었고 수 시간 뒤에야 드러났다. 위반 시 즉시 정지.
        assert_retrain_contract(ckpt, n, st, os.path.join(stream_data_dir, f"{st.cluster}.hdf5"),
                                smoke=args.smoke)
        st.ckpt = ckpt
        st.maturity.reinit_after_retrain()               # c=0.25 이월
        outcomes = []
        env = env_of(st.suite, st.task)
        probe_calls = 0
        for spec in probe_specs(st.suite, st.task, min(st.maturity.probe_rounds, 1))[:n_probe]:
            obs = spec.realize(env)
            ok, _t, nc = run_habit(env, obs, ckpt)
            outcomes.append(ok)
            probe_calls += nc
            ledger["formation_episodes"] += 1
        if force_fail:                                    # 스모크 전용: 부적격 경로 관측
            outcomes = [False] * len(outcomes)
        mature = st.maturity.record_probe_round(outcomes)
        st.next_grid_idx = min(st.next_grid_idx + 1, len(grid) - 1)
        dt = time.time() - t0
        ledger["formation_s"] += dt
        print(f"[E5-PROBE] {st.cluster} v{st.version} n={n} probe={sum(outcomes)}/{len(outcomes)} "
              f"mature={mature} state={st.state()} ({dt:.0f}s)", flush=True)
        return {"version": st.version, "n": n, "probe_round": st.maturity.probe_rounds,
                "passed": bool(mature), "formation_wall_s": round(dt, 1),
                "formation_episodes": len(outcomes), "forced_fail_injection": force_fail,
                "probe_success_count": int(sum(bool(o) for o in outcomes)),
                "probe_failure_count": int(sum(not bool(o) for o in outcomes)),
                "probe_habit_calls": probe_calls,
                "train_wall_s": round(t_train, 1),
                "bc_pool_at_trigger": len(st.bc_pool)}

    from envs.stream import E5_NOVEL_POOL
    cold_start_clusters = {f"{s_}_task{t_}" for s_, t_ in E5_NOVEL_POOL}
    events_path = os.path.join(outdir, f"lifecycle_events_{args.seed_idx}.jsonl")
    evf = open(events_path, "a")

    def emit_event(ev_type, i, st, before, after, extra=None):
        """§9 상태 전이 원장 — 그림은 만들지 않고 전이 데이터만 완전 저장한다."""
        evf.write(json.dumps({
            "seed": args.seed_idx, "episode": i, "cluster_id": st.cluster,
            "suite": st.suite, "task_id": st.task,
            "cold_start": st.cluster in cold_start_clusters,
            "event_type": ev_type, "state_before": before, "state_after": after,
            "B_k_size": len(st.bc_pool), "sigma_k": st.maturity.succ,
            "phi_k": st.maturity.fail, "tau_k": round(st.maturity.tau, 6),
            "policy_version": st.version, "probe_rounds": st.maturity.probe_rounds,
            **(extra or {}),
        }, ensure_ascii=False) + "\n")
        evf.flush()

    for i, (spec, cl, is_novel) in enumerate(specs):
        st = states[cl]
        env = env_of(spec.suite_name, spec.task_id)
        ep_t0 = time.time()
        ep_wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
        state_before = st.state()
        use_habit = False
        n_vla_calls = n_hab_calls = 0
        first_exposure = st.version == 0 and not st.bc_pool and not st.maturity.history
        try:
            obs = spec.realize(env)
            shadow = None
            if st.jur is not None:
                f = pca.transform(ext.embed([prep_gate_rgb(obs["agentview_image"])]))[0]
                sc = st.jur.score(f)
                shadow = {"score": round(float(sc), 3), "q": round(st.jur.q, 3),
                          "accept": bool(sc <= st.jur.q)}
            use_habit = (state_before == "M" and st.ckpt is not None)
            reason = ("fire" if use_habit else
                      {"U": "unknown_cluster", "X": "habit_ineligible"}.get(state_before, "immature"))
            if use_habit:
                success, steps, n_hab_calls = run_habit(env, obs, st.ckpt)
                st.maturity.update(success, source="fire")
                st.aci.observe_fire(success, st.maturity)
                cf_queue.append({"uid": spec.uid, **spec.to_dict(), "cluster": cl,
                                 "habit_success": bool(success)})
            else:
                success, steps, tframes, tactions, n_vla_calls = run_teacher(env, obs)
                st.maturity.update(success, source="teacher")
                store_stream_trajectory(cl, spec, tframes, tactions,
                                        "success" if success else "fail", steps)
                if success:
                    st.bc_pool.append(spec.uid)
            outcome = "success" if success else "fail"
        except InfraError as e:
            outcome, steps, success, shadow, reason = "infra_error", 0, False, None, "infra"
            print(f"[E5-EP] {i+1}/{len(specs)} INFRA {e}", flush=True)
        wall = time.time() - ep_t0
        ledger["operational_s"] += wall

        # 재학습 트리거 (설계서 §1.1): |B_k| 도달 + I/U 상태 + **R_max 미소진**.
        # R_max 가드가 없으면 마지막 그리드 지점 이후 |B_k|가 계속 커져 재학습·probe가
        # 반복되고, 2라운드 통과 후 강등 시 3라운드 진입 = §4h(R_max=2 전역) 위반이 된다.
        # (50 ep 스모크 관문이 이 결함을 검출 — 본실행 전 차단)
        retrain_ev = None
        if (outcome != "infra_error" and st.state() in ("I", "U")
                and len(st.bc_pool) >= grid[st.next_grid_idx]
                and st.maturity.probe_rounds < MaturityGate.PROBE_MAX_ROUNDS
                and not st.maturity.ineligible):
            force = args.smoke and st.cluster == SMOKE_FORCE_FAIL_CLUSTER
            was_first_training = st.version == 0
            retrain_ev = retrain_and_probe(st, force_fail=force)
            emit_event("first_training" if was_first_training else "retraining", i, st,
                       state_before, st.state(),
                       {"n": retrain_ev["n"], "probe_round": retrain_ev["probe_round"],
                        "probe_success_count": retrain_ev["probe_success_count"],
                        "probe_failure_count": retrain_ev["probe_failure_count"],
                        "passed": retrain_ev["passed"]})

        # --- §9 상태 전이 판정 (로깅 전용 — 제어 흐름에 개입하지 않는다)
        state_after = st.state()
        demotion = bool(state_before == "M" and state_after != "M" and state_after != "X")
        transition_to_X = bool(state_before != "X" and state_after == "X")
        newly_mature = bool(state_before != "M" and state_after == "M")
        rematuration = bool(newly_mature and st.ever_mature)
        if first_exposure:
            emit_event("first_exposure", i, st, state_before, state_after)
        if newly_mature:
            emit_event("rematuration" if rematuration else "first_maturity", i, st,
                       state_before, state_after)
            st.ever_mature = True
        if demotion:
            emit_event("demotion", i, st, state_before, state_after)
        if transition_to_X:
            emit_event("transition_X", i, st, state_before, state_after)

        logf.write(json.dumps({
            "t": i, "spec_uid": spec.uid, **spec.to_dict(), "cluster": cl,
            "is_novel_injection": bool(is_novel), "lifecycle_state": state_before,
            "executor": "habit" if reason == "fire" else "vla", "decision_reason": reason,
            "outcome": outcome, "steps": steps, "bc_pool": len(st.bc_pool),
            "ledger_update": {"source": "fire" if reason == "fire" else "teacher",
                              "success": bool(success)},
            "sigma_k": st.maturity.succ, "phi_k": st.maturity.fail,
            "p_ge_tau": round(st.maturity.prob_ge_tau(), 4), "tau": round(st.maturity.tau, 4),
            "shadow_jur": shadow,
            "aci": {"fired": st.aci.fired, "fired_fail": st.aci.fired_fail,
                    "empirical_risk": round(st.aci.empirical_risk(), 4)},
            "retrain_event": retrain_ev, "wall_s": round(wall, 2), "ledger": "operational",
            # ---- §8 원장 필드 (2026-08-28 RGB-only rerun). 전부 **가산 로깅**이며
            # 제어 흐름·통계에 개입하지 않는다. 기존 키는 하나도 제거되지 않았다.
            "seed": args.seed_idx, "episode": i,
            "cluster_id": cl, "task_id": spec.task_id,
            "cold_start": cl in cold_start_clusters,
            "initial_state_id": spec.base_idx, "episode_seed": spec.seed,
            "observation_noise_seed": spec.noise_seed, "perturbation_width": spec.w,
            "controller": "habit" if reason == "fire" else "vla",
            "state_before": state_before, "state_after": state_after,
            "B_k_size": len(st.bc_pool),
            "training_triggered": retrain_ev is not None,
            "training_round": st.version,
            "probe_triggered": retrain_ev is not None,
            "probe_success_count": retrain_ev["probe_success_count"] if retrain_ev else None,
            "probe_failure_count": retrain_ev["probe_failure_count"] if retrain_ev else None,
            "tau_k": round(st.maturity.tau, 6),
            "habit_fired": bool(use_habit),
            "habit_success": (bool(success) if use_habit else None),
            "teacher_used": bool(not use_habit and outcome != "infra_error"),
            "teacher_success": (bool(success) if (not use_habit and outcome != "infra_error")
                                else None),
            "demotion": demotion, "rematuration": rematuration,
            "transition_to_X": transition_to_X,
            "episode_success": bool(success) if outcome != "infra_error" else None,
            "VLA_calls": n_vla_calls, "habit_calls": n_hab_calls,
            "episode_latency": round(wall, 3), "wall_clock_time": ep_wall_start,
            "policy_version": st.version, "probe_rounds": st.maturity.probe_rounds,
            "ineligible": bool(st.maturity.ineligible),
        }, ensure_ascii=False) + "\n")
        logf.flush()
        if (i + 1) % 10 == 0 or retrain_ev:
            fired = sum(1 for s in states.values() if s.state() == "M")
            print(f"[E5-EP] {i+1}/{len(specs)} {cl} {reason} {outcome} | 성숙 클러스터 {fired} "
                  f"| BC {len(st.bc_pool)}", flush=True)

    logf.close()
    evf.close()
    with open(os.path.join(outdir, f"cf_queue_{args.seed_idx}.jsonl"), "w") as f:
        for e in cf_queue:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {"seed_idx": args.seed_idx, "n_episodes": len(specs), "smoke": args.smoke,
               "no_depth": bool(args.no_depth),
               "modality": "rgb_only" if args.no_depth else "rgbd",
               "ck_root": ckroot, "data_root": stream_data_dir,
               "lifecycle_events": os.path.basename(events_path),
               "ever_mature": [cl for cl, s in states.items() if s.ever_mature],
               "ledger_s": {k: round(v, 1) if isinstance(v, float) else v
                            for k, v in ledger.items()},
               "cf_queue_size": len(cf_queue),
               "final_states": {cl: s.state() for cl, s in states.items()},
               "ineligible": [cl for cl, s in states.items() if s.maturity.ineligible],
               "total_wall_s": round(time.time() - t_start, 1)}
    with open(os.path.join(outdir, f"summary_{args.seed_idx}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for e in envs.values():
        e.close()
    print(f"[E5-DONE] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
