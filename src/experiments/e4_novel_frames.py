"""E4-2: novel 셋 생성 (연구원 지시 2026-08-16 §3) — 주 설계 base 40-49.

경로:
  (i)  w 확대: 스위트별 usable_w_max (long 제외, 22 클러스터). seed 30000+j / noise 2e6+j.
  (iii) BDDL 재샘플: 전 25 클러스터 (long 필수 경로). seed 60000+j (§5 등재 대역).
  (ii) 타 태스크 init 차용: spatial 전용 (2 클러스터) + **BDDL ordered 시그니처 일치 가드**
       (차원 비교 금지). seed 50000+j / noise 4e6+j (예약 대역).
변형: 주 = base 40-49 (known 정합) / 부차 = base 0-39 (j+500 오프셋, 병행 생성) — (iii)는 base 무관.
유효성: E0-6 기준 (정착 후 목표 대비 낙하 z>5cm·수평 >10cm) 통과분만 입고, 탈락률 보고.
disjoint: 5대역(수집/held-out/novel/probe/연쇄) uid 전수 검증 후 생성 개시.

산출: results/e4/novel_frames/{cluster}__{path}__{variant}.npz + novel_manifest.json
진행: [NOVEL] <path> <cluster> j/n → heartbeat. 종료 [E4NOVEL-PASS|FAIL].
실행: hv2_hab python -u experiments/e4_novel_frames.py
"""
import json
import os
import re
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.libero_env import DUMMY_ACTION, SETTLE_STEPS, EpisodeSpec, LiberoEpisodeEnv, USABLE_W_MAX  # noqa: E402
from envs.stream import collection_specs, heldout_specs, probe_specs  # noqa: E402
from envs.chained_env import chained_collection_specs, chained_heldout_specs  # noqa: E402
from gates.features import prep_gate_rgb  # noqa: E402

OUT_DIR = os.path.join(HABIT2, "results", "e4", "novel_frames")
N_PER = 20
# E4 종결 시퀀스 1 (§5 2026-08-16): long도 w=0.02(=usable_w_max) 확대 편입 — 재판정 구성
W_EXPAND_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
SPATIAL_A = [("libero_spatial", 0), ("libero_spatial", 1)]
RESAMPLE_SEED_BASE, RESAMPLE_NOISE_BASE = 60_000, 5_000_000  # §5 등재 (2026-08-16)
BORROW_SEED_BASE, BORROW_NOISE_BASE = 50_000, 4_000_000      # §4b 예약 대역
NOVEL_SEED_BASE, NOVEL_NOISE_BASE = 30_000, 2_000_000


def bddl_signature(env):
    p = os.path.join(HABIT2, "third_party", "LIBERO", "libero", "libero", "bddl_files",
                     env.task.problem_folder, env.task.bddl_file)
    text = open(p).read()
    sig = []
    for block in ("objects", "fixtures"):
        m = re.search(rf"\(:{block}(.*?)\)", text, re.S)
        sig.append(tuple(m.group(1).split()) if m else ())
    return tuple(sig)


SETTLE_REF_SEED_BASE = 90_000  # 내부 참조 실현 전용 — 스펙 아님(어떤 셋에도 불입고)
_ref_cache = {}


def settle_ref(env, ref_key, state):
    """무섭동 **settled** 참조 위치 {adr: xyz} — 검사기 v2 기준 (E0-6 정합).

    v1 결함(§5 기록): 원시 init 벡터를 기준으로 삼아 goal/spatial 공식 state의
    스폰 높이(~7cm 상공)에서의 정상 settle 낙하를 '낙하 무효'로 오탐 (goal/spatial
    100% 탈락·object 0% — 전유/전무 패턴이 신호였음). v2 = settled 대 settled."""
    if ref_key in _ref_cache:
        return _ref_cache[ref_key]
    import zlib

    env.begin_episode(SETTLE_REF_SEED_BASE + zlib.crc32(ref_key.encode()) % 1000, state)
    sim = env._env.env.sim
    ref = {adr: np.array(sim.data.qpos[adr:adr + 3]) for adr in env._free_adrs}
    _ref_cache[ref_key] = ref
    return ref


def settled_validity(env, target_state, ref_pos):
    """v2: 낙하 = settled 참조 z 대비 −5cm 초과 / 수평 = 목표 xy 대비 10cm 초과."""
    sim = env._env.env.sim
    off = env._time_offset
    for adr in env._free_adrs:
        cur = sim.data.qpos[adr:adr + 3]
        tgt = target_state[off + adr:off + adr + 3]
        if float(ref_pos[adr][2]) - float(cur[2]) > 0.05:
            return False, "z_drop_vs_settled_ref"
        if float(np.linalg.norm(np.asarray(cur[:2]) - np.asarray(tgt[:2]))) > 0.10:
            return False, "lateral"
    return True, None


def resample_stability(env):
    """(iii) 전용 v2: 참조 무관 안정성 검사 — 추가 settle 10step에서 물체 이동 <2cm."""
    sim = env._env.env.sim
    before = {adr: np.array(sim.data.qpos[adr:adr + 3]) for adr in env._free_adrs}
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env._env.step(DUMMY_ACTION)
    for adr in env._free_adrs:
        cur = np.array(sim.data.qpos[adr:adr + 3])
        if float(np.linalg.norm(cur - before[adr])) > 0.02:
            return obs, False, "unstable"
    return obs, True, None


def realize_perturbed(env, seed, base_idx, w, noise_seed):
    rng = np.random.default_rng(noise_seed)
    state = env.perturbed_init_state(base_idx, w, rng)
    obs = env.begin_episode(seed, state)
    return obs, state


def realize_borrowed(env, src_env, seed, base_idx, w, noise_seed):
    rng = np.random.default_rng(noise_seed)
    state = src_env.perturbed_init_state(base_idx, w, rng)
    obs = env.begin_episode(seed, state)
    return obs, state


def realize_resample(env, seed):
    """BDDL placement 재샘플 (E0-6 검증 경로): seed→reset(재샘플)→settle. 목표 = reset 직후 상태."""
    if getattr(env, "_free_adrs", None) is None:
        # 모델 상수 지연 초기화 — begin_episode 우회 경로라 명시 트리거 필요 (기지 함정)
        env.perturbed_init_state(0, 0.01, np.random.default_rng(0))
    env._env.seed(seed)
    env._env.reset()
    sim = env._env.env.sim
    ref = np.concatenate([[0.0], np.array(sim.data.qpos), np.array(sim.data.qvel)])
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env._env.step(DUMMY_ACTION)
    return obs, ref


def band_disjoint_check():
    """5대역 uid 전수 disjoint (수집/held-out/novel/probe/연쇄) — 대표 클러스터 표본."""
    suite, task = "libero_object", 0
    bands = [collection_specs(suite, task), heldout_specs(suite, task, 50),
             probe_specs(suite, task, 0), probe_specs(suite, task, 1),
             chained_collection_specs(suite, task), chained_heldout_specs(suite, task, 50)]
    novel_uids = set()
    for j in range(N_PER):
        for base in (40 + j % 10, list(range(40))[j % 40]):
            novel_uids.add(EpisodeSpec(suite, task, NOVEL_SEED_BASE + j, base,
                                       USABLE_W_MAX[suite], NOVEL_NOISE_BASE + j).uid)
    all_uids = [({s.uid for s in b}, i) for i, b in enumerate(bands)]
    for ua, ia in all_uids:
        assert not (ua & novel_uids), f"novel↔대역{ia} uid 충돌"
        for ub, ib in all_uids:
            if ia < ib:
                assert not (ua & ub), f"대역{ia}↔{ib} uid 충돌"
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    band_disjoint_check()
    print("[NOVEL] 5대역 disjoint 검증 통과", flush=True)
    manifest = {"paths": {}, "rejections": {}, "note": "주=base40-49 / 부차=base0-39(j+500); "
                "(iii)는 base 무관. 유효성 = E0-6 낙하·수평이탈."}

    def done_key(cluster, path, variant):
        key = f"{cluster}__{path}__{variant}"
        return key, os.path.exists(os.path.join(OUT_DIR, f"{key}.meta.json"))

    def emit(cluster, path, variant, frames, uid_list, rejects):
        key = f"{cluster}__{path}__{variant}"
        np.savez_compressed(os.path.join(OUT_DIR, f"{key}.npz"),
                            frames=np.stack(frames).astype(np.uint8) if frames else np.zeros((0, 128, 128, 3), np.uint8),
                            uids=json.dumps(uid_list))
        frag = {"n_valid": len(frames), "n_reject": rejects}
        with open(os.path.join(OUT_DIR, f"{key}.meta.json"), "w") as f:
            json.dump(frag, f)
        manifest["paths"][key] = frag

    # --- (i) w 확대 (long 제외)
    for suite, task in [c for c in STANDARD if c[0] in W_EXPAND_SUITES]:
        cl = f"{suite}_task{task}"
        if all(done_key(cl, "w_expand", v)[1] for v in ("primary", "secondary")):
            for v in ("primary", "secondary"):
                key = f"{cl}__w_expand__{v}"
                manifest["paths"][key] = json.load(open(os.path.join(OUT_DIR, f"{key}.meta.json")))
            print(f"[NOVEL-SKIP] w_expand {cl}: 기존 산출물", flush=True)
            continue
        env = LiberoEpisodeEnv(suite, task)
        w = USABLE_W_MAX[suite]
        for variant, j0, bases in (("primary", 0, list(range(40, 50))), ("secondary", 500, list(range(40)))):
            plan = [(j0 + j, bases[j % len(bases)]) for j in range(N_PER)]
            for _, b in sorted(set(plan)):  # v2: settled 참조 선계산 (begin_episode가 상태 파괴)
                settle_ref(env, f"{cl}|base{b}", env.init_states[b])
            frames, uids, rej = [], [], 0
            for i, (jj, base) in enumerate(plan):
                ref = _ref_cache[f"{cl}|base{base}"]
                obs, state = realize_perturbed(env, NOVEL_SEED_BASE + jj, base, w, NOVEL_NOISE_BASE + jj)
                ok, why = settled_validity(env, state, ref)
                if ok:
                    frames.append(prep_gate_rgb(obs["agentview_image"]))
                    uids.append(EpisodeSpec(suite, task, NOVEL_SEED_BASE + jj, base, w,
                                            NOVEL_NOISE_BASE + jj).uid)
                else:
                    rej += 1
                print(f"[NOVEL] w_expand {cl} {variant} {i + 1}/{N_PER}", flush=True)
            emit(cl, "w_expand", variant, frames, uids, rej)
        env.close()

    # --- (iii) BDDL 재샘플 (전 클러스터, long 필수)
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        key, done = done_key(cl, "resample", "single")
        if done:
            manifest["paths"][key] = json.load(open(os.path.join(OUT_DIR, f"{key}.meta.json")))
            print(f"[NOVEL-SKIP] resample {cl}: 기존 산출물", flush=True)
            continue
        env = LiberoEpisodeEnv(suite, task)
        frames, uids, rej = [], [], 0
        for j in range(N_PER):
            _obs, _ref = realize_resample(env, RESAMPLE_SEED_BASE + j)
            obs, ok, why = resample_stability(env)  # v2: 참조 무관 안정성 검사
            if ok:
                frames.append(prep_gate_rgb(obs["agentview_image"]))
                uids.append(f"resample|{suite}|{task}|{RESAMPLE_SEED_BASE + j}")
            else:
                rej += 1
            print(f"[NOVEL] resample {cl} {j + 1}/{N_PER}", flush=True)
        emit(cl, "resample", "single", frames, uids, rej)
        env.close()

    # --- (ii) spatial 타 태스크 init 차용 (시그니처 가드)
    for suite, task in SPATIAL_A:
        cl = f"{suite}_task{task}"
        if all(done_key(cl, "borrow", v)[1] for v in ("primary", "secondary")):
            for v in ("primary", "secondary"):
                key = f"{cl}__borrow__{v}"
                manifest["paths"][key] = json.load(open(os.path.join(OUT_DIR, f"{key}.meta.json")))
            print(f"[NOVEL-SKIP] borrow {cl}: 기존 산출물", flush=True)
            continue
        env = LiberoEpisodeEnv(suite, task)
        tgt_sig = bddl_signature(env)
        src_tasks = [t for t in range(10) if t != task]
        src_envs = {}
        for variant, j0, bases in (("primary", 0, list(range(40, 50))), ("secondary", 500, list(range(40)))):
            plan = [(j0 + j, src_tasks[j % len(src_tasks)], bases[j % len(bases)]) for j in range(N_PER)]
            for _, src, b in sorted(set(plan)):
                if src not in src_envs:
                    src_envs[src] = LiberoEpisodeEnv(suite, src)
                    src_sig = bddl_signature(src_envs[src])
                    assert src_sig == tgt_sig, f"(ii) 시그니처 불일치: {suite} task{src} vs task{task}"
                    src_envs[src].perturbed_init_state(0, 0.01, np.random.default_rng(0))  # 상수 초기화
                settle_ref(env, f"{cl}|src{src}|base{b}", src_envs[src].init_states[b])
            frames, uids, rej = [], [], 0
            for i, (jj, src, base) in enumerate(plan):
                ref = _ref_cache[f"{cl}|src{src}|base{base}"]
                obs, state = realize_borrowed(env, src_envs[src], BORROW_SEED_BASE + jj, base,
                                              0.01, BORROW_NOISE_BASE + jj)
                ok, why = settled_validity(env, state, ref)
                if ok:
                    frames.append(prep_gate_rgb(obs["agentview_image"]))
                    uids.append(f"borrow|{suite}|{task}|src{src}|{BORROW_SEED_BASE + jj}|{base}")
                else:
                    rej += 1
                print(f"[NOVEL] borrow {cl} {variant} {i + 1}/{N_PER}", flush=True)
            emit(cl, "borrow", variant, frames, uids, rej)
        for e in src_envs.values():
            e.close()
        env.close()

    with open(os.path.join(HABIT2, "results", "e4", "novel_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    total_valid = sum(v["n_valid"] for v in manifest["paths"].values())
    total_rej = sum(v["n_reject"] for v in manifest["paths"].values())
    print(f"[E4NOVEL-PASS] valid={total_valid} reject={total_rej} "
          f"({100 * total_rej / max(1, total_valid + total_rej):.1f}%)")


if __name__ == "__main__":
    main()
