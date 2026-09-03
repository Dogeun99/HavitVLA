"""HabitVLA-2 LIBERO 환경 래퍼.

설계 근거:
  - depth 노출 (E0-3: camera_depths=True → agentview_depth/robot0_eye_in_hand_depth, [0,1] 정규화)
  - 에피소드 수준 재현 프로토콜 (E0-6 실측 확정): env.seed(seed) → reset() → set_init_state(state)
    → settle. reset 생략 시 OSC 컨트롤러 상태 이월, re-seed 생략 시 placement RNG 스트림 진행으로
    재현 불가 — log.md 2026-08-15.
  - 인프라 오류는 raise (공식 run_episode처럼 삼켜서 Success:False로 위장하지 않음 —
    검증 워크플로우 발견 반영. 정책 실패와 인프라 실패는 라벨이 다르다).

주의: teacher S_V 측정(E1)은 공식 스크립트(run 수준 재현)를 그대로 쓰고,
본 래퍼는 궤적 수집(E2+)·held-out 평가·E5 스트림에 쓴다.
"""
import hashlib
import os

import numpy as np

_HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(_HABIT2, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

SETTLE_STEPS = 10  # 공식 eval의 num_steps_wait와 동일
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# E0-5/공식 eval과 동일한 에피소드 상한 (run_libero_eval.py TASK_MAX_STEPS)
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# E0-6 실측 가용 변이 폭 (results/e0/e0_6_variation.json) — 스트림 생성기의 스위트별 상한
USABLE_W_MAX = {
    "libero_spatial": 0.04,
    "libero_object": 0.04,
    "libero_goal": 0.04,
    "libero_10": 0.02,
}


class InfraError(RuntimeError):
    """시뮬레이터/렌더러 오류 — 정책 실패(success=False)와 반드시 구분해 전파한다."""


def quat2axisangle(quat):
    """(x,y,z,w) → axis-angle. robosuite transform_utils 사본
    (openvla-oft libero_utils.py와 동일식 — TF 의존 없는 로컬 구현, hv2_hab 호환)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def proprio_vector(obs):
    """공식 prepare_observation의 state 구성과 동일: eef pos + axis-angle + gripper qpos (8차원)."""
    return np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)


class LiberoEpisodeEnv:
    """단일 태스크 LIBERO env + 에피소드 수준 재현 프로토콜."""

    def __init__(self, suite_name, task_id, resolution=256, depth=True):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite_name = suite_name
        self.task_id = task_id
        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.task = suite.get_task(task_id)
        self.language = self.task.language
        self.init_states = suite.get_task_init_states(task_id)
        self.max_steps = TASK_MAX_STEPS[suite_name]

        bddl = os.path.join(
            get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=resolution,
            camera_widths=resolution,
            camera_depths=depth,
        )
        self._free_adrs = None
        self._time_offset = None

    # ---- 재현 프로토콜 (E0-6 확정) ----
    def begin_episode(self, seed, init_state):
        """seed → reset → set_init_state → settle. settle 후 관측 반환."""
        try:
            self._env.seed(seed)
            self._env.reset()
            self._capture_model_constants()
            obs = self._env.set_init_state(init_state)
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = self._env.step(DUMMY_ACTION)
            return obs
        except Exception as e:  # 인프라 오류는 위장 없이 전파
            raise InfraError(f"begin_episode failed: {type(e).__name__}: {e}") from e

    def step(self, action):
        try:
            return self._env.step(action)
        except Exception as e:
            raise InfraError(f"step failed: {type(e).__name__}: {e}") from e

    def check_success(self):
        return bool(self._env.check_success())

    def close(self):
        self._env.close()

    # ---- 초기상태 생성 (섭동 경로, E0-6) ----
    def _capture_model_constants(self):
        """reset 직후 model 상수 취득 (reset이 MjSim을 재생성하므로 매번 갱신 — E0-6 v2 교훈)."""
        sim = self._env.env.sim
        self._free_adrs = [
            sim.model.jnt_qposadr[j]
            for j in range(sim.model.njnt)
            if sim.model.jnt_type[j] == 0
        ]
        nq, nv = sim.model.nq, sim.model.nv
        self._time_offset = self.init_states.shape[1] - nq - nv
        assert self._time_offset in (0, 1)

    def perturbed_init_state(self, base_idx, w, rng, allow_beyond_usable=False):
        """공식 init state[base_idx]의 물체 (x,y)에 uniform(-w,w) 섭동.

        기본은 스위트별 usable_w_max 초과를 **차단**한다 (E0-6: 상한 초과는 "물리 무효"와
        "분포 밖"을 혼동시키므로 수집·스트림·novel 생성기에서 금지, §4b).
        `allow_beyond_usable=True`는 **역량 경계 탐색 진단(E4-R) 전용 명시적 예외** —
        상한 밖 폭에서 물리 유효분만 걸러 계수하는 것이 실험 목적이며, 호출측이 E0-6
        유효성 검사(낙하·수평 이탈)를 반드시 수행해야 한다 (§5 등재 2026-08-16).
        """
        w_max = USABLE_W_MAX[self.suite_name]
        if w > w_max and not allow_beyond_usable:
            raise ValueError(f"w={w} exceeds usable_w_max={w_max} for {self.suite_name} (E0-6)")
        if self._free_adrs is None:
            # 상수 취득을 위해 1회 reset (begin_episode가 다시 결정적으로 초기화함)
            self._env.seed(0)
            self._env.reset()
            self._capture_model_constants()
        s = self.init_states[base_idx].copy()
        for adr in self._free_adrs:
            s[self._time_offset + adr : self._time_offset + adr + 2] += rng.uniform(-w, w, size=2)
        return s

    def state_hash(self):
        sim = self._env.env.sim
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(sim.data.qpos).tobytes())
        h.update(np.ascontiguousarray(sim.data.qvel).tobytes())
        return h.hexdigest()[:16]


def episode_spec_hash(suite_name, task_id, seed, base_idx, w, noise_seed):
    """에피소드 명세의 결정적 식별자 (스트림·held-out 정의와 paired 비교의 키)."""
    key = f"{suite_name}|{task_id}|{seed}|{base_idx}|{w}|{noise_seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class EpisodeSpec:
    """스트림/held-out 에피소드의 완전한 결정적 명세."""

    def __init__(self, suite_name, task_id, seed, base_idx=0, w=0.0, noise_seed=0):
        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = seed          # env.seed (reset placement + controller init)
        self.base_idx = base_idx  # 공식 init state 인덱스
        self.w = w                # 섭동 폭 (≤ USABLE_W_MAX[suite])
        self.noise_seed = noise_seed
        self.uid = episode_spec_hash(suite_name, task_id, seed, base_idx, w, noise_seed)

    def realize(self, env: "LiberoEpisodeEnv"):
        """명세 → 초기 관측. 완전 결정적 (E0-6 재현 프로토콜)."""
        if self.w > 0:
            rng = np.random.default_rng(self.noise_seed)
            state = env.perturbed_init_state(self.base_idx, self.w, rng)
        else:
            state = env.init_states[self.base_idx]
        return env.begin_episode(self.seed, state)

    def to_dict(self):
        return {
            "uid": self.uid,
            "suite": self.suite_name,
            "task_id": self.task_id,
            "seed": self.seed,
            "base_idx": self.base_idx,
            "w": self.w,
            "noise_seed": self.noise_seed,
        }
