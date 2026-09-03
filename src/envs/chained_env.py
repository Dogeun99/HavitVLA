"""C-T2: 단일 태스크 2연쇄 래퍼 (설계서 §3 — 유일한 커스텀 엔지니어링).

설계: 동일 태스크를 한 에피소드 안에서 2회 연속 수행 — L(의미 복잡도)을 상수로 고정한 채
horizon T만 2배로 늘리는 통제 셀. T-스윕: T1 = C-L1(동일 태스크 단일) → T2 = 본 래퍼 → T3 = Long.

메커니즘 (§4e 개정 — 옵션 A, 연구원 판정 2026-08-15):
  stage 1: 통상 에피소드 (begin_episode 재현 프로토콜 그대로).
  stage 1 성공 감지 → **전환 (v3)**: 두 번째 init state(spec의 relocate_base_idx + 섭동)
    전체 상태 벡터로 **완전한 에피소드 경계 프로토콜**(E0-6 3단: seed → reset →
    set_init_state → settle = begin_episode 재사용) 수행 — 물체는 재배치 포즈로,
    **로봇·컨트롤러는 fresh 에피소드와 동등하게 재초기화**.
    의미론 = "재발 조우 연속의 압축"(스트림 에피소드 경계와 동형), 로봇 포즈 연속성
    단절은 preregistration §4e에 명시.
    근거: teacher(OFT)는 홈 포즈 시작 분포로만 학습 — stage 1 종료 포즈 재시작은 관할 밖
    (v1: OOD 동결 0/10). v2(set_init_state만)는 OSC 이월로 로봇이 홈에서 0.48 rad 이탈
    (t2_diag4) — task5 오인 절벽에서 0/9. v3 = fresh 동등 (task5 fresh 통제 성공 실증).
  stage 2: 동일 predicate가 다시 참이 되면 최종 성공.
  총 예산 = 2 × 원 태스크 max_steps (+settle).

결정성: 재배치가 spec (relocate_base_idx, relocate_noise_seed)에서 유도 — 완전 결정적.
성공 의미론: check_success() = stage 2 완료. 수집기·평가기는 무수정 재사용
(step에서 stage 전환을 내부 처리, 외부 인터페이스는 LiberoEpisodeEnv와 동일).

검증(수집 전 필수, e3_t2_validate.py): ① 동일 spec 2회 → 상태 해시 일치 ② 재배치 유효성
(z-낙하·수평 이탈 검사, E0-6 기준) ③ teacher가 stage 2를 실제로 수행하는가 (10-ep 스모크,
stage별 성공 분해). ③ 실패 시 설계서 §3의 대체(Long 길이 층화) 발동 — 임의 우회 금지.
"""
import numpy as np

from .libero_env import DUMMY_ACTION, SETTLE_STEPS, InfraError, LiberoEpisodeEnv, USABLE_W_MAX


class ChainedEpisodeEnv(LiberoEpisodeEnv):
    """단일 태스크 2연쇄. 외부 인터페이스는 LiberoEpisodeEnv와 동일."""

    def __init__(self, suite_name, task_id, resolution=256, depth=True):
        super().__init__(suite_name, task_id, resolution, depth)
        self.max_steps = 2 * self.max_steps  # T2 = 2×T1
        self._stage = 1
        self._relocate_state = None
        self.stage_steps = {1: None, 2: None}  # 분석용: stage별 소요 스텝
        self._t = 0

    # ---- spec 확장: 재배치 목표 상태를 에피소드 시작 시 고정 ----
    def begin_chained_episode(self, seed, init_state, relocate_base_idx, relocate_noise_seed, w):
        self._stage = 1
        self._t = 0
        self._transition_seed = seed  # 전환 프로토콜(v3)의 결정적 seed — fresh 통제(e)와 동일 구성
        self.stage_steps = {1: None, 2: None}
        obs = self.begin_episode(seed, init_state)
        # 재배치 목표: 두 번째 base state의 물체 포즈 (+동일 폭 섭동) — 시작 시점에 확정(결정성)
        rng = np.random.default_rng(relocate_noise_seed)
        w_max = USABLE_W_MAX[self.suite_name]
        w = min(w, w_max)
        s = self.init_states[relocate_base_idx].copy()
        for adr in self._free_adrs:
            s[self._time_offset + adr : self._time_offset + adr + 2] += rng.uniform(-w, w, size=2)
        self._relocate_state = s
        return obs

    def _relocate_objects(self):
        """stage 전환 v3 (§4e 옵션 A의 정확한 구현): **완전한 에피소드 경계 프로토콜** —
        E0-6 3단 (seed → reset → set_init_state → settle) = begin_episode 재사용.

        v2(set_init_state + settle만)의 결함 (t2_diag4 실측): reset 없이는 OSC 컨트롤러
        내부 상태(stage 1의 stale goal·interpolator)가 이월되어 settle 동안 로봇이 홈에서
        최대 0.48 rad 이탈 (물체는 Δ=0.0으로 완전 동일) — E0-6이 재현 프로토콜을 3단으로
        강제한 바로 그 이유. reset이 컨트롤러를 재초기화해 stage 2 시작 상태가 fresh
        에피소드와 구성적으로 동등해진다 (task5 fresh 통제 성공 118스텝 vs v2 전환 0/9)."""
        obs = self.begin_episode(self._transition_seed, self._relocate_state)
        self._t += SETTLE_STEPS  # begin_episode 내부 settle을 스텝 예산에 계상 (v1·v2와 동일)
        return obs

    def step(self, action):
        try:
            obs, r, done, info = self._env.step(action)
            self._t += 1
        except Exception as e:
            raise InfraError(f"chained step failed: {type(e).__name__}: {e}") from e
        if self._stage == 1 and bool(self._env.check_success()):
            self.stage_steps[1] = self._t
            self._stage = 2
            obs = self._relocate_objects()  # stage 2 개시 (predicate는 재배치로 다시 거짓)
            done = False
        elif self._stage == 2 and bool(self._env.check_success()):
            self.stage_steps[2] = self._t
        return obs, r, done, info

    def check_success(self):
        return self._stage == 2 and bool(self._env.check_success())

    def stage(self):
        return self._stage


def execute_chunk_with_boundary(env, actions, t, max_steps, on_step=None):
    """chunk 실행 공용 헬퍼 (§4e 개정 α, 2026-08-15) — 수집·평가 실행기 동형 보장.

    stage 전환(1→2) 감지 시 **잔여 stale 행동을 폐기**하고 즉시 반환(호출자가 재질의) —
    전환 전 관측으로 계산된 행동이 전환 후 상태를 교란하는 결함의 교정
    (diag5: fresh 18/20 성공 / diag5b: chunk-break 18/18 성공으로 확증).
    "경계 재질의"는 K=8 open-loop와 양립 — 경계 = 에피소드 시작과 동형(§4b).

    반환: (obs, t, n_executed, stale_discarded|None). 일반(비 chained) env는 전 행동 실행.
    on_step: 스텝별 obs 콜백(영상 프레임 캡처 등) — 실행 의미론에 영향 없음.
    """
    obs, stale, n_exec = None, None, 0
    for j, a in enumerate(actions):
        pre = env.stage() if hasattr(env, "stage") else None
        obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
        t += 1
        n_exec = j + 1
        if on_step is not None:
            on_step(obs)
        if pre == 1 and hasattr(env, "stage") and env.stage() == 2:
            stale = len(actions) - n_exec
            break
        if done or t >= max_steps:
            break
    return obs, t, n_exec, stale


class ChainedEpisodeSpec:
    """C-T2 에피소드 명세 — EpisodeSpec 확장 (재배치 base/noise 추가)."""

    def __init__(self, suite_name, task_id, seed, base_idx, w, noise_seed,
                 relocate_base_idx, relocate_noise_seed):
        import hashlib

        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = seed
        self.base_idx = base_idx
        self.w = w
        self.noise_seed = noise_seed
        self.relocate_base_idx = relocate_base_idx
        self.relocate_noise_seed = relocate_noise_seed
        key = f"chain|{suite_name}|{task_id}|{seed}|{base_idx}|{w}|{noise_seed}|{relocate_base_idx}|{relocate_noise_seed}"
        self.uid = hashlib.sha256(key.encode()).hexdigest()[:16]

    def realize(self, env: ChainedEpisodeEnv):
        rng = np.random.default_rng(self.noise_seed)
        if self.w > 0:
            state = env.perturbed_init_state(self.base_idx, self.w, rng)
        else:
            state = env.init_states[self.base_idx]
        return env.begin_chained_episode(
            self.seed, state, self.relocate_base_idx, self.relocate_noise_seed, self.w
        )

    def to_dict(self):
        return {
            "uid": self.uid, "suite": self.suite_name, "task_id": self.task_id,
            "seed": self.seed, "base_idx": self.base_idx, "w": self.w,
            "noise_seed": self.noise_seed, "relocate_base_idx": self.relocate_base_idx,
            "relocate_noise_seed": self.relocate_noise_seed, "chained": True,
        }


def chained_collection_specs(suite_name, task_id, n_episodes=120):
    """C-T2 수집 명세 — stream.py의 3중 disjoint 규약 유지 + 재배치 대역."""
    from .stream import COLLECT_BASE_RANGE, COLLECT_SEED_BASE, W_ID

    bases = list(COLLECT_BASE_RANGE)
    specs = []
    for i in range(n_episodes):
        specs.append(
            ChainedEpisodeSpec(
                suite_name, task_id,
                seed=COLLECT_SEED_BASE + i,
                base_idx=bases[i % len(bases)],
                w=W_ID,
                noise_seed=i,
                relocate_base_idx=bases[(i + 17) % len(bases)],  # 시작과 다른 base (결정적 오프셋)
                relocate_noise_seed=500_000 + i,
            )
        )
    return specs


def chained_heldout_specs(suite_name, task_id, n_episodes):
    from .stream import HELDOUT_BASE_RANGE, HELDOUT_SEED_BASE, HELDOUT_NOISE_BASE, W_ID

    bases = list(HELDOUT_BASE_RANGE)
    specs = []
    for j in range(n_episodes):
        specs.append(
            ChainedEpisodeSpec(
                suite_name, task_id,
                seed=HELDOUT_SEED_BASE + j,
                base_idx=bases[j % len(bases)],
                w=W_ID,
                noise_seed=HELDOUT_NOISE_BASE + j,
                relocate_base_idx=bases[(j + 3) % len(bases)],
                relocate_noise_seed=1_500_000 + j,
            )
        )
    return specs
