"""실행기 chunk-break 단위 테스트 (§4e 개정 α — 연구원 지시 1항).

핵심 단언: stage 전환이 chunk 중간에 오면 잔여 stale 행동이 **폐기**된다 (diag5b 확증 결함의
회귀 방지). 실제 코드 경로(envs.chained_env.execute_chunk_with_boundary)를 FakeEnv로 검정.

실행: hv2_hab python -u experiments/executor_chunkbreak_test.py → [EXECUTOR-TEST-PASS]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.chained_env import execute_chunk_with_boundary  # noqa: E402


class FakeChainedEnv:
    """step 호출 카운트 기반으로 지정 스텝에서 stage 1→2 전환하는 스텁."""

    def __init__(self, transition_at):
        self.transition_at = transition_at
        self.n_steps = 0
        self._stage = 1

    def stage(self):
        return self._stage

    def step(self, action):
        self.n_steps += 1
        if self.n_steps >= self.transition_at:
            self._stage = 2
        return {"obs": self.n_steps}, 0.0, False, {}


class FakePlainEnv:
    def __init__(self):
        self.n_steps = 0

    def step(self, action):
        self.n_steps += 1
        return {"obs": self.n_steps}, 0.0, False, {}


ACTIONS = [[0.0] * 7 for _ in range(8)]


def t1_mid_chunk_transition_discards_tail():
    env = FakeChainedEnv(transition_at=3)
    obs, t, n_exec, stale = execute_chunk_with_boundary(env, ACTIONS, 0, 1000)
    assert n_exec == 3 and stale == 5, (n_exec, stale)
    assert env.n_steps == 3, "전환 후 stale 행동이 실행됨 — 폐기 실패"
    print(f"t1 중간 전환 폐기 OK (실행 {n_exec}, stale 폐기 {stale})")


def t2_no_transition_runs_full_chunk():
    env = FakeChainedEnv(transition_at=10_000)
    obs, t, n_exec, stale = execute_chunk_with_boundary(env, ACTIONS, 0, 1000)
    assert n_exec == 8 and stale is None, (n_exec, stale)
    print("t2 무전환 전량 실행 OK")


def t3_last_action_transition_zero_stale():
    env = FakeChainedEnv(transition_at=8)
    obs, t, n_exec, stale = execute_chunk_with_boundary(env, ACTIONS, 0, 1000)
    assert n_exec == 8 and stale == 0, (n_exec, stale)
    print("t3 경계 일치 전환 stale=0 OK")


def t4_plain_env_unaffected():
    env = FakePlainEnv()
    obs, t, n_exec, stale = execute_chunk_with_boundary(env, ACTIONS, 0, 1000)
    assert n_exec == 8 and stale is None
    print("t4 일반 env 무영향 OK (E2/E3 표준·E5 영향 반경 밖 — §4e (e))")


def t5_max_steps_cutoff():
    env = FakeChainedEnv(transition_at=10_000)
    obs, t, n_exec, stale = execute_chunk_with_boundary(env, ACTIONS, 998, 1000)
    assert t == 1000 and n_exec == 2 and stale is None, (t, n_exec, stale)
    print("t5 max_steps 절단 OK")


if __name__ == "__main__":
    t1_mid_chunk_transition_discards_tail()
    t2_no_transition_runs_full_chunk()
    t3_last_action_transition_zero_stale()
    t4_plain_env_unaffected()
    t5_max_steps_cutoff()
    print("[EXECUTOR-TEST-PASS]")
