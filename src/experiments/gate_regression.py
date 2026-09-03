"""게이트 회귀 테스트 (통합 지시서 §2·§10 — 장부 분리 정확성 렌즈 포함).

핵심 단언 (§2): **teacher-only 이력 클러스터는 A_mat을 절대 통과하지 못한다** —
A_mat 사후(σ, φ)는 습관 출처(probe + fire)만 산입 (preregistration §4h).
위반 시 n=80 재학습 이월만으로 probe 0회 무검증 성숙 통과(게이트가 s_H 아닌 s_V 측정).

실행: $HV2_HAB_PY -u experiments/gate_regression.py
판정: 마지막 줄 [GATE-REGRESSION-PASS] (부분 실행·예외 = FAIL).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scipy.stats import beta as beta_dist  # noqa: E402

from gates.two_stage import ACIRiskController, MaturityGate, TwoStageGate  # noqa: E402
from envs.stream import (  # noqa: E402
    collection_specs,
    heldout_specs,
    novel_specs,
    probe_specs,
    assert_disjoint,
)


def t1_teacher_only_never_matures():
    """§2 핵심: teacher 80 성공 + 재학습 이월 → A_mat 미통과 (probe 0회 성숙 차단)."""
    m = MaturityGate()
    for _ in range(80):
        m.update(True, source="teacher")
    m.reinit_after_retrain()
    assert m.succ == 0 and m.fail == 0, "teacher 결과가 A_mat 계수에 산입됨"
    p = m.prob_ge_tau()
    expected = float(beta_dist.sf(0.8, 1, 1))  # Beta(1,1): 0.2
    assert abs(p - expected) < 1e-9, f"teacher-only 사후 {p} != Beta(1,1) {expected}"
    assert not m.accepts(), "teacher-only 클러스터가 A_mat 통과 — §2 위반"
    # 2차 재학습 이월까지 반복해도 동일 (n=80 시나리오)
    m.reinit_after_retrain()
    assert not m.accepts()
    print(f"t1 teacher-only 차단 OK (p={p:.4f} < 0.9)")


def t2_probe_counts_into_a_mat():
    """신선 클러스터 첫 probe = 정확히 Beta(1,1) 기점: 19/20 → 0.9424 통과, 18/20 → 0.8213 탈락."""
    m = MaturityGate()
    for _ in range(20):
        m.update(True, source="teacher")  # n=20 재학습 트리거분 — A_mat 불산입
    m.reinit_after_retrain()
    ok = m.record_probe_round([True] * 19 + [False])
    p = m.prob_ge_tau()
    assert ok and abs(p - float(beta_dist.sf(0.8, 20, 2))) < 1e-9, (ok, p)
    m2 = MaturityGate()
    m2.reinit_after_retrain()
    ok2 = m2.record_probe_round([True] * 18 + [False] * 2)
    assert not ok2 and abs(m2.prob_ge_tau() - float(beta_dist.sf(0.8, 19, 3))) < 1e-9
    print(f"t2 probe 산입 OK (19/20 p={p:.4f} 통과, 18/20 p={m2.prob_ge_tau():.4f} 탈락)")


def t3_fire_carryover():
    """발화 이력은 이월: fire 20 성공 → 재학습 Beta(6,1) → 18/20 probe → Beta(24,3) = 0.9159 통과."""
    m = MaturityGate()
    for _ in range(20):
        m.update(True, source="fire")
    assert m.succ == 20
    m.reinit_after_retrain()
    assert m.succ == 5.0 and m.fail == 0.0
    ok = m.record_probe_round([True] * 18 + [False] * 2)
    p = m.prob_ge_tau()
    assert ok and abs(p - float(beta_dist.sf(0.8, 24, 3))) < 1e-9, (ok, p)
    print(f"t3 fire 이월 OK (Beta(24,3) p={p:.4f} 통과)")


def t4_ineligible_flow():
    """2라운드 미달 → 부적격 전이 → decide=habit_ineligible → 3라운드 차단."""
    m = MaturityGate()
    r1 = m.record_probe_round([True] * 10 + [False] * 10)
    r2 = m.record_probe_round([True] * 12 + [False] * 8)
    assert not r1 and not r2 and m.ineligible and m.probe_rounds == 2
    g = TwoStageGate()
    g.ensure_cluster("k")
    g.maturity["k"] = m
    assert g.decide("k", None) == ("vla", "habit_ineligible")
    try:
        m.record_probe_round([True] * 20)
        raise AssertionError("부적격 후 3라운드가 차단되지 않음")
    except RuntimeError:
        pass
    print("t4 부적격 전이·차단 OK")


def t5_source_validation_and_history():
    m = MaturityGate()
    for bad in ("stream", "retrain", None):
        try:
            m.update(True, source=bad)
            raise AssertionError(f"잘못된 source {bad!r} 통과")
        except ValueError:
            pass
    m.update(True, source="teacher")
    m.update(False, source="fire")
    assert m.history == [("teacher", True), ("fire", False)]
    assert m.succ == 0 and m.fail == 1  # teacher 불산입, fire 산입
    print("t5 source 검증·history OK")


def t6_aci_separation():
    """observe_fire는 τ만 움직이고 A_mat 계수는 불변 (별도 계정)."""
    m = MaturityGate()
    aci = ACIRiskController()
    m.update(True, source="fire")
    s0, f0, tau0 = m.succ, m.fail, m.tau
    aci.observe_fire(False, m)
    assert (m.succ, m.fail) == (s0, f0), "observe_fire가 A_mat 계수를 오염"
    assert m.tau > tau0 and aci.fired == 1 and aci.fired_fail == 1
    print("t6 ACI 분리 OK")


def t7_band_disjointness():
    c = collection_specs("libero_object", 0)
    h = heldout_specs("libero_object", 0, 50)
    n = novel_specs("libero_object", 0, 40)
    p0 = probe_specs("libero_object", 0, round_idx=0)
    p1 = probe_specs("libero_object", 0, round_idx=1)
    bands = [("collect", c), ("heldout", h), ("novel", n), ("probe0", p0), ("probe1", p1)]
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            assert_disjoint(bands[i][1], bands[j][1])
    try:
        probe_specs("libero_object", 0, round_idx=2)
        raise AssertionError("probe round_idx 상한 미차단")
    except ValueError:
        pass
    print("t7 대역 disjoint 10쌍 + probe 라운드 상한 OK")


if __name__ == "__main__":
    t1_teacher_only_never_matures()
    t2_probe_counts_into_a_mat()
    t3_fire_carryover()
    t4_ineligible_flow()
    t5_source_validation_and_history()
    t6_aci_separation()
    t7_band_disjointness()
    print("[GATE-REGRESSION-PASS]")
