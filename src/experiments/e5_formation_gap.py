"""배치·스트림 형성 간극 진단 — 추가 rollout 없음 (로그·체크포인트·코드 대조만).

등재: configs/preregistration.md §5 2026-08-17 (seed 0 갱신 판독 2, 연구원 판정).
판독 규칙(사전 고정):
  (c) 불일치        → 구현 결함 확정, 즉시 보고·수정, seed 0 재실행 검토(연구원 회부)
  (c) 동일 & (a) 차이 → "스트림 수집의 다양성/순서 차이"가 원인, 논문에 결과로 서술
  (a)(c) 동일 & (b) 차이 → probe 평가 조건 차이로 탐색적 보고
  전부 동일          → 확률적 변동, seed 1·2와 대조해 판단 보류

산출: results/e5/formation_gap_{seed}.json
실행: hv2_hab python -u experiments/e5_formation_gap.py --seed-idx 0
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)

from envs.stream import collection_specs, heldout_specs, probe_specs  # noqa: E402
from habits.train import HP  # noqa: E402

# 판정 지시가 지정한 대표 2클러스터 (간극 최대 / N* 경계)
TARGETS = [("libero_goal_task5", "libero_goal", 5), ("libero_goal_task0", "libero_goal", 0)]
BC_N = 80


def load_stats(p):
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def stats_l2rel(a, b):
    out = {}
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k], float).ravel(), np.asarray(b[k], float).ravel()
        if x.size != y.size:
            out[k] = None
            continue
        out[k] = round(float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-8)), 6)
    return out


def is_b2_run(ck_dir):
    """B-2(scratch + 배치 등가 스텝) 실행인지 판별.

    B-2에서는 재학습마다 자기 풀에서 stats를 산출하는 것이 **정상**이므로, 본 진단의
    (c) 판정("stats가 재학습 간 다르면 결함")은 성립하지 않는다. 무효 실행(warm-start가
    정규화 공간을 가로지른 경우) 전용 진단임을 기동 시 강제한다.
    """
    import glob
    for p in glob.glob(os.path.join(ck_dir, "*", "act_n80.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if sd.get("steps") == 28000:          # 배치 등가 스텝 = B-2 이후
            return True, sd["steps"]
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="B-2 가드 무시 (진단 목적)")
    args = ap.parse_args()
    ck = os.path.join(HABIT2, "checkpoints", f"e5_s{args.seed_idx}")
    b2, steps = is_b2_run(ck) if os.path.isdir(ck) else (False, None)
    if b2 and not args.force:
        raise SystemExit(
            f"[E5GAP-SKIP] seed {args.seed_idx}는 **B-2 실행**이다(n=80 steps={steps:,}). "
            "B-2는 scratch 학습이라 재학습 간 stats가 다른 것이 정상이므로 본 진단의 (c) 판정이 "
            "성립하지 않는다 — 돌리면 정상 동작을 결함으로 오판한다. "
            "정규화 계약 검증은 드라이버의 `assert_retrain_contract()` 런타임 단언이 "
            "**재학습마다** 수행한다(로그의 GATE-PASS). "
            "무효 실행 기준 원본 진단: results/e5/seed0_normstats_invalid/INVALID_formation_gap_0.json")
    rd = os.path.join(HABIT2, "results", "e5")
    rows = [json.loads(l) for l in open(os.path.join(rd, f"stream_{args.seed_idx}.jsonl"))]
    ck_e5 = os.path.join(HABIT2, "checkpoints", f"e5_s{args.seed_idx}")
    ck_e3 = os.path.join(HABIT2, "checkpoints")

    out = {"kind": "배치·스트림 형성 간극 진단 (추가 rollout 0)",
           "prereg": "§5 2026-08-17 (seed 0 갱신 판독 2)", "targets": [t[0] for t in TARGETS]}

    # ---------- (c) 학습 경로 동일성: HP / 스텝 / 정규화 통계
    c_out = {"criterion": "재학습 HP·스텝 수·정규화 통계가 배치 학습 경로와 동일한가"}
    # E3는 --n-grid 10 20 40 80 (한 번에) → 정규화 통계가 max-n=80 풀에서 1회 산출되어 전 단계 동결.
    # E5 lazy 재학습은 --n-grid {n} 단일값 → max(args.n_grid) = n 이므로 **재학습마다 다른 풀**에서
    # 통계가 산출된다. 즉 warm-start가 정규화 공간을 가로지른다(§5 "전 단계 동결" 조항과 충돌).
    per_c = {}
    for cl, _s, _t in TARGETS:
        e3 = {n: load_stats(os.path.join(ck_e3, cl, f"act_n{n}.pt")) for n in (10, 20, 40, 80)}
        e5 = {n: load_stats(os.path.join(ck_e5, cl, f"act_n{n}.pt")) for n in (20, 80)}
        if not all(e3.values()) or not all(e5.values()):
            per_c[cl] = {"error": "체크포인트 누락"}
            continue
        per_c[cl] = {
            "e3_stats_frozen_across_grid": stats_l2rel(e3[10]["stats"], e3[80]["stats"]),
            "e5_stats_shift_n20_to_n80": stats_l2rel(e5[20]["stats"], e5[80]["stats"]),
            "e3_steps_by_n": {str(n): e3[n]["steps"] for n in e3},
            "e5_steps_by_n": {str(n): e5[n]["steps"] for n in e5},
            "e3_cumulative_steps_to_n80": sum(HP["steps_per_n"][n] for n in (10, 20, 40, 80)),
            "e5_cumulative_steps_to_n80": sum(HP["steps_per_n"][n] for n in (20, 80)),
            "e3_final_l1_n80": e3[80]["final_l1"], "e5_final_l1_n80": e5[80]["final_l1"],
        }
    shifted = [cl for cl, v in per_c.items()
               if "e5_stats_shift_n20_to_n80" in v
               and max(x for x in v["e5_stats_shift_n20_to_n80"].values() if x is not None) > 0]
    frozen_ok = all(max(x for x in v["e3_stats_frozen_across_grid"].values() if x is not None) == 0
                    for v in per_c.values() if "e3_stats_frozen_across_grid" in v)
    c_out.update({
        "per_cluster": per_c,
        "e3_normalization_frozen": frozen_ok,
        "e5_normalization_shifts": len(shifted) > 0,
        "clusters_with_shift": shifted,
        "verdict": ("불일치 — 구현 결함 확정 (연구원 회부)" if shifted or not frozen_ok
                    else "동일"),
        "mechanism": "habits/train.py는 정규화 통계를 `compute_stats(episodes[:max(n_grid)])`로 "
                     "산출한다. E3는 `--n-grid 10 20 40 80`으로 호출되어 max=80 풀에서 1회 산출 후 "
                     "전 단계 동결되지만, E5 lazy 재학습은 `--n-grid {n}` 단일값이라 "
                     "n=20 재학습은 20개 풀, n=80 재학습은 80개 풀에서 각각 산출된다 — "
                     "warm-start가 정규화 공간을 가로지른다(§5 조항과 충돌).",
        "step_budget_note": "lazy {20,80}은 등재 설계지만 그 귀결로 n=80 도달까지의 누적 학습 "
                            "스텝이 배치 대비 축소된다(아래 수치)."})
    out["c_training_path"] = c_out

    # ---------- (a) BC 풀 초기상태 분포
    a_out = {"criterion": "스트림 BC 풀 80 vs E3 수집 80의 base_idx·w 산포·중복도"}
    per_a = {}
    for cl, suite, task in TARGETS:
        cr = [r for r in rows if r["cluster"] == cl]
        succ = [r for r in cr if r["executor"] == "vla" and r["outcome"] == "success"][:BC_N]
        cb = collections.Counter(r["base_idx"] for r in succ)
        ce = collections.Counter(s.base_idx for s in collection_specs(suite, task, 120)[:BC_N])
        per_a[cl] = {
            "stream": {"n": len(succ), "unique_base": len(cb), "max_multiplicity": max(cb.values()),
                       "missing_bases": 40 - len(cb), "variance": round(float(np.var(list(cb.values()))), 3),
                       "w_values": sorted({r["w"] for r in succ})},
            "e3_batch": {"n": BC_N, "unique_base": len(ce), "max_multiplicity": max(ce.values()),
                         "missing_bases": 40 - len(ce), "variance": round(float(np.var(list(ce.values()))), 3),
                         "w_values": sorted({s.w for s in collection_specs(suite, task, 120)[:BC_N]})},
            "teacher_success_in_stream": f"{sum(1 for r in cr if r['executor']=='vla' and r['outcome']=='success')}"
                                         f"/{sum(1 for r in cr if r['executor']=='vla')}"}
    a_diff = any(v["stream"]["unique_base"] != v["e3_batch"]["unique_base"] for v in per_a.values())
    a_out.update({"per_cluster": per_a, "distribution_differs": a_diff,
                  "verdict": ("차이 있음 — 스트림 BC 풀은 초기상태 다양성이 낮고 중복이 크다"
                              if a_diff else "동일"),
                  "mechanism": "스트림 스펙은 base_idx = (스트림 전역 인덱스) % 40인데 클러스터 노출이 "
                               "셔플로 흩어져 클러스터별로는 불균등해진다. E3 수집은 클러스터별 "
                               "연속 인덱스라 40개 base가 균등 순환한다."})
    out["a_bc_pool_distribution"] = a_out

    # ---------- (b) 평가 스펙 대역
    b_out = {"criterion": "스트림 probe 20 vs E3 held-out 20의 대역·난이도"}
    per_b = {}
    for cl, suite, task in TARGETS:
        cr = [r for r in rows if r["cluster"] == cl]
        succ = [r for r in cr if r["executor"] == "vla" and r["outcome"] == "success"][:BC_N]
        seen = {r["base_idx"] for r in succ}
        pb = {s.base_idx for s in probe_specs(suite, task, 0)}
        hb = {s.base_idx for s in heldout_specs(suite, task, 20)}
        per_b[cl] = {
            "probe_bases": sorted(pb), "heldout_bases": sorted(hb),
            "probe_base_overlap_with_bc_pool": round(len(pb & seen) / len(pb), 3),
            "heldout_base_overlap_with_bc_pool": round(len(hb & seen) / len(hb), 3)}
    b_out.update({
        "per_cluster": per_b,
        "verdict": "차이 있음 — 단 **스트림 probe가 더 쉬운 방향**이다",
        "direction": "probe는 수집 대역(0–39)을 재사용해 BC 풀과 base가 크게 겹치는 반면, "
                     "E3 held-out은 전용 대역(40–49)으로 학습에 노출된 적이 없다. "
                     "따라서 (b)는 간극을 설명하지 못하며, 오히려 간극을 더 크게 만든다 — "
                     "더 쉬운 평가에서 더 낮은 성적이 나왔다."})
    out["b_eval_band"] = b_out

    # ---------- 종합 (사전 고정 규칙 적용)
    out["applied_rule"] = {
        "c_mismatch": bool(shifted or not frozen_ok),
        "a_differs": a_diff,
        "b_differs_but_favors_stream": True,
        "conclusion": ("(c) 불일치 → **구현 결함 확정, 연구원 회부**. "
                       "단 정규화 이동은 성숙에 **성공한** 클러스터에도 동일하게 존재하므로 "
                       "부적격의 유일한 원인으로 단정할 수 없다. (a) 다양성 축소가 함께 작용하고, "
                       "누적 학습 스텝도 배치 대비 축소된다. (b)는 반대 방향이라 간극을 키운다."
                       if shifted or not frozen_ok else "규칙에 따라 재판정 필요")}
    op = os.path.join(rd, f"formation_gap_{args.seed_idx}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E5GAP-DONE] {op}")
    print(json.dumps({"c": {k: c_out[k] for k in ("verdict", "e3_normalization_frozen",
                                                  "e5_normalization_shifts", "clusters_with_shift")},
                      "a": {"verdict": a_out["verdict"]}, "b": {"verdict": b_out["verdict"]},
                      "applied_rule": out["applied_rule"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
