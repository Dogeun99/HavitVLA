"""C-T2 수집 게이트 (통합 지시서 §1-2·§5 + envs/chained_env.py 검증 ③).

--mode smoke  : 10-ep 스모크 판정 — "teacher가 stage 2를 실제로 수행하는가".
                stage 분해(1/2 도달·최종 성공) 보고 + gate: 최종 성공 ≥ 7/10.
                미달 = FAIL — 설계서 §3 대체 경로(Long 길이 층화)는 연구원 보고 사항,
                임의 우회 금지.
--mode trigger: 120-ep 수집 상대 트리거 (§1-2, preregistration §4e 보충) —
                관측 성공률이 클러스터별 기대치 S_V,k² 대비 단측 이항 검정 α=0.01로
                유의 미달일 때만 중단·보고. S_V,k는 data/e2 summary에서 프로그래밍
                취득(수동 입력 금지) — 등재값(task0 0.951 / task5 0.871)과 일치 검증.

실행: $HV2_HAB_PY -u experiments/e3_t2_check.py \
        --mode {smoke|trigger} --task {0|5}
종료 코드 0 = 통과. 비0 = 러너 중단.
"""
import argparse
import json
import os
from math import sqrt

from scipy.stats import binom

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALPHA = 0.01          # §1-2 단측 이항
SMOKE_MIN_SUCC = 7    # /10 — 검증 ③ gate (기대 S_V² 0.87~0.95에서 P(<7) ≤ ~2.4%)
# §4e 등재값 — 프로그래밍 취득값과의 일치 검증용.
# task0·5 = 점추정 S_V² (구 규칙, 소급 없음 — §5 2026-08-15).
# task6 = (Wilson 95% 하한)² (신규 일반 규칙 — 점추정 1.0 퇴화 방지 + draw 취약성 교훈).
REGISTERED_P0 = {0: 0.951, 5: 0.871, 6: 0.939}


def wilson_lower(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def base_sv2(task):
    if task in (0, 5):  # 구 규칙: E2 수집 점추정²
        p = os.path.join(HABIT2, "data", "e2", f"libero_object_task{task}_summary.json")
        sv = json.load(open(p))["S_V_cluster"]
        p0 = sv * sv
    else:  # 신규 일반 규칙 (task6+): 자체 수집 summary의 Wilson 95% 하한²
        p = os.path.join(HABIT2, "data", "e3", f"libero_object_task{task}_summary.json")
        s = json.load(open(p))
        k, n = s["n_success"], s["n_success"] + s["n_fail"]
        p0 = wilson_lower(k, n) ** 2
    assert abs(p0 - REGISTERED_P0[task]) < 0.005, (
        f"p₀ 재계산 {p0:.4f} != 등재값 {REGISTERED_P0[task]} (task{task})")
    return p0


def load_meta(h5_path):
    import h5py

    with h5py.File(h5_path, "r") as f:
        return json.loads(f["meta_json"][()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "trigger"], required=True)
    ap.add_argument("--task", type=int, required=True, choices=[0, 5, 6])
    args = ap.parse_args()

    cl = f"chained_libero_object_task{args.task}"

    if args.mode == "smoke":
        meta = load_meta(os.path.join(HABIT2, "data", "e3", "t2_smoke", f"{cl}.hdf5"))
        scored = [m for m in meta if m["outcome"] != "infra_error"]
        n_succ = sum(1 for m in scored if m["outcome"] == "success")
        n_stage2 = sum(1 for m in scored if m.get("stage") == 2)
        print(f"[smoke {cl}] n={len(scored)} stage2 도달={n_stage2} 최종 성공={n_succ} "
              f"(기대 S_V²={base_sv2(args.task):.3f})")
        for m in scored:
            print(f"  {m['uid']}: outcome={m['outcome']} stage={m.get('stage')} "
                  f"stage_steps={m.get('stage_steps')}")
        if len(scored) < 10 or n_succ < SMOKE_MIN_SUCC:
            print(f"[T2-SMOKE-FAIL] {cl}: 성공 {n_succ}/10 < {SMOKE_MIN_SUCC} — "
                  f"teacher stage-2 수행 미확인. 연구원 보고 (대체: Long 길이 층화)")
            raise SystemExit(1)
        print(f"[T2-SMOKE-PASS] {cl} ({n_succ}/10, stage2 도달 {n_stage2}/10)")

    else:  # trigger
        s = json.load(open(os.path.join(HABIT2, "data", "e3", f"{cl}_summary.json")))
        k, n = s["n_success"], s["n_success"] + s["n_fail"]
        p0 = base_sv2(args.task)
        pval = float(binom.cdf(k, n, p0))  # 단측: 기대치 대비 하방
        obs = k / max(n, 1)
        line = (f"{cl}: 관측 {k}/{n}={obs:.4f}, 기대 S_V²={p0:.4f}, "
                f"P(X≤{k}|n={n},p₀)={pval:.4f} (α={ALPHA})")
        if pval < ALPHA:
            print(f"[T2-TRIGGER-FAIL] {line} — 유의 미달, 수집 중단·보고 (§1-2)")
            raise SystemExit(1)
        print(f"[T2-TRIGGER-OK] {line}")


if __name__ == "__main__":
    main()
