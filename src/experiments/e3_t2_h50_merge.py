"""C-T2 held-out 50 병합 (§4e·§5 이력 — 결과 판독 전 등재; R1b 확장).

대상:
  1) chained {task0, task6}: 기본 곡선(1–20) + 보충(21–50) 병합 → canonical
     `results/e3/{cluster}_curve.json` 교체 (원본 20은 `_curve_h20.json` 보존).
  2) **task6 싱글** (R1b — 곱 기준선 T1 참조): E3 표준 20 곡선 + 보충(21–50) 병합 →
     **신규 파일** `libero_object_task6_curve_h50.json` (표준 20 곡선은 불변 —
     출력 경로 유일성 렌즈: E3 집계의 paired 20과 참조 50을 분리 유지).
uid disjoint·50 완결 검증. 비맹검 규약: 병합 전 부분 수치는 별도 보고하지 않는다.
"""
import json
import os
import shutil

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRID = [10, 20, 40, 80]
CHAINED = ["chained_libero_object_task0", "chained_libero_object_task5"]  # α 판정 복원
SINGLE_H50 = {}  # task6 싱글 경로는 α 판정으로 회귀 제거 (§5 이력 참조)


def merge(base_p, supp_p, out_p, cl, preserve_h20=None):
    base = json.load(open(base_p))
    supp = json.load(open(supp_p))
    assert supp.get("heldout_start") == 20, f"{cl}: 보충 곡선 start != 20"
    merged = {"cluster": cl, "n_heldout": 50, "merged_from": ["h20(1-20)", "supp(21-50)"], "curve": []}
    base_by_n = {c["n"]: c for c in base["curve"] if "per_episode" in c}
    supp_by_n = {c["n"]: c for c in supp["curve"] if "per_episode" in c}
    for n in GRID:
        b, s = base_by_n.get(n), supp_by_n.get(n)
        assert b and s, f"{cl} n={n}: 곡선 누락 (base={bool(b)}, supp={bool(s)})"
        ub = {e["uid"] for e in b["per_episode"]}
        us = {e["uid"] for e in s["per_episode"]}
        assert not (ub & us), f"{cl} n={n}: uid 중복 {sorted(ub & us)[:3]}"
        eps = b["per_episode"] + s["per_episode"]
        assert len(eps) == 50, f"{cl} n={n}: 병합 {len(eps)} != 50"
        scored = [e for e in eps if e["outcome"] != "infra_error"]
        k = sum(1 for e in scored if e["outcome"] == "success")
        merged["curve"].append({
            "n": n, "ckpt": b.get("ckpt"), "n_eval": len(scored), "n_success": k,
            "n_infra_error": len(eps) - len(scored),
            "s_hat": round(k / len(scored), 4) if scored else None,
            "per_episode": eps,
        })
    if preserve_h20 and not os.path.exists(preserve_h20):
        shutil.copy(base_p, preserve_h20)
    with open(out_p, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[H50-MERGE] {cl}: 50 ep 병합 -> {os.path.relpath(out_p, HABIT2)}")


def main():
    e3 = os.path.join(HABIT2, "results", "e3")
    supp_dir = os.path.join(e3, "t2_h50")
    n_done = 0
    for cl in CHAINED:
        base_p = os.path.join(e3, f"{cl}_curve.json")
        supp_p = os.path.join(supp_dir, f"{cl}_h50supp_curve.json")
        if not (os.path.exists(base_p) and os.path.exists(supp_p)):
            print(f"[H50-SKIP] {cl}: 산출물 미비 (base={os.path.exists(base_p)}, supp={os.path.exists(supp_p)})")
            continue
        merge(base_p, supp_p, os.path.join(e3, f"{cl}_curve.json"), cl,
              preserve_h20=os.path.join(e3, f"{cl}_curve_h20.json"))
        n_done += 1
    for cl, out_name in SINGLE_H50.items():
        base_p = os.path.join(e3, f"{cl}_curve.json")
        supp_p = os.path.join(supp_dir, f"{cl}_h50supp_curve.json")
        if not (os.path.exists(base_p) and os.path.exists(supp_p)):
            print(f"[H50-SKIP] {cl}(single): 산출물 미비")
            continue
        merge(base_p, supp_p, os.path.join(e3, out_name), cl, preserve_h20=None)  # 표준 20 불변
        n_done += 1
    print(f"[H50-MERGE-PASS] {n_done}건")


if __name__ == "__main__":
    main()
