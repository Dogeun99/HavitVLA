"""§4f 공변량 단일 테이블 산출 (통합 지시서 §4) — H2-L′ 회귀 원자료.

클러스터별 3종 공변량을 사전등록된 단일 진입점들에서 프로그래밍 방식으로만 취합:
  - free_joints: results/e3/free_joints_census.json (40 태스크 전수조사)
  - S_V_cluster: data/{e2,e3}/{cluster}_summary.json (수집 실측)
  - median_len:  수집 HDF5 meta_json의 성공 에피소드 steps 중앙값 (§4f 등재 정의)

산출: results/e3/covariates.json. chained 2 클러스터는 수집 전이면 pending으로 명시
(부분 데이터를 완결로 오인하지 않도록 — 재실행 시 자동 편입).

실행: $HV2_HAB_PY -u experiments/e3_covariates.py
"""
import json
import os
import statistics

import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "e3", "covariates.json")
CENSUS = os.path.join(ROOT, "results", "e3", "free_joints_census.json")

# 표준 25 클러스터 (preregistration §4e; object task0/5는 E2 수집 재사용)
STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}
# α 판정 (§5 2026-08-15): 체인 구성 = task0 + task5 복원 (diag5/5b — 실행기 아티팩트 재귀속)
CHAINED = ["chained_libero_object_task0", "chained_libero_object_task5"]


def cluster_row(suite, task, census):
    cl = f"{suite}_task{task}"
    ddir = "e2" if (suite, task) in E2_REUSE else "e3"
    summ_p = os.path.join(ROOT, "data", ddir, f"{cl}_summary.json")
    h5_p = os.path.join(ROOT, "data", ddir, f"{cl}.hdf5")
    if not (os.path.exists(summ_p) and os.path.exists(h5_p)):
        return cl, None
    summ = json.load(open(summ_p))
    with h5py.File(h5_p, "r") as f:
        meta = json.loads(f["meta_json"][()])
    succ_steps = [m["steps"] for m in meta if m["outcome"] == "success"]
    return cl, {
        "suite": suite,
        "task_id": task,
        "free_joints": census["suites"][suite][str(task)]["n_free_joints"],
        "S_V_cluster": summ["S_V_cluster"],
        "n_success": summ["n_success"],
        "median_len_success": statistics.median(succ_steps),
        "data_dir": ddir,
    }


def main():
    census = json.load(open(CENSUS))
    out = {
        "note": "H2-L′ 회귀 공변량 (preregistration §4f). free_joints=census, "
        "S_V_cluster=수집 summary, median_len=성공 에피소드 steps 중앙값.",
        "clusters": {},
        "pending": [],
        "status": "FAIL",
    }
    for suite, task in STANDARD:
        cl, row = cluster_row(suite, task, census)
        if row is None:
            out["pending"].append(cl)
        else:
            out["clusters"][cl] = row
    # chained: 수집 완료 시 편입 (free_joints·S_V는 원 태스크와 동일 씬 — 연쇄는 T 조작만)
    for cl in CHAINED:
        summ_p = os.path.join(ROOT, "data", "e3", f"{cl}_summary.json")
        h5_p = os.path.join(ROOT, "data", "e3", f"{cl}.hdf5")
        if not (os.path.exists(summ_p) and os.path.exists(h5_p)):
            out["pending"].append(cl)
            continue
        base_task = int(cl.rsplit("task", 1)[1])
        summ = json.load(open(summ_p))
        with h5py.File(h5_p, "r") as f:
            meta = json.loads(f["meta_json"][()])
        succ_steps = [m["steps"] for m in meta if m["outcome"] == "success"]
        out["clusters"][cl] = {
            "suite": "libero_object",
            "task_id": base_task,
            "chained": True,
            "free_joints": census["suites"]["libero_object"][str(base_task)]["n_free_joints"],
            "S_V_cluster": summ["S_V_cluster"],
            "n_success": summ["n_success"],
            "median_len_success": statistics.median(succ_steps),
            "data_dir": "e3",
        }

    n = len(out["clusters"])
    out["status"] = "COMPLETE(27)" if n == 27 and not out["pending"] else f"PARTIAL({n})"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for cl, r in sorted(out["clusters"].items()):
        print(f"{cl}: free={r['free_joints']} S_V={r['S_V_cluster']:.3f} "
              f"med_len={r['median_len_success']}")
    print(f"[COVARIATES] status={out['status']} pending={out['pending']} -> {OUT}")


if __name__ == "__main__":
    main()
