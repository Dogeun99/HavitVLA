"""작업공간 실측 (§5 등재 2026-08-16) — w*를 도달 영역 대비 %로 환산하기 위한 기준.

CPU 전용 (렌더러 없음):
 1. 40 태스크 공식 init state 50개에서 free 물체 xy → 태스크별/스위트별/전체 bbox.
 2. teacher 성공 배치 분포: E2/E3 수집 meta의 **성공 에피소드 spec**(base_idx·w·noise)을
    재구성해 목표 물체 xy 산포 → 99% 분위 + 볼록껍질 면적 = **검증된 도달 영역** 실측 근사.
 3. 로봇 base·테이블 상판 xy 경계: 씬 모델(geom)에서 취득 — **Franka 공칭 반경 문헌값 미사용**.
 4. 산출: results/e4/workspace_extent.json (+ 플롯은 별도 스크립트).
판독: w*(0.02–0.06 m)가 검증 도달 영역의 몇 %인지 수치 명시.

실행: hv2_hab python -u experiments/workspace_extent.py
"""
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")

OUT = os.path.join(HABIT2, "results", "e4", "workspace_extent.json")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}


def main():
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    out = {"note": "작업공간 실측 (§5). 렌더 없이 모델·init state 파싱. "
                   "Franka 공칭 반경 문헌값 미사용 — 검증 도달 영역은 teacher 성공 배치 실측.",
           "per_task": {}, "per_suite": {}, "verified_reach": {}, "scene": {}}
    all_xy = []
    for suite in SUITES:
        bench = benchmark.get_benchmark_dict()[suite]()
        suite_xy = []
        for tid in range(bench.n_tasks):
            task = bench.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False,
                             has_offscreen_renderer=False, has_renderer=False)
            sim = env.env.sim
            init = bench.get_task_init_states(tid)
            nq, nv = sim.model.nq, sim.model.nv
            off = init.shape[1] - nq - nv
            free = [sim.model.jnt_qposadr[j] for j in range(sim.model.njnt)
                    if sim.model.jnt_type[j] == 0]
            xy = np.array([[st[off + a], st[off + a + 1]] for st in init for a in free])
            cl = f"{suite}_task{tid}"
            out["per_task"][cl] = {"n_states": len(init), "n_free": len(free),
                                   "x_range": [round(float(xy[:, 0].min()), 4),
                                               round(float(xy[:, 0].max()), 4)],
                                   "y_range": [round(float(xy[:, 1].min()), 4),
                                               round(float(xy[:, 1].max()), 4)]}
            suite_xy.append(xy)
            if suite == "libero_object" and tid == 0:  # 대표 씬에서 테이블·base 취득
                names = list(sim.model.geom_names)
                tgeom = [n for n in names if n and "table" in n.lower()]
                if tgeom:
                    gid = sim.model.geom_name2id(tgeom[0])
                    size = sim.model.geom_size[gid]
                    pos = sim.model.geom_pos[gid]
                    out["scene"]["table_geom"] = tgeom[0]
                    out["scene"]["table_xy_halfsize"] = [round(float(size[0]), 4),
                                                         round(float(size[1]), 4)]
                    out["scene"]["table_center_xy"] = [round(float(pos[0]), 4),
                                                       round(float(pos[1]), 4)]
                try:
                    bid = sim.model.body_name2id("robot0_base")
                    out["scene"]["robot_base_xy"] = [round(float(v), 4)
                                                     for v in sim.model.body_pos[bid][:2]]
                except Exception:
                    out["scene"]["robot_base_xy"] = None
            env.close()
            print(f"[WS] {cl}: free={len(free)} x{out['per_task'][cl]['x_range']} "
                  f"y{out['per_task'][cl]['y_range']}", flush=True)
        s_xy = np.concatenate(suite_xy)
        all_xy.append(s_xy)
        out["per_suite"][suite] = {
            "x_range": [round(float(s_xy[:, 0].min()), 4), round(float(s_xy[:, 0].max()), 4)],
            "y_range": [round(float(s_xy[:, 1].min()), 4), round(float(s_xy[:, 1].max()), 4)],
            "x_span": round(float(s_xy[:, 0].ptp()), 4), "y_span": round(float(s_xy[:, 1].ptp()), 4)}
    A = np.concatenate(all_xy)
    out["official_placement_bbox"] = {
        "x_range": [round(float(A[:, 0].min()), 4), round(float(A[:, 0].max()), 4)],
        "y_range": [round(float(A[:, 1].min()), 4), round(float(A[:, 1].max()), 4)],
        "x_span": round(float(A[:, 0].ptp()), 4), "y_span": round(float(A[:, 1].ptp()), 4),
        "n_points": int(len(A))}

    # --- 2. teacher 성공 배치 분포 (수집 meta의 성공 spec 재구성)
    from envs.libero_env import LiberoEpisodeEnv

    succ_xy = []
    for suite in SUITES:
        bench = benchmark.get_benchmark_dict()[suite]()
        for tid in range(bench.n_tasks):
            cl = f"{suite}_task{tid}"
            ddir = "e2" if (suite, tid) in E2_REUSE else "e3"
            p = os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5")
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                meta = json.loads(f["meta_json"][()])
            env = LiberoEpisodeEnv(suite, tid)
            env.perturbed_init_state(0, 0.01, np.random.default_rng(0))  # 상수 초기화
            for m in meta:
                if m["outcome"] != "success":
                    continue
                st = env.perturbed_init_state(m["base_idx"], m["w"],
                                              np.random.default_rng(m["noise_seed"]))
                off = env._time_offset
                for a in env._free_adrs:
                    succ_xy.append([st[off + a], st[off + a + 1]])
            env.close()
            print(f"[WS-SUCC] {cl} 누적 {len(succ_xy)}", flush=True)
    S = np.array(succ_xy)
    from scipy.spatial import ConvexHull

    hull = ConvexHull(S)
    q = {ax: [round(float(np.quantile(S[:, i], 0.005)), 4),
              round(float(np.quantile(S[:, i], 0.995)), 4)] for i, ax in enumerate("xy")}
    out["verified_reach"] = {
        "n_points": int(len(S)), "quantile99_x": q["x"], "quantile99_y": q["y"],
        "q99_span_x": round(q["x"][1] - q["x"][0], 4), "q99_span_y": round(q["y"][1] - q["y"][0], 4),
        "convex_hull_area_m2": round(float(hull.volume), 5),
        "equivalent_radius_m": round(float(np.sqrt(hull.volume / np.pi)), 4),
        "source": "E2/E3 teacher 성공 에피소드 spec 재구성 (문헌 반경 미사용)"}

    # --- 판독: w*를 도달 영역 대비 %로
    vr = out["verified_reach"]
    ratios = {}
    for w in (0.01, 0.02, 0.04, 0.06, 0.08):
        d = 2 * w  # 섭동 폭 uniform(-w,w) → 변이 지름
        ratios[str(w)] = {
            "perturbation_diameter_m": round(d, 4),
            "pct_of_q99_span_x": round(100 * d / vr["q99_span_x"], 1),
            "pct_of_q99_span_y": round(100 * d / vr["q99_span_y"], 1),
            "pct_of_equiv_diameter": round(100 * d / (2 * vr["equivalent_radius_m"]), 1),
            "area_pct_of_hull": round(100 * (np.pi * w ** 2) / vr["convex_hull_area_m2"], 2)}
    out["w_vs_reach"] = ratios
    # 플롯용 좌표 (서브샘플 + hull 정점) — 그림 재현이 JSON 단일 진입점에서 되도록
    rng = np.random.default_rng(0)
    out["plot_data"] = {
        "official_xy_sample": A[rng.choice(len(A), min(3000, len(A)), replace=False)].round(4).tolist(),
        "verified_xy_sample": S[rng.choice(len(S), min(3000, len(S)), replace=False)].round(4).tolist(),
        "hull_vertices_xy": S[hull.vertices].round(4).tolist(),
        "note": "official = 40 태스크 공식 init state free 물체 xy / verified = teacher 성공 spec 재구성"}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[WS-PASS] 공식 배치 bbox x{out['official_placement_bbox']['x_span']}m × "
          f"y{out['official_placement_bbox']['y_span']}m | 검증 도달: q99 "
          f"{vr['q99_span_x']}×{vr['q99_span_y']}m, hull {vr['convex_hull_area_m2']}m², "
          f"등가반경 {vr['equivalent_radius_m']}m | w=0.04 → 도달 지름의 "
          f"{ratios['0.04']['pct_of_equiv_diameter']}%")


if __name__ == "__main__":
    main()
