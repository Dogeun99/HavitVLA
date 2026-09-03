"""§12 FAMILIARITY — dependency audit 후 habit 의존 값만 재계산.

의존성 판정 원칙: 값이 **RGB-only habit의 성공 여부에 의존하는가**로 가른다.
  · 재계산 필요 = habit rollout 결과가 들어가는 값 (역량 지도, 폭별 습관 성공률, w*)
  · 재사용 가능 = 표현/에피소드 조건만으로 결정되는 값 (DINOv2 특징, Mahalanobis·kNN 분위수,
    teacher L32 히든 — teacher는 §1 동결). 재사용한 값은 source를 반드시 기록한다.

산출 (06_familiarity/):
  FAMILIARITY_DEPENDENCY_AUDIT.json · competence_map_rgb_only.json ·
  FAMILIARITY_EPISODES.csv · FAMILIARITY_SUMMARY.json
실행: hv2_hab python -u experiments/rgb_only_rerun/analyze_familiarity.py
마커: [FAMILIARITY-DONE]
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import PY_HAB, ROOT  # noqa: E402

OUT = f"{ROOT}/06_familiarity"
CKROOT = "checkpoints/rgb_only_rerun/batch"
MAP_OUT = f"{OUT}/competence_map_rgb_only.json"
FRAG = f"{OUT}/frag"
W_ID = 0.01
USABLE = {"libero_spatial": 0.04, "libero_object": 0.04, "libero_goal": 0.04, "libero_10": 0.02}

# --- §12 의존성 대장 (metric_name / depends_on_habit_modality / recomputed / source)
AUDIT = [
    {"metric_name": "competence_map.habit_success_rate_by_w",
     "depends_on_habit_modality": True, "recomputed": True,
     "source": "experiments/e4r_competence_map.py --ckpt-root checkpoints/rgb_only_rerun/batch",
     "why": "RGB-only habit(n=80) rollout 결과가 직접 들어간다."},
    {"metric_name": "competence_map.w_star",
     "depends_on_habit_modality": True, "recomputed": True,
     "source": "동일 (재계산 산출물에서 판독 규칙 그대로 적용)",
     "why": "습관 성공률이 0.8 아래로 내려가는 첫 폭 — 습관 성능의 함수."},
    {"metric_name": "competence_map.gate_score / gate_reject_rate",
     "depends_on_habit_modality": False, "recomputed": True,
     "source": "동일 실행의 부산물 (DINOv2 + 공용 PCA + Mahalanobis)",
     "why": "초기 프레임 표현만의 함수라 modality 무관이지만, 같은 실행에서 무료로 나오므로 "
            "재사용 대신 재계산해 에피소드 단위 정합을 유지한다."},
    {"metric_name": "competence_map.physical_reject_rate",
     "depends_on_habit_modality": False, "recomputed": True,
     "source": "동일 실행의 부산물 (E0-6 settled 유효성 검사)",
     "why": "환경·섭동만의 함수."},
    {"metric_name": "jurisdiction_AUC (known vs novel)",
     "depends_on_habit_modality": False, "recomputed": False,
     "source": "results/e4/e4_pilot_auc.json",
     "why": "DINOv2 특징과 에피소드 조건만으로 결정된다. habit이 관여하지 않는다."},
    {"metric_name": "kNN scorer (k=5,10) AUC·FR·quantile",
     "depends_on_habit_modality": False, "recomputed": False,
     "source": "results/e4/e4_knn_scorer.json",
     "why": "동일 특징·동일 conformal 절차. habit 무관."},
    {"metric_name": "teacher hidden L32 scorer",
     "depends_on_habit_modality": False, "recomputed": False,
     "source": "results/e4/e4_hidden_scorer.json",
     "why": "teacher(OpenVLA-OFT)는 §1 ABSOLUTE FREEZE 대상이며 habit 경로가 관여하지 않는다."},
    {"metric_name": "scorer comparison table",
     "depends_on_habit_modality": False, "recomputed": False,
     "source": "results/e4/e4_scorer_table.json",
     "why": "위 세 scorer의 집계."},
    {"metric_name": "shared PCA (32) basis",
     "depends_on_habit_modality": False, "recomputed": False,
     "source": "results/e4/shared_pca_e4.joblib",
     "why": "수집 풀 fit — teacher 궤적에서만 산출. §1 동결."},
]


def knn_scores(fit_feats, query, ks=(5, 10)):
    """fit-half 내 k-최근접 평균 거리 (e4_knn_scorer와 동일 정의)."""
    d = np.linalg.norm(fit_feats[None, :, :] - query[:, None, :], axis=2)
    d.sort(axis=1)
    return {k: d[:, :k].mean(1) for k in ks}


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. habit 의존 값 재계산: 역량 지도
    if not os.path.exists(MAP_OUT):
        print("[FAMILIARITY] 역량 지도 재계산 (RGB-only n=80 체크포인트)", flush=True)
        r = subprocess.run([PY_HAB, "-u", "experiments/e4r_competence_map.py",
                            "--ckpt-root", CKROOT, "--out", MAP_OUT, "--frag-dir", FRAG],
                           cwd=HABIT2)
        if r.returncode != 0 or not os.path.exists(MAP_OUT):
            print("[FAMILIARITY-FAIL] 역량 지도 재계산 실패")
            sys.exit(1)
    cm = json.load(open(MAP_OUT))

    # ---- 2. 에피소드 단위 CSV (+ kNN 점수는 보존된 PCA 특징에서 사후 산출)
    from gates.features import DinoFeatureExtractor, SharedPCA
    import h5py
    pca = SharedPCA.load("results/e4/shared_pca_e4.joblib")
    ext = None
    rows = []
    for cl, entry in cm["clusters"].items():
        suite = cl.rsplit("_task", 1)[0]
        # fit-half = 수집 성공 초기 프레임 (kNN 기준 집합) — habit 무관, 동결 자산
        ddir = "e2" if cl in ("libero_object_task0", "libero_object_task5") else "e3"
        h5p = f"data/{ddir}/{cl}.hdf5"
        fit = None
        if os.path.exists(h5p):
            if ext is None:
                ext = DinoFeatureExtractor()
            with h5py.File(h5p, "r") as f:
                frames = [f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]
            fit = pca.transform(np.concatenate([ext.embed(frames[i:i + 64])
                                                for i in range(0, len(frames), 64)]))
        for w, blk in entry["by_w"].items():
            eps = [e for e in blk["episodes"] if e.get("valid") and "pca32" in e]
            if not eps:
                continue
            q = np.asarray([e["pca32"] for e in eps], float)
            kn = knn_scores(fit, q) if fit is not None else {5: [None] * len(eps),
                                                             10: [None] * len(eps)}
            wf = float(w)
            for e, k5, k10 in zip(eps, kn[5], kn[10]):
                rows.append({
                    "cluster_id": cl, "suite": suite,
                    "perturbation_width": wf,
                    "id_ood_label": ("ID" if wf <= W_ID else
                                     ("boundary" if wf <= USABLE[suite] else "OOD")),
                    "within_usable_w_max": int(wf <= USABLE[suite]),
                    "episode_uid": e["uid"], "base_idx": e.get("base_idx"),
                    "episode_seed": e.get("seed"), "noise_seed": e.get("noise_seed"),
                    "habit_success": int(bool(e["success"])), "steps": e.get("steps"),
                    "familiarity_score_mahalanobis": e["gate_score"],
                    "familiarity_reject": int(bool(e["gate_rejected"])),
                    "jurisdiction_q": entry["q"],
                    "knn_score_k5": round(float(k5), 5) if k5 is not None else None,
                    "knn_score_k10": round(float(k10), 5) if k10 is not None else None,
                    "teacher_hidden_score": None,   # 아래 audit 참조 — 6셀 범위 재사용 자산
                })
    if rows:
        with open(f"{OUT}/FAMILIARITY_EPISODES.csv", "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w_.writeheader()
            w_.writerows(rows)

    # ---- 3. 요약 (숫자만)
    def auc(neg, pos):
        from scipy.stats import rankdata
        if not len(neg) or not len(pos):
            return None
        x = np.concatenate([neg, pos])
        r = rankdata(x)
        n1, n0 = len(pos), len(neg)
        return round(float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)), 4)

    summ = {"run_id": os.path.basename(ROOT), "modality": "rgb_only",
            "habit_ckpt_root": cm.get("habit_ckpt_root"),
            "n_episodes": len(rows),
            "competence_summary": cm.get("summary"),
            "note": "그림 없음. 아래 값은 재구성용 숫자다."}
    if rows:
        fail = [r["familiarity_score_mahalanobis"] for r in rows if not r["habit_success"]]
        succ = [r["familiarity_score_mahalanobis"] for r in rows if r["habit_success"]]
        summ["score_vs_habit_success"] = {
            "auc_mahalanobis_predicts_failure": auc(succ, fail),
            "n_success": len(succ), "n_fail": len(fail),
            "mean_score_success": round(float(np.mean(succ)), 4) if succ else None,
            "mean_score_fail": round(float(np.mean(fail)), 4) if fail else None,
            "definition": "AUC = P(score(fail) > score(success)) — 높을수록 친숙도 점수가 "
                          "습관 실패를 잘 가려낸다."}
        for k in (5, 10):
            f_ = [r[f"knn_score_k{k}"] for r in rows if not r["habit_success"]
                  and r[f"knn_score_k{k}"] is not None]
            s_ = [r[f"knn_score_k{k}"] for r in rows if r["habit_success"]
                  and r[f"knn_score_k{k}"] is not None]
            summ["score_vs_habit_success"][f"auc_knn_k{k}_predicts_failure"] = auc(s_, f_)
        by_w = {}
        for r in rows:
            by_w.setdefault(r["perturbation_width"], []).append(r)
        summ["by_width"] = {str(w): {
            "n": len(v),
            "habit_success_rate": round(float(np.mean([x["habit_success"] for x in v])), 4),
            "familiarity_reject_rate": round(float(np.mean([x["familiarity_reject"] for x in v])), 4),
            "mean_mahalanobis": round(float(np.mean(
                [x["familiarity_score_mahalanobis"] for x in v])), 4),
        } for w, v in sorted(by_w.items())}

    # ---- 4. 재사용 자산 스냅샷 (source 기록)
    reuse = {}
    for name, path in (("jurisdiction_auc", "results/e4/e4_pilot_auc.json"),
                       ("knn_scorer", "results/e4/e4_knn_scorer.json"),
                       ("hidden_scorer", "results/e4/e4_hidden_scorer.json"),
                       ("scorer_table", "results/e4/e4_scorer_table.json")):
        if os.path.exists(path):
            import hashlib
            reuse[name] = {"source": path,
                           "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                           "reused_unchanged": True}
    summ["reused_artifacts"] = reuse

    json.dump({"run_id": os.path.basename(ROOT),
               "principle": "값이 RGB-only habit 성공에 의존하는가로 재계산 여부를 가른다.",
               "metrics": AUDIT, "reused_artifacts": reuse},
              open(f"{OUT}/FAMILIARITY_DEPENDENCY_AUDIT.json", "w"), indent=1, ensure_ascii=False)
    json.dump(summ, open(f"{OUT}/FAMILIARITY_SUMMARY.json", "w"), indent=1, ensure_ascii=False)
    print(f"[FAMILIARITY-DONE] episodes={len(rows)} recomputed="
          f"{sum(1 for a in AUDIT if a['recomputed'])} reused={len(reuse)} "
          f"w_star={cm.get('summary', {}).get('w_star_by_cluster')}")


if __name__ == "__main__":
    main()
