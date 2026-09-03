"""E4-3: 관할 오프라인 파일럿 AUC (연구원 지시 2026-08-16 §4 — 사전등재 임계 0.75).

- fit: 클러스터별 수집 성공 I₀ 프레임 (HDF5 저장본, base 0–39) — μ/Σ + calib 분위수
  (gates/two_stage.JurisdictionGate, 결정적 분할). 공용 PCA도 수집 풀만 (features.py 규율).
- known = held-out 프레임만 (E4-1 덤프, base 40–49) — in-sample 금지.
- novel = E4-2 산출 (주 = base 40–49 정합: w_expand primary + resample + borrow primary;
  부차 = base 0–39 variants 병행 보고).
- AUC: 클러스터별(known vs novel-주) + 통합. **판정 = macro 평균 AUC ≥ 0.75**
  (micro pooled는 클러스터 간 거리 스케일 비교 불가 캐비앳과 함께 참조 보고).
산출: results/e4/e4_pilot_auc.json + [E4AUC-GO|E4AUC-REDUCE]
실행: hv2_hab python -u experiments/e4_pilot_auc.py
"""
import glob
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))

from gates.features import DinoFeatureExtractor, SharedPCA  # noqa: E402
from gates.two_stage import JurisdictionGate  # noqa: E402

E4 = os.path.join(HABIT2, "results", "e4")
STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}
THRESH = 0.75  # preregistration §1


def auc(neg_scores, pos_scores):
    """AUC = P(novel score > known score) — Mann-Whitney 순위 기반."""
    from scipy.stats import rankdata

    x = np.concatenate([neg_scores, pos_scores])
    r = rankdata(x)
    n0, n1 = len(neg_scores), len(pos_scores)
    if n0 == 0 or n1 == 0:
        return None
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def collection_frames(cl, suite, task):
    import h5py

    ddir = "e2" if (suite, task) in E2_REUSE else "e3"
    with h5py.File(os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5"), "r") as f:
        return [f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]


def main():
    ext = DinoFeatureExtractor()

    def embed_all(frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            out.append(ext.embed(list(frames[i:i + bs])))
        return np.concatenate(out) if out else np.zeros((0, 384))

    # 1) 수집 프레임 임베딩 + 공용 PCA (수집 풀만 — features.py 규율)
    col_embeds = {}
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        col_embeds[cl] = embed_all(collection_frames(cl, suite, task))
        print(f"[AUC] embed collection {cl} n={len(col_embeds[cl])}", flush=True)
    pca = SharedPCA().fit(np.concatenate(list(col_embeds.values())))
    pca.save(os.path.join(E4, "shared_pca_e4.joblib"))

    report = {"threshold": THRESH, "decision_basis": "macro mean AUC (primary novel, base 40-49 정합)",
              "clusters": {}, "paths_macro": {}}
    macro_primary, macro_secondary, micro_scores = [], [], {"known": [], "novel": []}
    path_pool = {}
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        gate = JurisdictionGate().fit(pca.transform(col_embeds[cl]))
        kz = np.load(os.path.join(E4, "known_frames", f"{cl}.npz"))
        known_scores = np.array([gate.score(f) for f in pca.transform(embed_all(kz["frames"]))])
        entry = {"n_known": len(known_scores), "paths": {}}
        prim_scores, sec_scores = [], []
        for p in sorted(glob.glob(os.path.join(E4, "novel_frames", f"{cl}__*.npz"))):
            key = os.path.basename(p)[:-4]
            _, path, variant = key.split("__")
            nz = np.load(p)
            if len(nz["frames"]) == 0:
                continue
            sc = np.array([gate.score(f) for f in pca.transform(embed_all(nz["frames"]))])
            a = auc(known_scores, sc)
            entry["paths"][f"{path}__{variant}"] = {"n": len(sc), "auc": round(a, 4) if a else None}
            path_pool.setdefault(f"{path}__{variant}", []).append(a)
            if variant in ("primary", "single"):
                prim_scores.append(sc)
            else:
                sec_scores.append(sc)
        if prim_scores:
            pooled = np.concatenate(prim_scores)
            a = auc(known_scores, pooled)
            entry["auc_primary"] = round(a, 4)
            macro_primary.append(a)
            micro_scores["known"].append(known_scores)
            micro_scores["novel"].append(pooled)
        if sec_scores:
            a2 = auc(known_scores, np.concatenate(sec_scores))
            entry["auc_secondary_base0_39"] = round(a2, 4)
            macro_secondary.append(a2)
        report["clusters"][cl] = entry
        print(f"[AUC] {cl}: primary={entry.get('auc_primary')} secondary={entry.get('auc_secondary_base0_39')}", flush=True)

    report["macro_auc_primary"] = round(float(np.mean(macro_primary)), 4)
    report["macro_auc_secondary"] = round(float(np.mean(macro_secondary)), 4) if macro_secondary else None
    report["micro_auc_primary_caveat_scale"] = round(
        auc(np.concatenate(micro_scores["known"]), np.concatenate(micro_scores["novel"])), 4)
    report["n_clusters_below_threshold"] = sum(1 for a in macro_primary if a < THRESH)
    report["paths_macro"] = {k: round(float(np.mean([x for x in v if x is not None])), 4)
                             for k, v in path_pool.items()}
    go = report["macro_auc_primary"] >= THRESH
    report["verdict"] = "GO" if go else "REDUCE(E5 성숙도 단독 — §3 축소 경로)"
    with open(os.path.join(E4, "e4_pilot_auc.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[{'E4AUC-GO' if go else 'E4AUC-REDUCE'}] macro={report['macro_auc_primary']} "
          f"(임계 {THRESH}) | secondary={report['macro_auc_secondary']} | "
          f"micro={report['micro_auc_primary_caveat_scale']} | <임계 클러스터 {report['n_clusters_below_threshold']}")


if __name__ == "__main__":
    main()
