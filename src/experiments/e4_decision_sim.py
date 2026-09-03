"""E4-3 갈래 판정 지원 시뮬레이션 (사전등록 아님 — 결정 재료 전용).

기존 산출물(수집 HDF5·known/novel 프레임)만으로:
  1. (a) 재구성안의 가정적 AUC — 주 novel = w_expand primary ∪ borrow primary
     (재샘플 제외; long은 w=0.02 미생성이라 제외 명기) — 클러스터별 + macro.
  2. 전 클러스터 점수 기하 (calib/known/경로별 평균±표준편차) — 판독 근거 전수화.
주의: (a) 채택 시 정식 재판정은 §5 등재 후 별도 실행 — 본 산출은 판정 문서 전용.

산출: results/e4/e4_decision_support.json
실행: hv2_hab python -u experiments/e4_decision_sim.py
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


def auc(neg, pos):
    from scipy.stats import rankdata

    if len(neg) == 0 or len(pos) == 0:
        return None
    x = np.concatenate([neg, pos])
    r = rankdata(x)
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main():
    import h5py

    ext = DinoFeatureExtractor()
    pca = SharedPCA.load(os.path.join(E4, "shared_pca_e4.joblib"))

    def embed(frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            out.append(ext.embed(list(frames[i:i + bs])))
        return np.concatenate(out) if out else np.zeros((0, 384))

    sup = {"note": "판정 지원 시뮬레이션 (사전등록 아님). 재구성 = w_expand primary ∪ borrow primary; "
                   "재샘플 제외. long은 w=0.02 novel 미생성 — 재구성 macro에서 제외 명기.",
           "clusters": {}, "geometry": {}}
    recomposed = []
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        ddir = "e2" if (suite, task) in E2_REUSE else "e3"
        with h5py.File(os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5"), "r") as f:
            col = [f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]
        feats = pca.transform(embed(col))
        gate = JurisdictionGate().fit(feats)
        perm = np.random.default_rng(0).permutation(len(feats))
        calib = feats[perm[len(feats) // 2:]]
        kz = np.load(os.path.join(E4, "known_frames", f"{cl}.npz"))
        known = np.array([gate.score(x) for x in pca.transform(embed(kz["frames"]))])

        geo = {"calib": None, "known": [round(float(known.mean()), 3), round(float(known.std()), 3)],
               "q": round(gate.q, 3)}
        cd = np.array([gate.score(x) for x in calib])
        geo["calib"] = [round(float(cd.mean()), 3), round(float(cd.std()), 3)]
        pool = []
        for p in sorted(glob.glob(os.path.join(E4, "novel_frames", f"{cl}__*.npz"))):
            key = os.path.basename(p)[:-4].split("__", 1)[1]
            nz = np.load(p)
            if len(nz["frames"]) == 0:
                continue
            sc = np.array([gate.score(x) for x in pca.transform(embed(nz["frames"]))])
            geo[key] = [round(float(sc.mean()), 3), round(float(sc.std()), 3)]
            if key in ("w_expand__primary", "borrow__primary"):
                pool.append(sc)
        sup["geometry"][cl] = geo
        if pool:
            a = auc(known, np.concatenate(pool))
            sup["clusters"][cl] = {"recomposed_auc": round(a, 4),
                                   "n_novel": int(sum(len(s) for s in pool))}
            recomposed.append(a)
        else:
            sup["clusters"][cl] = {"recomposed_auc": None, "reason": "long — w=0.02 novel 미생성"}
        print(f"[SIM] {cl}: recomposed={sup['clusters'][cl].get('recomposed_auc')}", flush=True)

    sup["macro_recomposed_excl_long"] = round(float(np.mean(recomposed)), 4)
    sup["n_clusters_in_macro"] = len(recomposed)
    sup["n_below_075"] = sum(1 for a in recomposed if a < 0.75)
    with open(os.path.join(E4, "e4_decision_support.json"), "w") as f:
        json.dump(sup, f, indent=2, ensure_ascii=False)
    print(f"[E4-DECISION-SIM] 재구성 macro(장외 long 제외)={sup['macro_recomposed_excl_long']} "
          f"(n={sup['n_clusters_in_macro']}, <0.75: {sup['n_below_075']})")


if __name__ == "__main__":
    main()
