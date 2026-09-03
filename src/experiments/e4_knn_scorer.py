"""E4 scorer 표 6행 — kNN 관할 (연구원 추가 지시 2026-08-16, 기존 실행 불간섭·사후 병합).

동일 특징(DINOv2 + 공용 PCA32) · 동일 conformal 절차(결정적 반반 분할, 유한표본 분위수):
  score(x) = fit-half 내 k-최근접 평균 거리 (k ∈ {5, 10})
  q = calib-half 점수의 ⌈(n+1)(1−α_j)⌉ 순서통계량 / FR = known 전량 기준
  + 보정/평가 분할 준수: known 50:50(seed 0) 재보정 FR 병기 (행5와 동일 절차)
산출: results/e4/e4_knn_scorer.json → 표 병합용
실행: hv2_hab python -u experiments/e4_knn_scorer.py
"""
import glob
import json
import os
import sys
from math import ceil

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))

from gates.features import DinoFeatureExtractor, SharedPCA  # noqa: E402
from gates.two_stage import ALPHA_J  # noqa: E402

E4 = os.path.join(HABIT2, "results", "e4")
STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}
PRIMARY_KEYS = ("w_expand__primary", "borrow__primary")
KS = (5, 10)


def auc(neg, pos):
    from scipy.stats import rankdata

    x = np.concatenate([neg, pos])
    r = rankdata(x)
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def knn_score(X_fit, Q, k):
    d2 = ((Q[:, None, :] - X_fit[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.sort(d2, axis=1)[:, :k]).mean(1)


def main():
    import h5py

    ext = DinoFeatureExtractor()
    pca = SharedPCA.load(os.path.join(E4, "shared_pca_e4.joblib"))

    def embed(frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            out.append(ext.embed(list(frames[i:i + bs])))
        return np.concatenate(out) if out else np.zeros((0, 384))

    out = {"note": "6행 kNN 관할 — 동일 특징·동일 conformal(결정적 반반 + 유한표본 분위수), "
                   "known 50:50 재보정 FR 병기 (행5 절차)",
           "rows": {}, "per_cluster": {}}
    agg = {k: {"auc": [], "fr": [], "recal_fr": []} for k in KS}
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        ddir = "e2" if (suite, task) in E2_REUSE else "e3"
        with h5py.File(os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5"), "r") as f:
            col = pca.transform(embed([f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]))
        perm = np.random.default_rng(0).permutation(len(col))
        half = len(col) // 2
        fit_f, calib_f = col[perm[:half]], col[perm[half:]]
        kz = np.load(os.path.join(E4, "known_frames", f"{cl}.npz"))
        kf = pca.transform(embed(kz["frames"]))
        nf = []
        for p in sorted(glob.glob(os.path.join(E4, "novel_frames", f"{cl}__*.npz"))):
            key = os.path.basename(p)[:-4].split("__", 1)[1]
            if key in PRIMARY_KEYS:
                z = np.load(p)
                if len(z["frames"]):
                    nf.append(pca.transform(embed(z["frames"])))
        nf = np.concatenate(nf)
        entry = {}
        for k in KS:
            cal = knn_score(fit_f, calib_f, k)
            n_c = len(cal)
            q = float(np.sort(cal)[min(ceil((n_c + 1) * (1 - ALPHA_J)), n_c) - 1])
            ks_ = knn_score(fit_f, kf, k)
            ns_ = knn_score(fit_f, nf, k)
            a = auc(ks_, ns_)
            fr = float((ks_ > q).mean())
            p2 = np.random.default_rng(0).permutation(len(ks_))
            cal2, ev2 = ks_[p2[:10]], ks_[p2[10:]]
            q2 = float(np.sort(cal2)[min(ceil(11 * (1 - ALPHA_J)), 10) - 1])
            rfr = float((ev2 > q2).mean())
            agg[k]["auc"].append(a)
            agg[k]["fr"].append(fr)
            agg[k]["recal_fr"].append(rfr)
            entry[f"k{k}"] = {"auc": round(a, 4), "fr": round(fr, 4), "recal_fr": round(rfr, 4)}
        out["per_cluster"][cl] = entry
        print(f"[KNN] {cl}: " + " ".join(f"k{k}={entry[f'k{k}']['auc']:.3f}" for k in KS), flush=True)

    def suite_mean(vals):
        m = {}
        for (s, t), v in zip(STANDARD, vals):
            m.setdefault(s, []).append(v)
        return {s: round(float(np.mean(v)), 4) for s, v in sorted(m.items())}

    for k in KS:
        out["rows"][f"6_knn_jurisdiction_k{k}"] = {
            "cost_ms": 4.0, "macro_auc": round(float(np.mean(agg[k]["auc"])), 4),
            "mean_fr": round(float(np.mean(agg[k]["fr"])), 4),
            "mean_recal_fr": round(float(np.mean(agg[k]["recal_fr"])), 4),
            "suite_auc": suite_mean(agg[k]["auc"])}
    with open(os.path.join(E4, "e4_knn_scorer.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[E4KNN-PASS] " + " | ".join(
        f"k{k}: macro={out['rows'][f'6_knn_jurisdiction_k{k}']['macro_auc']}" for k in KS))


if __name__ == "__main__":
    main()
