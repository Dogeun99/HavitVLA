"""E4 정식 재판정 (§5 2026-08-16 등재 구성 — 1회 약정, 임계 0.75 불변).

주 novel = w 확대(전 스위트; long w=0.02) ∪ 차용(spatial). 재샘플 = in-distribution
negative control (수용률 병기). 산출: 25셀 macro AUC + 경로별·스위트별 표 +
클러스터별 false-reject율(운용 q 기준). 미달 시 즉시 REDUCE — 재시도 금지.

산출: results/e4/e4_readjudication.json + [E4READJ-GO|E4READJ-REDUCE]
실행: hv2_hab python -u experiments/e4_readjudication.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

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
THRESH = 0.75
PRIMARY_KEYS = ("w_expand__primary", "borrow__primary")


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

    rep = {"threshold": THRESH,
           "composition": "primary = w_expand(전 스위트, long w=0.02) ∪ borrow(spatial); "
                          "resample = in-distribution negative control",
           "clusters": {}, "per_path_macro": {}, "per_suite_macro": {},
           "negative_control": {}}
    macro, path_pool, suite_pool = [], defaultdict(list), defaultdict(list)
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        ddir = "e2" if (suite, task) in E2_REUSE else "e3"
        with h5py.File(os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5"), "r") as f:
            col = [f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]
        gate = JurisdictionGate().fit(pca.transform(embed(col)))
        kz = np.load(os.path.join(E4, "known_frames", f"{cl}.npz"))
        known = np.array([gate.score(x) for x in pca.transform(embed(kz["frames"]))])
        false_reject = float((known > gate.q).mean())

        pool, entry_paths = [], {}
        for p in sorted(glob.glob(os.path.join(E4, "novel_frames", f"{cl}__*.npz"))):
            key = os.path.basename(p)[:-4].split("__", 1)[1]
            nz = np.load(p)
            if len(nz["frames"]) == 0:
                continue
            sc = np.array([gate.score(x) for x in pca.transform(embed(nz["frames"]))])
            a = auc(known, sc)
            entry_paths[key] = {"n": len(sc), "auc": round(a, 4)}
            path_pool[key].append(a)
            if key in PRIMARY_KEYS:
                pool.append(sc)
            if key == "resample__single":
                rep["negative_control"][cl] = {
                    "n": len(sc), "accept_rate_at_q": round(float((sc <= gate.q).mean()), 4)}
        a_cl = auc(known, np.concatenate(pool))
        macro.append(a_cl)
        suite_pool[suite].append(a_cl)
        rep["clusters"][cl] = {"auc_primary": round(a_cl, 4),
                               "false_reject_at_q": round(false_reject, 4),
                               "q": round(gate.q, 3), "paths": entry_paths}
        print(f"[READJ] {cl}: AUC={a_cl:.4f} FR={false_reject:.2f}", flush=True)

    rep["per_path_macro"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(path_pool.items())}
    rep["per_suite_macro"] = {s: round(float(np.mean(v)), 4) for s, v in sorted(suite_pool.items())}
    rep["macro_auc"] = round(float(np.mean(macro)), 4)
    rep["n_below_threshold"] = sum(1 for a in macro if a < THRESH)
    rep["mean_false_reject"] = round(float(np.mean(
        [v["false_reject_at_q"] for v in rep["clusters"].values()])), 4)
    rep["mean_negative_control_accept"] = round(float(np.mean(
        [v["accept_rate_at_q"] for v in rep["negative_control"].values()])), 4)
    go = rep["macro_auc"] >= THRESH
    rep["verdict"] = "GO" if go else "REDUCE (§3 축소 경로 — E5 성숙도 단독, 1회 약정에 따라 확정)"
    with open(os.path.join(E4, "e4_readjudication.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[{'E4READJ-GO' if go else 'E4READJ-REDUCE'}] macro={rep['macro_auc']} "
          f"(<0.75: {rep['n_below_threshold']}/25) | suite={rep['per_suite_macro']} | "
          f"FR={rep['mean_false_reject']} | NC수용={rep['mean_negative_control_accept']}")


if __name__ == "__main__":
    main()
