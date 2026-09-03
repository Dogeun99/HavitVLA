"""E4 종결 시퀀스 2 — scorer 진단 프레임 (관할/feasibility/oracle + ★q 재보정 행).

행 구성 (각 AUC known-vs-novel주 + false-reject):
  1. 관할 Mahalanobis(+4.0ms): e4_readjudication.json에서 전재.
  2. feasibility head: DINOv2-PCA 특징 → 수집 성공/실패 로지스틱(class-weighted).
     실패 I₀는 수집 meta 스펙에서 결정적 재렌더(수집 프레임은 성공만 저장 — 이중 장부).
     누출 가드: 학습 = 수집만 / 평가 = known·novel. FR 임계 = 수집 예측의 (1−α_j) 분위수.
  4. oracle(스펙 대역 GT): AUC=1.0·FR=0 — 명목 상한 행.
  5. ★q 재보정 관할: known 20을 50:50 보정/평가 분할(seed 0, 이중 사용 금지) →
     q' = 보정 half 유한표본 분위수, FR'는 평가 half만. **AUC는 임계 무관이므로 불변 명기.**
  (3. 히든 L32는 e4_hidden_scorer.py — hv2_oft 별도 실행 후 표 병합.)

산출: results/e4/e4_scorer_diag.json
실행: hv2_hab python -u experiments/e4_scorer_diag.py
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
os.environ.setdefault("MUJOCO_GL", "egl")

from gates.features import DinoFeatureExtractor, SharedPCA, prep_gate_rgb  # noqa: E402
from gates.two_stage import ALPHA_J, JurisdictionGate  # noqa: E402

E4 = os.path.join(HABIT2, "results", "e4")
STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
E2_REUSE = {("libero_object", 0), ("libero_object", 5)}
PRIMARY_KEYS = ("w_expand__primary", "borrow__primary")


def auc(neg, pos):
    from scipy.stats import rankdata

    x = np.concatenate([neg, pos])
    r = rankdata(x)
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main():
    import h5py

    from envs.libero_env import EpisodeSpec, LiberoEpisodeEnv

    ext = DinoFeatureExtractor()
    pca = SharedPCA.load(os.path.join(E4, "shared_pca_e4.joblib"))

    def embed(frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            out.append(ext.embed(list(frames[i:i + bs])))
        return np.concatenate(out) if out else np.zeros((0, 384))

    # ---- 수집 성공/실패 특징 (실패 I₀ = meta 스펙 결정적 재렌더)
    X_all, y_all, per_cluster = [], [], {}
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        ddir = "e2" if (suite, task) in E2_REUSE else "e3"
        with h5py.File(os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5"), "r") as f:
            succ_frames = [f[f"episodes/{k}/agentview_rgb"][0] for k in f["episodes"]]
            meta = json.loads(f["meta_json"][()])
        fails = [m for m in meta if m["outcome"] == "fail"]
        fail_frames = []
        if fails:
            env = LiberoEpisodeEnv(suite, task)
            for m in fails:
                spec = EpisodeSpec(m["suite"], m["task_id"], m["seed"], m["base_idx"],
                                   m["w"], m["noise_seed"])
                obs = spec.realize(env)
                fail_frames.append(prep_gate_rgb(obs["agentview_image"]))
            env.close()
        fs = pca.transform(embed(succ_frames))
        ff = pca.transform(embed(fail_frames)) if fail_frames else np.zeros((0, 32))
        per_cluster[cl] = {"succ": fs, "fail": ff}
        X_all.append(np.concatenate([fs, ff]) if len(ff) else fs)
        y_all.append(np.concatenate([np.zeros(len(fs)), np.ones(len(ff))]))
        print(f"[SCORER] {cl}: succ={len(fs)} fail재렌더={len(ff)}", flush=True)

    from sklearn.linear_model import LogisticRegression

    X, y = np.concatenate(X_all), np.concatenate(y_all)
    head = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0).fit(X, y)
    col_pred = head.predict_proba(X)[:, 1]
    head_thresh = float(np.quantile(col_pred, 1 - ALPHA_J))

    readj = json.load(open(os.path.join(E4, "e4_readjudication.json")))
    out = {"note": "scorer 진단 프레임 (E4 종결 시퀀스 2). AUC = known vs novel주. "
                   "행5 재보정: AUC 불변(임계 무관 — 명기), FR만 재측정.",
           "rows": {}, "per_cluster": {}}
    feas_auc, feas_fr, recal_fr, base_fr = [], [], [], []
    for suite, task in STANDARD:
        cl = f"{suite}_task{task}"
        gate = JurisdictionGate().fit(per_cluster[cl]["succ"])
        kz = np.load(os.path.join(E4, "known_frames", f"{cl}.npz"))
        kf = pca.transform(embed(kz["frames"]))
        k_jur = np.array([gate.score(x) for x in kf])
        k_feas = head.predict_proba(kf)[:, 1]
        nf_pool = []
        for p in sorted(glob.glob(os.path.join(E4, "novel_frames", f"{cl}__*.npz"))):
            key = os.path.basename(p)[:-4].split("__", 1)[1]
            if key not in PRIMARY_KEYS:
                continue
            nz = np.load(p)
            if len(nz["frames"]):
                nf_pool.append(pca.transform(embed(nz["frames"])))
        nf = np.concatenate(nf_pool)
        n_feas = head.predict_proba(nf)[:, 1]
        feas_auc.append(auc(k_feas, n_feas))
        feas_fr.append(float((k_feas > head_thresh).mean()))
        base_fr.append(float((k_jur > gate.q).mean()))
        # 행5: q 재보정 (50:50, seed 0; 이중 사용 금지)
        perm = np.random.default_rng(0).permutation(len(k_jur))
        cal, ev = k_jur[perm[:10]], k_jur[perm[10:]]
        q2 = float(np.sort(cal)[min(ceil(11 * (1 - ALPHA_J)), 10) - 1])
        recal_fr.append(float((ev > q2).mean()))
        out["per_cluster"][cl] = {
            "jur_auc": readj["clusters"][cl]["auc_primary"],
            "jur_fr": readj["clusters"][cl]["false_reject_at_q"],
            "feas_auc": round(feas_auc[-1], 4), "feas_fr": round(feas_fr[-1], 4),
            "recal_q": round(q2, 3), "recal_fr_evalhalf": round(recal_fr[-1], 4),
        }
        print(f"[SCORER] {cl}: feas_auc={feas_auc[-1]:.3f} recal_FR={recal_fr[-1]:.2f}", flush=True)

    def suite_mean(vals):
        agg = {}
        for (s, t), v in zip(STANDARD, vals):
            agg.setdefault(s, []).append(v)
        return {s: round(float(np.mean(v)), 4) for s, v in sorted(agg.items())}

    out["rows"]["1_jurisdiction_mahalanobis"] = {
        "cost_ms": 4.0, "macro_auc": readj["macro_auc"],
        "mean_fr": readj["mean_false_reject"], "suite_auc": readj["per_suite_macro"]}
    out["rows"]["2_feasibility_head"] = {
        "cost_ms": 4.0, "macro_auc": round(float(np.mean(feas_auc)), 4),
        "mean_fr": round(float(np.mean(feas_fr)), 4), "suite_auc": suite_mean(feas_auc),
        "note": "실패 I₀ 재렌더로 학습 표본 복원 (~3% 양성 — class-weighted)"}
    out["rows"]["4_oracle_band_gt"] = {"cost_ms": 0.0, "macro_auc": 1.0, "mean_fr": 0.0,
                                       "note": "스펙 대역 GT — 명목 상한"}
    out["rows"]["5_jurisdiction_recalibrated"] = {
        "cost_ms": 4.0, "macro_auc": readj["macro_auc"],
        "auc_note": "AUC 불변 — 임계 무관 (명기)",
        "mean_fr": round(float(np.mean(recal_fr)), 4),
        "mean_fr_before": round(float(np.mean(base_fr)), 4),
        "procedure": "known 50:50 보정/평가 (seed 0), q' = 유한표본 분위수(n=10 → max), FR은 평가 half만"}
    with open(os.path.join(E4, "e4_scorer_diag.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E4SCORER-PASS] feas macro={out['rows']['2_feasibility_head']['macro_auc']} "
          f"recal FR {out['rows']['5_jurisdiction_recalibrated']['mean_fr_before']}→"
          f"{out['rows']['5_jurisdiction_recalibrated']['mean_fr']}")


if __name__ == "__main__":
    main()
