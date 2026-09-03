"""E4 종결 시퀀스 2 — 행3: 히든 L32 visual-token mean scorer (hv2_oft, +85ms).

규격 (Paper 1 이월): OFT forward의 마지막 층 hidden에서 시각 토큰(:NUM_PATCHES) 평균.
구현: language_model forward hook — get_vla_action 경로 그대로(부산물 추출, E1 앵커④).
범위 (예산 1h 제약, goal 중심 판독 목적): goal task0·2·4·7 + 대조 object_task2·spatial_task0.
정합성: fit(수집 HDF5 저장 128 obs)과 eval(known/novel 스펙 재실현 → 동일 128 저장 규격)
모두 128→모델 해상도 경로로 통일. PCA(32)는 이 scorer 전용(6셀 수집 풀 fit — 문서화).

산출: results/e4/e4_hidden_scorer.json
실행: hv2_oft python -u experiments/e4_hidden_scorer.py
"""
import glob
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.libero_env import EpisodeSpec, LiberoEpisodeEnv  # noqa: E402
from envs.chained_env import ChainedEpisodeSpec  # noqa: E402  (uid 재구성 대비)
from teacher.collector import load_teacher, teacher_observation, store_frame  # noqa: E402
from gates.two_stage import JurisdictionGate  # noqa: E402

E4 = os.path.join(HABIT2, "results", "e4")
SCOPE = [("libero_goal", 0), ("libero_goal", 2), ("libero_goal", 4), ("libero_goal", 7),
         ("libero_object", 2), ("libero_spatial", 0)]
PRIMARY_KEYS = ("w_expand__primary", "borrow__primary")
E2_REUSE = set()


def auc(neg, pos):
    from scipy.stats import rankdata

    x = np.concatenate([neg, pos])
    r = rankdata(x)
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


class HiddenTap:
    """language_model forward hook — 마지막 층 hidden의 시각 토큰 평균 캡처."""

    def __init__(self, model):
        # NUM_PATCHES는 모듈 상수가 아니라 호출 시 계산됨(modeling_prismatic:1019) —
        # 순수 시각 토큰 스팬 = patches × images (proprio 토큰 제외, Paper 1 규격 visual-mean)
        self.num_patches = (model.vision_backbone.get_num_patches()
                            * model.vision_backbone.get_num_images_in_input())
        self.captured = None
        self.h = model.language_model.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        hs = output.hidden_states[-1]  # (B, seq, D)
        self.captured = hs[:, : self.num_patches].mean(1).float().cpu().numpy()[0]


def upscale2(img128):
    return np.kron(img128, np.ones((2, 2, 1))).astype(np.uint8)


def main():
    import h5py

    from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: F401
    from experiments.robot.openvla_utils import get_vla_action
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    set_seed_everywhere(7)
    out = {"scope": [f"{s}_task{t}" for s, t in SCOPE],
           "note": "행3 히든 scorer — 6셀 범위(예산 제약, goal 중심). fit/eval 모두 128 저장 규격 경유. "
                   "PCA(32) 전용 fit(6셀 수집 풀).", "clusters": {}}

    by_suite = {}
    for s, t in SCOPE:
        by_suite.setdefault(s, []).append(t)

    feats_col, feats_known, feats_novel = {}, {}, {}
    for suite, tasks in by_suite.items():
        teacher = load_teacher(suite)
        cfg, model = teacher[0], teacher[1]
        resize_size = get_image_resize_size(cfg)
        tap = HiddenTap(model)

        from experiments.robot.openvla_utils import resize_image_for_policy

        def hidden_of(agent128_rot, wrist128_rot, proprio, language):
            """저장 규격 부품(회전 128 RGB×2 + proprio)에서 직접 모델 입력 구성 —
            teacher_observation 우회 (fit=HDF5·eval=재실현 모두 동일 경로 = 정합)."""
            t_obs = {"full_image": resize_image_for_policy(upscale2(agent128_rot), resize_size),
                     "wrist_image": resize_image_for_policy(upscale2(wrist128_rot), resize_size),
                     "state": np.asarray(proprio)}
            get_vla_action(cfg, model, teacher[5], t_obs, language,
                           action_head=teacher[2], proprio_projector=teacher[3],
                           noisy_action_projector=teacher[4])
            return tap.captured.copy()

        for task in tasks:
            cl = f"{suite}_task{task}"
            env = LiberoEpisodeEnv(suite, task)
            with h5py.File(os.path.join(HABIT2, "data", "e3", f"{cl}.hdf5"), "r") as f:
                col = []
                for k in list(f["episodes"]):
                    g = f[f"episodes/{k}"]
                    col.append(hidden_of(g["agentview_rgb"][0], g["wrist_rgb"][0],
                                         g["proprio"][0], env.language))
            feats_col[cl] = np.stack(col)
            # known 재실현 (full obs → 저장 규격 경유)
            from envs.stream import heldout_specs

            kn = []
            for spec in heldout_specs(suite, task, 20):
                obs = spec.realize(env)
                sf = store_frame(obs)
                kn.append(hidden_of(sf["agentview_rgb"], sf["wrist_rgb"],
                                    sf["proprio"], env.language))
            feats_known[cl] = np.stack(kn)
            # novel primary 재실현: 저장된 uid 목록의 스펙 파라미터 재구성이 필요 —
            # w_expand/borrow 스펙은 결정적 대역식이므로 e4_novel_frames와 동일식으로 재생성
            from envs.libero_env import USABLE_W_MAX

            nv = []
            w = USABLE_W_MAX[suite]
            bases = list(range(40, 50))
            for j in range(20):
                spec = EpisodeSpec(suite, task, 30_000 + j, bases[j % 10], w, 2_000_000 + j)
                obs = spec.realize(env)
                sf = store_frame(obs)
                nv.append(hidden_of(sf["agentview_rgb"], sf["wrist_rgb"],
                                    sf["proprio"], env.language))
            feats_novel[cl] = np.stack(nv)
            env.close()
            print(f"[HIDDEN] {cl}: col={len(feats_col[cl])} known=20 novel=20", flush=True)
        tap.h.remove()
        del teacher
        import torch

        torch.cuda.empty_cache()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=32, random_state=0).fit(np.concatenate(list(feats_col.values())))
    aucs, frs = [], []
    for s, t in SCOPE:
        cl = f"{s}_task{t}"
        gate = JurisdictionGate().fit(pca.transform(feats_col[cl]))
        k = np.array([gate.score(x) for x in pca.transform(feats_known[cl])])
        n = np.array([gate.score(x) for x in pca.transform(feats_novel[cl])])
        a, fr = auc(k, n), float((k > gate.q).mean())
        aucs.append(a)
        frs.append(fr)
        out["clusters"][cl] = {"auc": round(a, 4), "fr_at_q": round(fr, 4)}
        print(f"[HIDDEN] {cl}: AUC={a:.4f} FR={fr:.2f}", flush=True)
    out["macro_auc_scope"] = round(float(np.mean(aucs)), 4)
    out["goal_macro_auc"] = round(float(np.mean(aucs[:4])), 4)
    out["mean_fr"] = round(float(np.mean(frs)), 4)
    with open(os.path.join(E4, "e4_hidden_scorer.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E4HIDDEN-PASS] scope macro={out['macro_auc_scope']} goal={out['goal_macro_auc']} "
          f"FR={out['mean_fr']}")


if __name__ == "__main__":
    main()
