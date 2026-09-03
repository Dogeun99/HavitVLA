"""RGB-D vs RGB-only 스크리닝 분석 — 지시서 §8-§11, §13-§14.

두 조건은 동일 held-out 스펙(uid)에서 평가되므로 **에피소드 단위 paired** 비교가 성립한다.
모든 수치는 results/rgb_depth_ablation/의 평가 JSON에서만 읽는다(수동 입력 금지).

산출: results/rgb_depth_ablation/{ablation_summary.json, table_detail.csv, table_cluster.md,
      fig_A_curves.png, fig_B_delta.png, fig_C_nstar.png}
실행: hv2_hab python -u experiments/rgb_depth_ablation/analyze_ablation.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

OUT = "results/rgb_depth_ablation"
NGRID = [10, 20, 40, 80]
CRIT = 0.8
CLUSTERS = [  # ABLA_RGBD_CLUSTER_SELECTION.md에서 결과 산출 전 고정
    ("libero_object_task1", "object", "easy"),
    ("libero_object_task0", "object", "difficult"),
    ("libero_goal_task1", "goal", "easy"),
    ("libero_goal_task0", "goal", "difficult"),
    ("libero_spatial_task1", "spatial", "medium"),
    ("libero_10_task0", "long", "censored"),
]
# 논문 Fig. 2/5 계열과 같은 계통. 단일 축·고정 배색.
C_RGBD, C_RGB, C_INK, C_GRID, C_MUTED = "#1f6fb2", "#d2731a", "#1c2530", "#dfe3e8", "#5b6672"


def load(cluster, cond):
    p = f"{OUT}/{cluster}_{cond}_h50.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for c in d["curve"]:
        n = int(c["ckpt"].replace("act_n", "").replace(".pt", ""))
        out[n] = {"s_hat": c["s_hat"], "n_eval": c["n_eval"],
                  "per": {e["uid"]: (e["outcome"] == "success", e["steps"]) for e in c["per_episode"]}}
    return out


def nstar(curve):
    for n in NGRID:
        if n in curve and curve[n]["s_hat"] >= CRIT:
            return n
    return ">80"


def paired_bootstrap(d, b=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(d, float)
    idx = rng.integers(0, len(a), size=(b, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)


def main():
    data, missing = {}, []
    for cl, suite, grp in CLUSTERS:
        rgbd, rgb = load(cl, "rgbd"), load(cl, "rgb")
        if rgbd is None or rgb is None:
            missing.append(cl)
            continue
        data[cl] = {"suite": suite, "group": grp, "rgbd": rgbd, "rgb": rgb}
    if missing:
        raise SystemExit(f"[ABLA-FAIL] 평가 결과 누락: {missing}")

    # ---- 에피소드 단위 paired
    rows, all_delta, disc_rgb_only, disc_rgbd_only = [], [], 0, 0
    for cl, v in data.items():
        for n in NGRID:
            A, B = v["rgbd"][n], v["rgb"][n]
            uids = sorted(set(A["per"]) & set(B["per"]))
            d = [int(B["per"][u][0]) - int(A["per"][u][0]) for u in uids]
            all_delta += d
            disc_rgb_only += sum(1 for x in d if x > 0)
            disc_rgbd_only += sum(1 for x in d if x < 0)
            rows.append({"cluster": cl, "suite": v["suite"], "difficulty_group": v["group"], "n": n,
                         "n_paired": len(uids),
                         "success_rgbd": round(A["s_hat"], 4), "success_rgb": round(B["s_hat"], 4),
                         "delta_success": round(B["s_hat"] - A["s_hat"], 4)})
    mean_d, ci = paired_bootstrap(all_delta)

    # ---- N*
    ns = {cl: {"rgbd": nstar(v["rgbd"]), "rgb": nstar(v["rgb"])} for cl, v in data.items()}
    def dn(a, b):
        if a == ">80" and b == ">80":
            return "both censored"
        if b == ">80":
            return ">+1 step (censored)"
        if a == ">80":
            return "<-1 step (uncensored)"
        return NGRID.index(b) - NGRID.index(a)

    # ---- McNemar (보조)
    from scipy.stats import binomtest
    b_, c_ = disc_rgb_only, disc_rgbd_only
    mcn = float(binomtest(b_, b_ + c_, 0.5).pvalue) if (b_ + c_) else None

    # ---- 실패 유형: 로그에서 자동 분류 가능한 것은 timeout뿐
    fail = {}
    for cl, v in data.items():
        # 에피소드 상한(=timeout)은 스위트별로 다르므로 관측된 최대 steps를 상한으로 본다.
        cap = max(st for cond in ("rgbd", "rgb") for n in NGRID
                  for _, st in v[cond][n]["per"].values())
        f = {"episode_step_cap": cap}
        for cond in ("rgbd", "rgb"):
            to = non = 0
            for n in NGRID:
                for _u, (ok, steps) in v[cond][n]["per"].items():
                    if ok:
                        continue
                    if steps >= cap:
                        to += 1
                    else:
                        non += 1
            f[cond] = {"timeout_like": to, "other_failure": non}
        fail[cl] = f

    # ---- 판정 (지시서 §11)
    drops = [-r["delta_success"] for r in rows if r["n"] == 80]  # n=80 기준 절대 감소(pp)
    mean_drop_pp = float(np.mean(drops)) * 100
    nstar_moved = sum(1 for cl in ns if ns[cl]["rgbd"] != ns[cl]["rgb"])
    if mean_drop_pp <= 3 and nstar_moved <= 2:
        case, verdict = "A", "Depth is unlikely to be the primary explanation for habit competence."
    elif mean_drop_pp <= 8:
        case, verdict = "B", "Depth contributes to habit formation, but the contribution is task-dependent."
    else:
        case, verdict = "C", "Current main results are materially dependent on privileged depth input."

    import torch
    from habits.act import ACTPolicy
    params = {c: sum(p.numel() for p in ACTPolicy(pretrained=False, in_ch=ch).parameters())
              for c, ch in (("rgbd", 4), ("rgb", 3))}

    summary = {
        "stage": "Stage 1 screening (지시서 2026-08-28). formal equivalence/non-inferiority 주장 아님.",
        "clusters": [c for c, _, _ in CLUSTERS],
        "n_grid": NGRID, "n_heldout": 50,
        "paired": {"n_episodes": len(all_delta),
                   "mean_delta_success": round(mean_d, 4),
                   "ci95_bootstrap": [round(ci[0], 4), round(ci[1], 4)],
                   "discordant_rgb_only_success": b_, "discordant_rgbd_only_success": c_,
                   "mcnemar_exact_p": round(mcn, 4) if mcn is not None else None},
        "per_cluster_nstar": ns,
        "nstar_shift": {cl: dn(ns[cl]["rgbd"], ns[cl]["rgb"]) for cl in ns},
        "s80": {cl: {"rgbd": data[cl]["rgbd"][80]["s_hat"], "rgb": data[cl]["rgb"][80]["s_hat"]}
                for cl in data},
        "mean_drop_pp_at_n80": round(mean_drop_pp, 2),
        "n_clusters_nstar_moved": nstar_moved,
        "failure_types": fail,
        "failure_type_caveat": "로그에서 자동 분류 가능한 것은 timeout 계열뿐이다(steps가 상한 도달). "
                               "grasp/localization/placement/trajectory 구분은 영상 판독이 필요하며 "
                               "본 스크리닝 범위 밖 — 미분류로 보고한다.",
        "params": params, "param_delta": params["rgbd"] - params["rgb"],
        "screening_case": case, "verdict": verdict,
        "threshold_note": "3pp / 8pp는 Stage-1 엔지니어링 스크리닝 기준이며 논문의 통계적 "
                          "non-inferiority margin(−3pp)과 무관하다.",
        "rows": rows,
    }
    json.dump(summary, open(f"{OUT}/ablation_summary.json", "w"), indent=2, ensure_ascii=False)

    # ---- CSV (지시서 §14)
    with open(f"{OUT}/table_detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["Nstar_rgbd", "Nstar_rgb", "delta_Nstar",
                                                          "rgbd_params", "rgb_params"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "Nstar_rgbd": ns[r["cluster"]]["rgbd"], "Nstar_rgb": ns[r["cluster"]]["rgb"],
                        "delta_Nstar": dn(ns[r["cluster"]]["rgbd"], ns[r["cluster"]]["rgb"]),
                        "rgbd_params": params["rgbd"], "rgb_params": params["rgb"]})

    # ---- 클러스터 요약표 (논문 Table II 계열 서식)
    L = ["| Cluster | Suite | N*(RGB-D) | N*(RGB) | ŝ(80) RGB-D | ŝ(80) RGB | Δŝ(80) |",
         "|---|---|---|---|---|---|---|"]
    for cl, v in data.items():
        a, b = v["rgbd"][80]["s_hat"], v["rgb"][80]["s_hat"]
        L.append(f"| {cl.replace('libero_','')} | {v['suite']} | {ns[cl]['rgbd']} | {ns[cl]['rgb']} "
                 f"| {a:.3f} | {b:.3f} | {b-a:+.3f} |")
    open(f"{OUT}/table_cluster.md", "w").write("\n".join(L) + "\n")

    # ---- FIG A: 형성 곡선 (논문 Fig. 2(a) 계열)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    for cond, col, lab in (("rgbd", C_RGBD, "RGB-D (main)"), ("rgb", C_RGB, "RGB-only")):
        M = []
        for cl, v in data.items():
            ys = [v[cond][n]["s_hat"] for n in NGRID]
            ax.plot(NGRID, ys, color=col, lw=0.9, alpha=0.35, zorder=2)
            M.append(ys)
        ax.plot(NGRID, np.median(M, axis=0), color=col, lw=2.6, label=lab, zorder=4)
    ax.axhline(CRIT, color=C_INK, lw=1.2, ls="--", zorder=3)
    ax.annotate(f"formation criterion $\\hat{{s}}$ = {CRIT}", (NGRID[0], CRIT),
                textcoords="offset points", xytext=(4, 6), fontsize=9, color=C_INK)
    ax.set_xscale("log")
    ax.set_xticks(NGRID, [str(n) for n in NGRID])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # 논문 Fig.2와 동일한 눈금
    ax.set_xlabel("teacher trajectories used for training, $n$", color=C_INK, fontsize=10)
    ax.set_ylabel("held-out habit success $\\hat{s}(n)$", color=C_INK, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Habit formation with and without depth (6 clusters, thin lines; median, thick)",
                 fontsize=10.5, color=C_INK, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_A_curves.png", dpi=200, facecolor="white"); plt.close(fig)

    # ---- FIG B: 클러스터별 Δ success (논문 Fig. 5(c) 계열)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    style(ax)
    labels, vals = [], []
    for cl, v in data.items():
        labels.append(cl.replace("libero_", ""))
        vals.append(v["rgb"][80]["s_hat"] - v["rgbd"][80]["s_hat"])
    y = np.arange(len(labels))
    ax.barh(y, vals, color=[C_RGB if x < 0 else C_RGBD for x in vals], height=0.55, zorder=3)
    ax.axvline(0, color=C_INK, lw=1.2, zorder=4)
    for i, x in enumerate(vals):
        ax.annotate(f"{x:+.3f}", (x, i), textcoords="offset points",
                    xytext=(6 if x >= 0 else -6, 0), va="center",
                    ha="left" if x >= 0 else "right", fontsize=9, color=C_INK)
    ax.set_yticks(y, labels, fontsize=9); ax.invert_yaxis()
    pad = 0.055 * max(abs(min(vals)), abs(max(vals))) / 0.14   # 라벨 폭 확보(눈금 라벨과 충돌 방지)
    ax.set_xlim(min(min(vals), 0) - pad, max(max(vals), 0) + pad)
    ax.set_xlabel("$\\Delta\\hat{s}(80)$  =  RGB-only $-$ RGB-D", color=C_INK, fontsize=10)
    ax.set_title("Per-cluster effect of removing depth at $n=80$", fontsize=10.5, color=C_INK, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_B_delta.png", dpi=200, facecolor="white"); plt.close(fig)

    # ---- FIG C: N* paired
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    style(ax)
    jit = np.linspace(-0.06, 0.06, len(data))
    for i, (cl, v) in enumerate(data.items()):
        xa = NGRID.index(ns[cl]["rgbd"]) if ns[cl]["rgbd"] != ">80" else len(NGRID)
        xb = NGRID.index(ns[cl]["rgb"]) if ns[cl]["rgb"] != ">80" else len(NGRID)
        ax.plot([0 + jit[i], 1 + jit[i]], [xa, xb], color=C_MUTED, lw=1.0, zorder=2)
        ax.scatter([0 + jit[i]], [xa], color=C_RGBD, s=42, zorder=3)
        ax.scatter([1 + jit[i]], [xb], color=C_RGB, s=42, zorder=3)
        stack = [c for c in data if (NGRID.index(ns[c]["rgb"]) if ns[c]["rgb"] != ">80" else len(NGRID)) == xb]
        off = (stack.index(cl) - (len(stack) - 1) / 2) * 11    # 같은 N*에 여러 클러스터가 겹칠 때 분리
        ax.annotate(cl.replace("libero_", ""), (1 + jit[i], xb), textcoords="offset points",
                    xytext=(9, off), va="center", fontsize=8, color=C_MUTED)
    ax.set_xticks([0, 1], ["RGB-D", "RGB-only"], fontsize=10)
    ax.set_yticks(range(len(NGRID) + 1), [str(n) for n in NGRID] + [">80"])
    ax.set_ylabel("formation threshold $N^\\star$", color=C_INK, fontsize=10)
    ax.set_xlim(-0.35, 1.55)
    ax.set_title("Formation threshold, paired", fontsize=10.5, color=C_INK, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_C_nstar.png", dpi=200, facecolor="white"); plt.close(fig)

    print(f"[ABLA-ANALYSIS-DONE] case {case}")
    print(f"  paired Δ = {mean_d:+.4f} CI[{ci[0]:+.4f}, {ci[1]:+.4f}] over {len(all_delta)} episodes")
    print(f"  mean drop at n=80: {mean_drop_pp:+.2f} pp · N* moved: {nstar_moved}/6")
    print(f"  discordant: RGB-only만 성공 {b_} / RGB-D만 성공 {c_} · McNemar p={mcn}")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
