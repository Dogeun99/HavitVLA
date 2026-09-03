"""논문(v11) 서식에 맞춘 depth ablation 산출물 생성기.

단일 진입점 = results/rgb_depth_ablation/ablation_summary.json (수동 입력 0).
산출: results/rgb_depth_ablation/paper/{ablation_numbers.json, SECTION_depth_ablation.tex,
      TABLE_depth_ablation.tex, fig_depth_ablation.png, INSTRUCTION_*.md}
실행: hv2_hab python -u experiments/rgb_depth_ablation/make_paper_materials.py
"""
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

SRC = "results/rgb_depth_ablation/ablation_summary.json"
OUT = "results/rgb_depth_ablation/paper"
NGRID = [10, 20, 40, 80]
C_RGBD, C_RGB, C_INK, C_GRID, C_MUTED = "#1f6fb2", "#d2731a", "#1c2530", "#dfe3e8", "#5b6672"
SUITE = {"libero_object_task1": ("Object", "easy"), "libero_object_task0": ("Object", "difficult"),
         "libero_goal_task1": ("Goal", "easy"), "libero_goal_task0": ("Goal", "difficult"),
         "libero_spatial_task1": ("Spatial", "medium"), "libero_10_task0": ("Long", "censored")}
LONG = "libero_10_task0"


def short(c):
    return c.replace("libero_", "").replace("10_", "long_")


def main():
    d = json.load(open(SRC))
    rows, p, ns, s80 = d["rows"], d["paired"], d["per_cluster_nstar"], d["s80"]
    delta80 = {r["cluster"]: r["delta_success"] for r in rows if r["n"] == 80}
    by_n = {n: [r["delta_success"] for r in rows if r["n"] == n] for n in NGRID}
    ex_long = [v for c, v in delta80.items() if c != LONG]
    n_same = sum(1 for c in ns if ns[c]["rgbd"] == ns[c]["rgb"])

    # ---------------- 매크로 소스 (build_numbers.py가 읽을 형태) ----------------
    def M(name, value, fmt, note):
        return {"macro": name, "value": value, "formatted": fmt,
                "source": f"{SRC} :: {note}"}

    macros = [
        M("AblaClusters", len(d["clusters"]), f"{len(d['clusters'])}", "clusters[]"),
        M("AblaEpisodes", p["n_episodes"], f"{p['n_episodes']:,}", "paired.n_episodes"),
        M("AblaHeldout", d["n_heldout"], f"{d['n_heldout']}", "n_heldout"),
        M("AblaParamDelta", d["param_delta"], f"{d['param_delta']:,}", "param_delta"),
        M("AblaParamPct", round(100 * d["param_delta"] / d["params"]["rgbd"], 4),
          f"{100 * d['param_delta'] / d['params']['rgbd']:.4f}", "param_delta / params.rgbd"),
        M("AblaPairedDelta", p["mean_delta_success"], f"{p['mean_delta_success']:+.4f}",
          "paired.mean_delta_success"),
        M("AblaPairedLo", p["ci95_bootstrap"][0], f"{p['ci95_bootstrap'][0]:+.4f}",
          "paired.ci95_bootstrap[0]"),
        M("AblaPairedHi", p["ci95_bootstrap"][1], f"{p['ci95_bootstrap'][1]:+.4f}",
          "paired.ci95_bootstrap[1]"),
        M("AblaMcNemarP", round(p["mcnemar_exact_p"], 4), f"{p['mcnemar_exact_p']:.4f}",
          "paired.mcnemar_exact_p (exact McNemar, two-sided)"),
        M("AblaDiscRGB", p["discordant_rgb_only_success"], f"{p['discordant_rgb_only_success']}",
          "paired.discordant_rgb_only_success"),
        M("AblaDiscRGBD", p["discordant_rgbd_only_success"], f"{p['discordant_rgbd_only_success']}",
          "paired.discordant_rgbd_only_success"),
        M("AblaDropEighty", d["mean_drop_pp_at_n80"], f"{d['mean_drop_pp_at_n80']:.2f}",
          "mean_drop_pp_at_n80 (percentage points)"),
        M("AblaDropEightyExLong", round(-100 * float(np.mean(ex_long)), 2),
          f"{-100 * float(np.mean(ex_long)):.2f}",
          "mean over rows[n=80].delta_success excluding libero_10_task0, sign-flipped to a drop"),
        M("AblaNstarSame", n_same, f"{n_same}", "per_cluster_nstar: rgbd == rgb"),
        M("AblaNstarMoved", d["n_clusters_nstar_moved"], f"{d['n_clusters_nstar_moved']}",
          "n_clusters_nstar_moved"),
        M("AblaLongDelta", delta80[LONG], f"{delta80[LONG]:+.2f}", "rows[libero_10_task0, n=80]"),
        M("AblaSpatialDelta", delta80["libero_spatial_task1"],
          f"{delta80['libero_spatial_task1']:+.2f}", "rows[libero_spatial_task1, n=80]"),
    ]
    for n in NGRID:
        macros.append(M(f"AblaDeltaN{n}", round(100 * float(np.mean(by_n[n])), 2),
                        f"{100 * float(np.mean(by_n[n])):+.2f}",
                        f"mean over rows[n={n}].delta_success, percentage points"))
    payload = {"generated_from": SRC, "stage": d["stage"],
               "screening_case": d["screening_case"], "macros": macros}
    json.dump(payload, open(f"{OUT}/ablation_numbers.json", "w"), indent=1, ensure_ascii=False)

    # ---------------- 논문 서식 표 (Table II 계열) ----------------
    T = []
    T.append(r"% Depth ablation, per-cluster. 수치는 numbers 매크로로 주입할 것 —")
    T.append(r"% 아래는 서식 확인용 렌더이며 본문 병합 시 \Num{} 치환 대상이다.")
    T.append(r"\begin{table}[t]")
    T.append(r"\caption{Habit formation with and without depth. Six clusters stratified by suite and")
    T.append(r"difficulty; identical teacher trajectories, training schedule, seed, and held-out")
    T.append(r"episode specifications in both conditions. $\Delta$ is RGB-only minus RGB-D.}")
    T.append(r"\label{tab:depth_ablation}")
    T.append(r"\centering\small")
    T.append(r"\begin{tabular}{llcccc}")
    T.append(r"\toprule")
    T.append(r"& & \multicolumn{2}{c}{$N^{\star}$} & \multicolumn{2}{c}{$\hat{s}(80)$} \\")
    T.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    T.append(r"Cluster & Suite & RGB-D & RGB & RGB-D & RGB \\")
    T.append(r"\midrule")
    for c in d["clusters"]:
        su, _ = SUITE[c]
        a, b = ns[c]["rgbd"], ns[c]["rgb"]
        fa = r"$>$80" if a == ">80" else str(a)
        fb = r"$>$80" if b == ">80" else str(b)
        T.append(f"\\texttt{{{short(c).replace('_', chr(92) + '_')}}} & {su} & {fa} & {fb} "
                 f"& {s80[c]['rgbd']:.2f} & {s80[c]['rgb']:.2f} \\\\")
    T.append(r"\midrule")
    T.append(f"\\multicolumn{{6}}{{l}}{{\\footnotesize Paired over {p['n_episodes']:,} episodes: "
             f"$\\Delta = {p['mean_delta_success']:+.4f}$, 95\\% CI "
             f"$[{p['ci95_bootstrap'][0]:+.4f}, {p['ci95_bootstrap'][1]:+.4f}]$,}} \\\\")
    T.append(f"\\multicolumn{{6}}{{l}}{{\\footnotesize exact McNemar $p = {p['mcnemar_exact_p']:.2f}$ "
             f"({p['discordant_rgb_only_success']}/{p['discordant_rgbd_only_success']} discordant).}} \\\\")
    T.append(r"\bottomrule")
    T.append(r"\end{tabular}")
    T.append(r"\end{table}")
    open(f"{OUT}/TABLE_depth_ablation.tex", "w").write("\n".join(T) + "\n")

    # ---------------- 논문 서식 본문 (V-F Scope 인접 삽입) ----------------
    S = []
    S.append(r"% ---- Depth ablation subsection. 삽입 위치: V-F Scope and Limitations 직전 ----")
    S.append(r"% 모든 수치는 \Num{...} 매크로. 본문 직접 숫자 입력 금지.")
    S.append(r"\subsection{Does Depth Explain the Habit's Performance?}")
    S.append(r"\label{sec:depth_ablation}")
    S.append("")
    S.append(r"The habit policy receives depth, which the OpenVLA-OFT teacher does not. This raises")
    S.append(r"the question of whether the habit's held-out success reflects habit formation or")
    S.append(r"simply privileged sensing. We screen this directly. On \Num{AblaClusters} clusters")
    S.append(r"stratified by suite and difficulty---selected and frozen before any result was")
    S.append(r"produced---we retrain the habit with the depth channel removed and nothing else")
    S.append(r"changed. Both conditions consume the same teacher trajectories, the same")
    S.append(r"$\mathcal{B}_k$ ordering, the same optimizer, schedule, and seed, and are evaluated on")
    S.append(r"the same \Num{AblaHeldout} held-out episode specifications, so every episode is paired.")
    S.append(r"Removing depth narrows the first convolution from four channels to three, a difference")
    S.append(r"of \Num{AblaParamDelta} parameters (\Num{AblaParamPct}\%).")
    S.append("")
    S.append(r"Over \Num{AblaEpisodes} paired episodes the mean difference is \Num{AblaPairedDelta}")
    S.append(r"with a bootstrap 95\% confidence interval of $[\Num{AblaPairedLo},")
    S.append(r"\Num{AblaPairedHi}]$, which contains zero; the exact McNemar test on")
    S.append(r"\Num{AblaDiscRGB}/\Num{AblaDiscRGBD} discordant pairs gives $p = \Num{AblaMcNemarP}$.")
    S.append(r"Table~\ref{tab:depth_ablation} and Fig.~\ref{fig:depth_ablation} resolve this by")
    S.append(r"cluster. The formation threshold $N^{\star}$ is unchanged in \Num{AblaNstarSame} of")
    S.append(r"\Num{AblaClusters} clusters, and the \Num{AblaNstarMoved} that move do so in opposite")
    S.append(r"directions, so depth removal does not systematically delay formation. The effect")
    S.append(r"across $n$ is not monotone: the mean difference is \Num{AblaDeltaN10}~pp at $n=10$,")
    S.append(r"\Num{AblaDeltaN20}~pp at $n=20$, \Num{AblaDeltaN40}~pp at $n=40$, and")
    S.append(r"\Num{AblaDeltaN80}~pp at $n=80$. Depth therefore helps where data is scarcest and")
    S.append(r"contributes little at the ceiling on which the maturity certificate---and hence every")
    S.append(r"claim in this paper about substituting the habit for the teacher---depends.")
    S.append("")
    S.append(r"Two qualifications are needed. First, the mean drop at $n=80$ is")
    S.append(r"\Num{AblaDropEighty}~pp, and it is carried almost entirely by the single")
    S.append(r"long-suite cluster ($\Delta = \Num{AblaLongDelta}$), which is right-censored under")
    S.append(r"both conditions and never forms at all; excluding it the mean drop is")
    S.append(r"\Num{AblaDropEightyExLong}~pp. Depth may thus matter more for long-horizon tasks than")
    S.append(r"this screen can resolve. Second, the cluster we expected to depend most on depth,")
    S.append(r"a Spatial cluster, shows $\Delta = \Num{AblaSpatialDelta}$ and an unchanged")
    S.append(r"$N^{\star}$, so a geometric reading of which tasks need depth is not supported. All")
    S.append(r"failures in both conditions reach the episode step cap rather than terminating early,")
    S.append(r"which indicates that removing depth degrades completion rather than causing")
    S.append(r"qualitatively different failures; separating grasp, localization, and placement")
    S.append(r"failures would require per-episode video and is left to future work. This is a")
    S.append(r"screening experiment on \Num{AblaClusters} of the 33 clusters, not a formal")
    S.append(r"equivalence test.")
    open(f"{OUT}/SECTION_depth_ablation.tex", "w").write("\n".join(S) + "\n")

    # ---------------- 논문 폭 2-패널 그림 (double column 7.16in) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1.05, 1]})
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=7.5)

    ax = axes[0]
    curves = {c: {"rgbd": [], "rgb": []} for c in d["clusters"]}
    for r in rows:
        curves[r["cluster"]]["rgbd"].append(r["success_rgbd"])
        curves[r["cluster"]]["rgb"].append(r["success_rgb"])
    for c in d["clusters"]:
        ax.plot(NGRID, curves[c]["rgbd"], color=C_RGBD, lw=0.8, alpha=0.3, zorder=2)
        ax.plot(NGRID, curves[c]["rgb"], color=C_RGB, lw=0.8, alpha=0.3, zorder=2)
    med_d = [np.median([curves[c]["rgbd"][i] for c in d["clusters"]]) for i in range(4)]
    med_r = [np.median([curves[c]["rgb"][i] for c in d["clusters"]]) for i in range(4)]
    ax.plot(NGRID, med_d, color=C_RGBD, lw=2.2, zorder=4, label="RGB-D (main)")
    ax.plot(NGRID, med_r, color=C_RGB, lw=2.2, zorder=4, label="RGB-only")
    ax.axhline(0.8, color=C_INK, lw=1.0, ls="--", zorder=3)
    ax.annotate(r"$\hat{s}=0.8$", (10.4, 0.815), fontsize=7, color=C_INK)
    ax.set_xscale("log")
    ax.set_xticks(NGRID, [str(n) for n in NGRID])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"teacher trajectories $n$", color=C_INK, fontsize=8.5)
    ax.set_ylabel(r"held-out success $\hat{s}(n)$", color=C_INK, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("(a) formation curves (median, thick)", fontsize=8.5, color=C_INK, loc="left")

    ax = axes[1]
    labels = [short(c) for c in d["clusters"]]
    vals = [delta80[c] for c in d["clusters"]]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=[C_RGB if x < 0 else C_RGBD for x in vals], height=0.6, zorder=3)
    ax.axvline(0, color=C_INK, lw=1.0, zorder=4)
    for i, x in enumerate(vals):
        ax.annotate(f"{x:+.2f}", (x, i), textcoords="offset points",
                    xytext=(5 if x >= 0 else -5, 0), va="center",
                    ha="left" if x >= 0 else "right", fontsize=7, color=C_INK)
    pad = 0.06 * max(abs(min(vals)), abs(max(vals))) / 0.14
    ax.set_xlim(min(min(vals), 0) - pad, max(max(vals), 0) + pad)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\hat{s}(80)$ = RGB-only $-$ RGB-D", color=C_INK, fontsize=8.5)
    ax.set_title(r"(b) effect of removing depth at $n=80$", fontsize=8.5, color=C_INK, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_depth_ablation.png", dpi=400, facecolor="white")
    fig.savefig(f"{OUT}/fig_depth_ablation.pdf", facecolor="white")
    plt.close(fig)

    print(f"[PAPER-MATERIALS-DONE] 매크로 {len(macros)}개 · 표/본문/그림(png+pdf) → {OUT}")
    for m in macros:
        print(f"  \\Num{{{m['macro']}}} = {m['formatted']}")


if __name__ == "__main__":
    main()
