"""E5 seed 판독 그림 — results/e5/의 JSON·JSONL에서만 읽는다 (수치 수동 입력 금지).

fig1 시스템 거동 : r_V 궤적 / 시스템 성공률 / lifecycle 상태 구성
fig2 기전        : 형성 이분(n별 probe) / 강등 사후확률 궤적 / 그림자 관할 예측 대 실측

색 규칙: 상태는 범주형(고정 순서, 순환 금지), 단일 계열에는 범례를 두지 않는다.
축은 단일 축만 사용한다(이중 축 금지) — 척도가 다른 양은 패널을 분리한다.
산출: results/e5/fig_e5_s{seed}_behavior.png, fig_e5_s{seed}_mechanism.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)

# 한글 라벨이 두부(□)로 깨지면 그림이 판독 불가가 된다 — 설치 폰트를 명시 지정.
matplotlib.rcParams["font.family"] = "NanumBarunGothic"
matplotlib.rcParams["axes.unicode_minus"] = False   # 마이너스 기호가 폰트에 없어 깨지는 것 방지

# CVD 안전 범주형 (고정 순서). 상태 4종 + 강조/보조.
C = {"M": "#1f6fb2", "I": "#d2731a", "X": "#6b7280", "U": "#c3c8ce",
     "line": "#1f6fb2", "ok": "#2a9d8f", "warn": "#b3401f",
     "grid": "#dfe3e8", "ink": "#1c2530", "muted": "#5b6672"}
STATES = ["M", "I", "X", "U"]
LABEL = {"M": "M 성숙(발화)", "I": "I 기지-미성숙", "X": "X 습관 부적격", "U": "U 미지"}


def style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=C["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    args = ap.parse_args()
    rd = os.path.join(HABIT2, "results", "e5")
    d = json.load(open(os.path.join(rd, f"reading_{args.seed_idx}.json")))
    rows = [json.loads(l) for l in open(os.path.join(rd, f"stream_{args.seed_idx}.jsonl"))]
    n = len(rows)

    # ---------------- fig1: 시스템 거동
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    traj = d["r_V_trajectory_bin200"]
    xs = [t["ep_start"] + 100 for t in traj]

    ax = axes[0]
    style(ax)
    h4a = d["H4a_call_rate_reduction"]
    ax.axvspan(0, 1000, color=C["line"], alpha=0.07, zorder=1)
    ax.axvspan(n - 1000, n, color=C["line"], alpha=0.07, zorder=1)
    ax.plot(xs, [t["r_V"] for t in traj], color=C["line"], lw=2, zorder=3)
    for x0, x1, p, lab in [(0, 1000, h4a["p_first"], "첫 1,000 ep"),
                           (n - 1000, n, h4a["p_last"], "끝 1,000 ep")]:
        ax.hlines(p, x0, x1, color=C["warn"], lw=2, zorder=4)
        ax.annotate(f"{lab}\n{p:.3f}", ((x0 + x1) / 2, p), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color=C["warn"], weight="bold")
    ax.set_ylabel("VLA 호출률 $r_V$", color=C["ink"], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"E5 seed {args.seed_idx+1} — 시스템 거동 (4,000 ep 스트림, 200 ep 구간 평균)",
                 fontsize=12, color=C["ink"], weight="bold", loc="left", pad=14)
    ax.text(0.99, 0.06, f"사전등록 단측 검정: Δ={h4a['diff']:.3f}, z={h4a['z']}, "
                        f"p={h4a['p_report']} → {h4a['verdict']}",
            transform=ax.transAxes, ha="right", fontsize=9, color=C["muted"])

    ax = axes[1]
    style(ax)
    ov = d["overview"]
    ax.plot(xs, [t["success"] for t in traj], color=C["ok"], lw=2, zorder=3)
    ax.axhline(ov["system_success_rate"], color=C["muted"], lw=1.2, ls="--", zorder=2)
    ax.annotate(f"전체 {ov['system_success_rate']:.3f}", (n, ov["system_success_rate"]),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=9, color=C["muted"])
    ax.set_ylabel("시스템 성공률", color=C["ink"], fontsize=10)
    ax.set_ylim(0.8, 1.02)
    ax.text(0.01, 0.08, f"발화 {ov['n_fire']}건 성공률 {ov['fire_success_rate']:.3f} · "
                        f"VLA {ov['n_vla']}건 {ov['vla_success_rate']:.3f}",
            transform=ax.transAxes, fontsize=9, color=C["muted"])

    ax = axes[2]
    style(ax)
    # 상태 구성 시계열: 각 구간 끝에서 클러스터별 최신 상태를 집계
    B = 200
    last, series = {}, {s: [] for s in STATES}
    for s0 in range(0, n, B):
        for r in rows[s0:s0 + B]:
            last[r["cluster"]] = r["lifecycle_state"]
        for s in STATES:
            series[s].append(sum(1 for v in last.values() if v == s))
    edges = [0] + [t["ep_start"] + B for t in traj]
    for s in STATES:                       # 스트림 시작(전 클러스터 미노출)을 좌단에 붙인다
        series[s].insert(0, 0)
    ax.stackplot(edges, [series[s] for s in STATES],
                 colors=[C[s] for s in STATES], labels=[LABEL[s] for s in STATES],
                 edgecolor="white", lw=0.8, zorder=3)
    ax.set_ylabel("클러스터 수", color=C["ink"], fontsize=10)
    ax.set_xlabel("스트림 에피소드", color=C["ink"], fontsize=10)
    ax.set_xlim(0, n)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=4,
              bbox_to_anchor=(0, 1.16))
    fig.tight_layout()
    p1 = os.path.join(rd, f"fig_e5_s{args.seed_idx}_behavior.png")
    fig.savefig(p1, dpi=170, facecolor="white")
    plt.close(fig)

    # ---------------- fig2: 기전
    # 그림자 관할의 두 지표(%p, ms)는 단위가 달라 한 축에 올리면 오도된다 — 패널을 분리한다.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    axes = axes.ravel()

    ax = axes[0]
    style(ax)
    fm = d["formation_ledger"]
    ev = [r["retrain_event"] for r in rows if r["retrain_event"]]
    ns = sorted({e["n"] for e in ev})
    passed = [sum(1 for e in ev if e["n"] == k and e["passed"]) for k in ns]
    failed = [sum(1 for e in ev if e["n"] == k and not e["passed"]) for k in ns]
    xpos = np.arange(len(ns))
    ax.bar(xpos, failed, width=0.55, color=C["I"], label="probe 미통과", zorder=3)
    ax.bar(xpos, passed, width=0.55, bottom=failed, color=C["M"], label="probe 통과", zorder=3)
    for i, k in enumerate(ns):
        ax.annotate(f"미통과 {failed[i]} · 통과 {passed[i]}", (i, failed[i] + passed[i]),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=9, color=C["ink"], weight="bold")
    ax.set_xticks(xpos, [f"n = {k}\n통과율 {fm['by_grid_n'][str(k)]['pass_rate']:.3f}" for k in ns])
    ax.set_ylabel("재학습 횟수", color=C["ink"], fontsize=10)
    ax.set_ylim(0, max(f + p for f, p in zip(failed, passed)) * 1.28)
    ax.set_title(f"형성: 학습 규모별 probe 결과\n(총 {fm['n_retrain']}회, 통과율 {fm['pass_rate']:.3f})",
                 fontsize=10.5, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

    ax = axes[1]
    style(ax)
    dem = d["demotions"]
    delta = 0.1   # (tau, delta) = (0.8, 0.1) -> 성숙 문턱 = 1 - delta
    # 강등 건수는 seed마다 다르다 — 고정 5색까지 개별 식별하고, 그 이상은 배경 회색으로
    # 겹쳐 그린다(순환 배색 금지). seed 1에서 3색 리스트가 IndexError를 낸 자리.
    PALETTE = [C["M"], C["I"], C["warn"], "#7b4f9d", "#2a9d8f"]
    for i, e in enumerate(dem["events"]):
        cr = [r for r in rows if r["cluster"] == e["cluster"]]
        seq = [r for r in cr if r["decision_reason"] == "fire" and r["t"] <= e["t"]]
        named = i < len(PALETTE)
        ax.plot(range(1, len(seq) + 1), [r["p_ge_tau"] for r in seq],
                marker="o" if named else None, ms=5, lw=1.8 if named else 1.0,
                color=PALETTE[i] if named else "#b8bec6",
                label=e["cluster"].replace("libero_", "") if named else None,
                zorder=3 if named else 2)
    ax.axhline(1 - delta, color=C["ink"], lw=1.4, ls="--", zorder=4)
    ax.annotate(f"성숙 문턱 $1-\\delta$ = {1-delta:.1f}", (1, 1 - delta), textcoords="offset points",
                xytext=(4, -14), fontsize=9, color=C["ink"])
    ax.set_xlabel("발화 회차", color=C["ink"], fontsize=10)
    ax.set_ylabel(r"$\Pr(s_k \geq \tau \mid \mathcal{D}_k)$", color=C["ink"], fontsize=10)
    extra = max(0, dem['n_demotions'] - len(PALETTE))
    ax.set_title(f"강등 {dem['n_demotions']}건: 성숙 직후 단일 실패로\n문턱 하회 "
                 f"(재형성 성공 {dem['n_regained']}건" +
                 (f", 회색 {extra}건 미표기" if extra else "") + ")",
                 fontsize=10.5, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 0.02))

    sj = d["shadow_jurisdiction_counterfactual"]
    pv = sj["prediction_vs_observed"]
    dg = sj["divergence_diagnosis"]

    ax = axes[2]
    style(ax)
    vals = [pv["routing_increase_pp"]["predicted"], pv["routing_increase_pp"]["observed"]]
    ax.bar([0, 1], vals, width=0.5, color=[C["U"], C["M"]], zorder=3)
    for x, v in zip([0, 1], vals):
        ax.annotate(f"+{v:g}%p", (x, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=10, color=C["ink"], weight="bold")
    ax.set_xticks([0, 1], ["사전 예측 (§5)", "실측"])
    ax.set_ylabel("VLA 라우팅 증가 (%p)", color=C["ink"], fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("그림자 관할 ①: 관할을 켰다면 늘었을 VLA 라우팅\n"
                 f"실측 발화 기각률 {dg['observed_stream_reject_rate']:.3f} ≈ "
                 f"재보정 FR {dg['recalibrated_fr_reference'].get('mean_fr')} "
                 f"(예측 유도에 쓰인 원 보정 기각률 {dg['e4r_reject_rate_at_w001_original_q']})",
                 fontsize=10.5, color=C["ink"], loc="left")

    ax = axes[3]
    style(ax)
    ob, pr = sj["observed"], sj["prereg_prediction"]
    labs = ["관할 OFF\n(공통 기준)", "관할 ON\n사전 예측", "관할 ON\n실측"]
    vals = [ob["query_latency_off_ms"], pr["latency_on_ms"], ob["query_latency_on_ms"]]
    ax.bar(range(3), vals, width=0.5, color=[C["X"], C["U"], C["M"]], zorder=3)
    for x, v in enumerate(vals):
        ax.annotate(f"{v:.2f} ms", (x, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=10, color=C["ink"], weight="bold")
    ax.set_xticks(range(3), labs, fontsize=9)
    ax.set_ylabel("질의당 지연 (ms)", color=C["ink"], fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("그림자 관할 ②: 발화 후보의 질의당 지연\n"
                 f"OFF = ACT+gate 보수 하한, ON = 기각분만 OFT로 라우팅 "
                 f"(예측 {pr['latency_ratio']}× vs 실측 {ob['latency_ratio']}×)",
                 fontsize=10.5, color=C["ink"], loc="left")

    fig.tight_layout()
    p2 = os.path.join(rd, f"fig_e5_s{args.seed_idx}_mechanism.png")
    fig.savefig(p2, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"[E5FIG-DONE] {p1}\n[E5FIG-DONE] {p2}")


if __name__ == "__main__":
    main()
