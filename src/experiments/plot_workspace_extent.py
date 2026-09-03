"""작업공간 산점도 (§5 등재 산출물 4) — results/e4/workspace_extent.json 단일 진입점.

레이어: 테이블 상판 경계 / 공식 배치 분포 / 검증 도달 영역(convex hull) / w 섭동 원.
색: 검증된 기본 팔레트 슬롯 1·2·3 순서 사용 (blue #2a78d6, orange #eb6834, aqua #1baf7a).
    ※ 팔레트 validator(node)는 이 워크스테이션에 node 미설치로 미실행 — 검증 인스턴스의
      고정 순서를 그대로 사용해 위험을 회피했음을 명기.
실행: hv2_hab python -u experiments/plot_workspace_extent.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

for _p in ("/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
           "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf"):
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(HABIT2, "results", "e4", "workspace_extent.json")
OUT = os.path.join(HABIT2, "results", "e4", "fig_workspace_extent.png")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"


def main():
    d = json.load(open(SRC))
    P = np.array(d["plot_data"]["official_xy_sample"])
    V = np.array(d["plot_data"]["verified_xy_sample"])
    H = np.array(d["plot_data"]["hull_vertices_xy"])
    vr, sc = d["verified_reach"], d["scene"]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # 테이블 상판은 MuJoCo geom으로 노출되지 않음(floor만 존재) → 공식 배치 bbox를
    # 실측 경계로 사용 (문헌값 대체 원칙). 미취득 사실은 캡션에 명기.
    bb = d["official_placement_bbox"]
    ax.add_patch(Rectangle((bb["x_range"][0], bb["y_range"][0]), bb["x_span"], bb["y_span"],
                           fill=False, ec=GRID, lw=2, ls="--", zorder=1,
                           label=f"공식 배치 bbox ({bb['x_span']:.2f}×{bb['y_span']:.2f} m)"))
    ax.scatter(P[:, 0], P[:, 1], s=5, c=BLUE, alpha=0.28, lw=0, zorder=2,
               label=f"공식 init 배치 (n={d['official_placement_bbox']['n_points']})")
    hull = np.vstack([H, H[:1]])
    ax.fill(hull[:, 0], hull[:, 1], color=AQUA, alpha=0.10, zorder=3)
    ax.plot(hull[:, 0], hull[:, 1], color=AQUA, lw=2, zorder=4,
            label=f"검증 도달 영역 (hull {vr['convex_hull_area_m2']:.3f} m²)")

    # w 섭동 원 — 배치 중앙에 실척으로 겹쳐 크기 대비를 보이게
    cx0, cy0 = float(np.median(V[:, 0])), float(np.median(V[:, 1]))
    for w, ls in ((0.01, ":"), (0.04, "-"), (0.08, "--")):
        ax.add_patch(Circle((cx0, cy0), w, fill=False, ec=ORANGE, lw=2, ls=ls, zorder=5))
        ax.annotate(f"w={w}", (cx0 + w * 0.72, cy0 + w * 0.72), color=ORANGE,
                    fontsize=9, zorder=6)
    ax.plot([], [], color=ORANGE, lw=2, label="섭동 폭 w (0.01 / 0.04 / 0.08 m)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=INK2)
    ax.set_ylabel("y (m)", color=INK2)
    ax.set_title("작업공간 실측: 공식 배치 · 검증 도달 영역 · 섭동 폭 대비",
                 color=INK, fontsize=12, pad=34)
    rb = sc.get("robot_base_xy")
    sub = (f"w=0.04 섭동 지름 = 검증 도달 등가지름의 {d['w_vs_reach']['0.04']['pct_of_equiv_diameter']}%"
           f" (면적 {d['w_vs_reach']['0.04']['area_pct_of_hull']}%)\n"
           + (f"로봇 base x={rb[0]} m (좌측 축 밖) · " if rb else "")
           + "테이블 상판 geom 미노출 → 공식 배치 bbox로 대체")
    ax.annotate(sub, xy=(0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom",
                color=INK2, fontsize=8.5, linespacing=1.4)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False,
              fontsize=9, labelcolor=INK2)
    fig.subplots_adjust(top=0.85, bottom=0.16)
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    print(f"[WS-FIG] {OUT} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
