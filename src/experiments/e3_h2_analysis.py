"""§4f 판정 분석 (R4) — N* 분산 분해 + 공변량 회귀. 입력 = e3_curves.json + covariates.json.

사전등재 추정량 (preregistration §4f):
  주 분석 = **순위 기반** (우측 절단 >80 = 공동 최상위 순위):
    (i) 레벨(L1/2·L3·L4a) 간/내 순위 분산 분해 + Kruskal–Wallis
    (ii) 공변량: Spearman + 순위 응답 OLS(suite 더미 포함), 순열 p (B=10⁴, seed 0)
  보조 = **구간(interval) 민감도**: N* 구간 [10→(0,10] … >80→(80,160] cap]의
    하한/중점/상한 대입 OLS 밴드 (log 스케일). cap=160은 문서화된 선택.
표본: 주 = 형성 셀 22 (L1/2+L3+L4a) / 민감도 = 표준 25 (suite 더미가 horizon 흡수 — 문서화).
출력: results/e3/h2_analysis.json (입력이 PARTIAL이면 status=DRY_RUN 명기).

실행: hv2_hab python -u experiments/e3_h2_analysis.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import kruskal, rankdata, spearmanr

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
OUT = os.path.join(HABIT2, "results", "e3", "h2_analysis.json")
B_PERM = 10_000
CENSOR_CAP = 160  # >80 구간 상한 (보조 분석용 — 문서화된 선택)
LEVEL_OF = {"libero_object": "L1_2", "libero_goal": "L3", "libero_spatial": "L4a"}
INTERVAL = {10: (1, 10), 20: (10, 20), 40: (20, 40), 80: (40, 80), ">80": (80, CENSOR_CAP)}


def nstar_value(v):
    return ">80" if v == ">80" else int(v)


def load_rows():
    curves = json.load(open(os.path.join(HABIT2, "results", "e3", "e3_curves.json")))
    cov = json.load(open(os.path.join(HABIT2, "results", "e3", "covariates.json")))
    rows = []
    for cl, e in curves["clusters"].items():
        if "curve" not in e or cl.startswith("chained_"):
            continue
        c = cov["clusters"].get(cl)
        if not c:
            continue
        rows.append({
            "cluster": cl, "suite": c["suite"], "level": LEVEL_OF.get(c["suite"], "T3"),
            "N_star": nstar_value(e["N_star"]),
            "free_joints": c["free_joints"], "S_V_cluster": c["S_V_cluster"],
            "median_len": c["median_len_success"],
        })
    return rows, curves.get("status", "?")


def censored_ranks(nstars):
    """>80 = 공동 최상위. 그리드 값은 수치 순위 (average ties)."""
    numeric = [CENSOR_CAP if v == ">80" else v for v in nstars]
    return rankdata(numeric)


def rank_decomposition(rows):
    """레벨 간/내 순위 분산 분해 + Kruskal–Wallis (형성 셀만)."""
    formation = [r for r in rows if r["level"] in ("L1_2", "L3", "L4a")]
    ranks = censored_ranks([r["N_star"] for r in formation])
    groups, out_groups = {}, {}
    for r, rk in zip(formation, ranks):
        groups.setdefault(r["level"], []).append(rk)
    grand = float(np.mean(ranks))
    ss_total = float(np.sum((ranks - grand) ** 2))
    ss_between = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values()))
    for lv, g in sorted(groups.items()):
        vals = [r["N_star"] for r in formation if r["level"] == lv]
        out_groups[lv] = {"n": len(g), "mean_rank": round(float(np.mean(g)), 2),
                          "N_star_values": vals,
                          "n_censored": sum(1 for v in vals if v == ">80")}
    kw = kruskal(*groups.values()) if len(groups) >= 2 and min(len(g) for g in groups.values()) >= 2 else None
    return {
        "n": len(formation),
        "between_share": round(ss_between / ss_total, 4) if ss_total else None,
        "within_share": round(1 - ss_between / ss_total, 4) if ss_total else None,
        "kruskal_H": round(float(kw.statistic), 4) if kw else None,
        "kruskal_p": round(float(kw.pvalue), 4) if kw else None,
        "groups": out_groups,
    }


def design_matrix(rows):
    suites = sorted({r["suite"] for r in rows})
    cols = ["free_joints", "S_V_cluster", "median_len"] + [f"suite:{s}" for s in suites[1:]]
    X = []
    for r in rows:
        x = [r["free_joints"], r["S_V_cluster"], r["median_len"]]
        x += [1.0 if r["suite"] == s else 0.0 for s in suites[1:]]
        X.append(x)
    X = np.asarray(X, float)
    # 연속 공변량 표준화 (계수 비교 가능성)
    for j in range(3):
        sd = X[:, j].std()
        X[:, j] = (X[:, j] - X[:, j].mean()) / (sd if sd > 0 else 1.0)
    return np.column_stack([np.ones(len(X)), X]), ["intercept"] + cols


def ols_perm(X, y, rng):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    null = np.empty((B_PERM, len(beta)))
    for b in range(B_PERM):
        null[b] = np.linalg.lstsq(X, rng.permutation(y), rcond=None)[0]
    p = ((np.abs(null) >= np.abs(beta)).sum(0) + 1) / (B_PERM + 1)
    return beta, p


def regression(rows, tag):
    y = censored_ranks([r["N_star"] for r in rows])
    X, names = design_matrix(rows)
    rng = np.random.default_rng(0)
    beta, p = ols_perm(X, y, rng)
    spear = {}
    for k in ("free_joints", "S_V_cluster", "median_len"):
        rho, sp = spearmanr([r[k] for r in rows], y)
        spear[k] = {"rho": round(float(rho), 4), "p": round(float(sp), 4)}
    # 보조: 구간 민감도 (log 중점/하한/상한 대입 OLS)
    sens = {}
    for which, idx in (("lower", 0), ("upper", 1), ("mid", None)):
        yv = []
        for r in rows:
            lo, hi = INTERVAL[r["N_star"]]
            v = np.sqrt(lo * hi) if idx is None else (lo, hi)[idx]
            yv.append(np.log(max(v, 1)))
        b2 = np.linalg.lstsq(X, np.asarray(yv), rcond=None)[0]
        sens[which] = {n: round(float(v), 4) for n, v in zip(names, b2)}
    return {
        "tag": tag, "n": len(rows),
        "rank_ols": {n: {"beta": round(float(b), 4), "perm_p": round(float(q), 4)}
                     for n, b, q in zip(names, beta, p)},
        "spearman_vs_rank": spear,
        "interval_sensitivity_log_ols": sens,
    }


def collinearity_appendix(rows_formation):
    """1a (연구원 지시 2026-08-16): 공선성 방어 부록 — 해석 변경 없음.
    corr = 27 공변량 테이블(chained 포함) / VIF = 형성22 설계행렬 기준."""
    from scipy.stats import pearsonr

    cov = json.load(open(os.path.join(HABIT2, "results", "e3", "covariates.json")))
    rows27 = [{"free_joints": v["free_joints"], "S_V_cluster": v["S_V_cluster"],
               "median_len": v["median_len_success"]} for v in cov["clusters"].values()]
    keys = ("free_joints", "S_V_cluster", "median_len")
    pairwise = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            xa, xb = [r[a] for r in rows27], [r[b] for r in rows27]
            pr, pp = pearsonr(xa, xb)
            sr, sp = spearmanr(xa, xb)
            pairwise[f"{a}~{b}"] = {"pearson": round(float(pr), 4), "p_pearson": round(float(pp), 4),
                                    "spearman": round(float(sr), 4), "p_spearman": round(float(sp), 4)}
    X, names = design_matrix(rows_formation)
    vif = {}
    for nm in keys:
        j = names.index(nm)
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        beta = np.linalg.lstsq(Xo, y, rcond=None)[0]
        resid = y - Xo @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
        vif[nm] = round(1.0 / max(1e-9, 1 - r2), 2)
    flags = [f"{k}: VIF>10 — 설계행렬 내 (준)완전 공선, 해당 계수는 이 표본에서 식별 불가"
             for k, v in vif.items() if v > 10]
    return {"note": "방어용 부록 (해석 변경 없음) — corr: 27 공변량 테이블, VIF: 형성22 설계행렬",
            "n_corr_rows": len(rows27), "pairwise": pairwise, "vif_formation_design": vif,
            "identifiability_flags": flags or None}


def main():
    rows, curves_status = load_rows()
    formation = [r for r in rows if r["level"] in ("L1_2", "L3", "L4a")]
    report = {
        "input_curves_status": curves_status,
        "status": "DRY_RUN" if curves_status != "COMPLETE" else "FINAL",
        "estimators": "preregistration §4f: rank-primary (censored=top ties), interval-secondary "
                      f"(cap={CENSOR_CAP}); perm B={B_PERM} seed 0",
        "decomposition_L": rank_decomposition(rows),
        "regression_formation22": regression(formation, "formation cells (L1_2+L3+L4a)"),
        "regression_all_standard": regression(rows, "all standard incl. libero_10 (suite dummy)"),
        "collinearity_appendix": collinearity_appendix(formation),
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    d = report["decomposition_L"]
    print(f"[H2-ANALYSIS-{report['status']}] n={len(rows)} between={d['between_share']} "
          f"KW p={d['kruskal_p']} -> {os.path.relpath(OUT, HABIT2)}")


if __name__ == "__main__":
    main()
