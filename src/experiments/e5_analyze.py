"""E5 스트림 판독기 — 사전등록된 판정 규칙만 집행한다 (수치 수동 입력 금지).

판정 규칙 출처 (configs/preregistration.md):
  §1 표 | E5 비열등 (H4)   = paired bootstrap 95% CI, margin −3%p
  §1 표 | E5 호출률 감소   = 첫 1,000 ep vs 끝 1,000 ep, 단측
  §4h   | 시간 회계 3장부  = 운영(지연 주장 유일 근거) / 형성(별도 보고) / 평가(미보고)
  §5    | 그림자 관할 반사실 = 사전 예측치 대조 (추가 rollout 0, shadow_jur 로그만)
  §3.5  | 위험 통제 Pr(fail|fire) ≤ ε = 0.2 (ACI 추적)

counterfactual(cf_{seed}.jsonl)이 있으면 비열등·관할 반사실 성공률 절까지 완성하고,
없으면 스트림 단독 절만 산출한다(1차 판독). 재실행 시 자동으로 완성본이 된다.

산출: results/e5/reading_{seed}.json
실행: hv2_hab python -u experiments/e5_analyze.py --seed-idx 0
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)

EPS_RISK = 0.2       # §3.5 Pr(fail|fire) 상한
NI_MARGIN = -0.03    # §1 비열등 margin −3%p
BOOT_B = 10_000
BOOT_SEED = 0
CHUNK = 8


def load_prereg_prediction():
    """§5 그림자 관할 사전 예측치를 사전등록 원문에서 추출 (하드코딩 금지)."""
    txt = open(os.path.join(HABIT2, "configs", "preregistration.md")).read()
    line = next((l for l in txt.splitlines()
                 if "그림자 관할 반사실" in l and "사전 예측치" in l), None)
    if line is None:
        raise SystemExit("[E5AN-FAIL] §5 그림자 관할 사전 예측치 라인을 찾지 못함")
    def grab(pat):
        m = re.search(pat, line)
        if not m:
            raise SystemExit(f"[E5AN-FAIL] 예측치 추출 실패: {pat}")
        return float(m.group(1))
    return {
        "conditional_gain_per_ep": grab(r"조건부 이득 \+([\d.]+)/ep"),
        "vla_routing_increase_pp": grab(r"VLA 라우팅 비율 \+(\d+)%p"),
        "latency_ratio": grab(r"평균 질의 지연 ([\d.]+)×"),
        "latency_on_ms": grab(r"\(([\d.]+) ms vs"),
        "latency_off_ms": grab(r"vs ([\d.]+) ms\)"),
        "source": "configs/preregistration.md §5 2026-08-16 (E5 판정 7)",
    }


def two_proportion_one_sided(x1, n1, x2, n2):
    """H0: p1 ≤ p2 vs H1: p1 > p2 (첫 구간 호출률이 끝 구간보다 높다 = 감소 실증)."""
    from scipy.stats import norm
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pv = float(norm.sf(z))
    # z가 크면 sf가 언더플로로 정확히 0이 된다 — "p=0"은 부정확한 보고이므로
    # 로그 스케일 상한을 함께 싣는다 (log10 sf는 언더플로하지 않는다).
    return {"p_first": round(p1, 4), "p_last": round(p2, 4), "diff": round(p1 - p2, 4),
            "z": round(float(z), 3), "p_value": pv,
            "log10_p": round(float(norm.logsf(z) / np.log(10)), 2),
            "p_report": (f"{pv:.3g}" if pv > 0 else
                         f"< 1e-{int(-np.floor(norm.logsf(z) / np.log(10)))}")}


def paired_bootstrap(sys_succ, vla_succ, b=BOOT_B, seed=BOOT_SEED):
    """에피소드 단위 paired 재표집 → (system − full-VLA) 성공률 차 95% CI."""
    rng = np.random.default_rng(seed)
    s, v = np.asarray(sys_succ, float), np.asarray(vla_succ, float)
    n = len(s)
    idx = rng.integers(0, n, size=(b, n))
    d = s[idx].mean(1) - v[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n_paired_episodes": n,
            "system_rate": round(float(s.mean()), 4),
            "full_vla_rate": round(float(v.mean()), 4),
            "diff": round(float(s.mean() - v.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "margin": NI_MARGIN, "B": b, "seed": seed,
            "noninferior": bool(lo > NI_MARGIN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    args = ap.parse_args()
    rd = os.path.join(HABIT2, "results", "e5")
    rows = [json.loads(l) for l in open(os.path.join(rd, f"stream_{args.seed_idx}.jsonl"))]
    summary = json.load(open(os.path.join(rd, f"summary_{args.seed_idx}.json")))
    lat = json.load(open(os.path.join(HABIT2, "results", "e1", "e1_latency.json")))
    ms_vla = lat["anchor1_oft_chunk_forward"]["median_ms"]
    ms_act = lat["anchor2_act_forward"]["median_ms"]
    ms_gate = lat["anchor3_gate_path"]["median_ms"]
    train_s = lat["anchor5_act_train_n40"]["train_seconds"]
    vla_eq = lat["anchor5_act_train_n40"]["vla_call_equivalents"]

    out = {"seed_idx": args.seed_idx, "n_episodes": len(rows),
           "prereg_rules": {
               "noninferiority": "paired bootstrap 95% CI, margin −3%p (§1)",
               "call_rate": "첫 1,000 ep vs 끝 1,000 ep, 단측 (§1)",
               "risk": f"Pr(fail|fire) ≤ ε = {EPS_RISK} (§3.5, ACI 추적)",
               "ledgers": "운영/형성/평가 3장부 (§4h)"}}

    # ---- A. 개요
    ok = [r for r in rows if r["outcome"] != "infra_error"]
    fire = [r for r in ok if r["executor"] == "habit"]
    vla = [r for r in ok if r["executor"] == "vla"]
    out["overview"] = {
        "n_valid": len(ok), "n_infra_error": len(rows) - len(ok),
        "n_fire": len(fire), "n_vla": len(vla),
        "r_V_overall": round(len(vla) / len(ok), 4),
        "system_success_rate": round(sum(r["outcome"] == "success" for r in ok) / len(ok), 4),
        "fire_success_rate": round(sum(r["outcome"] == "success" for r in fire) / max(len(fire), 1), 4),
        "vla_success_rate": round(sum(r["outcome"] == "success" for r in vla) / max(len(vla), 1), 4),
        "decision_reason": {k: sum(1 for r in rows if r["decision_reason"] == k)
                            for k in sorted({r["decision_reason"] for r in rows})}}

    # ---- B. H4-a 호출률 감소 (사전등록 검정)
    first, last = rows[:1000], rows[-1000:]
    x1 = sum(1 for r in first if r["executor"] == "vla")
    x2 = sum(1 for r in last if r["executor"] == "vla")
    test = two_proportion_one_sided(x1, len(first), x2, len(last))
    out["H4a_call_rate_reduction"] = {**test, "alpha": 0.05,
                                      "verdict": "PASS" if test["p_value"] < 0.05 else "FAIL"}

    # ---- C. r_V 궤적 (200 ep 빈)
    B = 200
    traj = []
    for s in range(0, len(rows), B):
        w = rows[s:s + B]
        wo = [r for r in w if r["outcome"] != "infra_error"]
        traj.append({"ep_start": s,
                     "r_V": round(sum(1 for r in wo if r["executor"] == "vla") / len(wo), 4),
                     "success": round(sum(1 for r in wo if r["outcome"] == "success") / len(wo), 4)})
    out["r_V_trajectory_bin200"] = traj

    # ---- D. lifecycle: 최종 상태 + 스트림 수준 성숙 소요 (이원 보고 ②)
    clusters = sorted({r["cluster"] for r in rows})
    life = {}
    for cl in clusters:
        cr = [r for r in rows if r["cluster"] == cl]
        first_fire = next((r for r in cr if r["decision_reason"] == "fire"), None)
        life[cl] = {
            "final_state": summary["final_states"][cl],
            "exposures": len(cr),
            "is_novel_pool": bool(cr[0]["is_novel_injection"]),
            "stream_eps_to_maturity": first_fire["t"] if first_fire else None,
            "exposures_to_maturity": (cr.index(first_fire) + 1) if first_fire else None,
            "n_fire": sum(1 for r in cr if r["decision_reason"] == "fire"),
            "fire_success": round(
                sum(1 for r in cr if r["decision_reason"] == "fire" and r["outcome"] == "success")
                / max(sum(1 for r in cr if r["decision_reason"] == "fire"), 1), 4),
            "final_bc_pool": cr[-1]["bc_pool"], "final_tau": cr[-1]["tau"]}
    out["lifecycle"] = life
    mat = [v for v in life.values() if v["stream_eps_to_maturity"] is not None]
    out["maturity_dual_report"] = {
        "n_reached_maturity": len(mat),
        "n_clusters": len(clusters),
        "stream_eps_to_maturity_median": int(np.median([v["stream_eps_to_maturity"] for v in mat])) if mat else None,
        "exposures_to_maturity_median": int(np.median([v["exposures_to_maturity"] for v in mat])) if mat else None,
        "exposures_to_maturity_range": [min(v["exposures_to_maturity"] for v in mat),
                                        max(v["exposures_to_maturity"] for v in mat)] if mat else None,
        "note": "이원 보고 ② — 클러스터 수준 N*(k)는 E3 배치 곡선(별도)"}

    # ---- E. 형성 회계 (§4h 형성 장부)
    ev = [r["retrain_event"] for r in rows if r["retrain_event"]]
    out["formation_ledger"] = {
        "n_retrain": len(ev),
        # 시도 수만 담으면 "n=20은 전부 실패" 같은 오독이 생긴다 — 시도/통과를 함께 싣는다.
        "by_grid_n": {str(n): {"attempts": sum(1 for e in ev if e["n"] == n),
                               "passed": sum(1 for e in ev if e["n"] == n and e["passed"]),
                               "pass_rate": round(sum(1 for e in ev if e["n"] == n and e["passed"])
                                                  / sum(1 for e in ev if e["n"] == n), 4)}
                      for n in sorted({e["n"] for e in ev})},
        "by_probe_round": {str(r_): {"attempts": sum(1 for e in ev if e["probe_round"] == r_),
                                     "passed": sum(1 for e in ev if e["probe_round"] == r_ and e["passed"])}
                           for r_ in sorted({e["probe_round"] for e in ev})},
        "n_passed": sum(1 for e in ev if e["passed"]),
        "pass_rate": round(sum(1 for e in ev if e["passed"]) / max(len(ev), 1), 4),
        "formation_wall_s": summary["ledger_s"]["formation_s"],
        "formation_episodes": summary["ledger_s"]["formation_episodes"],
        "act_train_anchor_s": train_s, "vla_call_equivalents_per_retrain": vla_eq,
        "note": "지연 주장에 불산입 — 별도 보고 (§4h 3장부)"}

    # ---- F. 위험 통제 (§3.5)
    n_fail_fire = sum(1 for r in fire if r["outcome"] == "fail")
    from scipy.stats import binomtest
    out["risk_control"] = {
        "pr_fail_given_fire": round(n_fail_fire / max(len(fire), 1), 4),
        "epsilon": EPS_RISK,
        "within_bound": bool(n_fail_fire / max(len(fire), 1) <= EPS_RISK),
        "ci95_wilson": [round(float(x), 4) for x in
                        binomtest(n_fail_fire, len(fire)).proportion_ci(0.95, method="wilson")],
        "tau_final_max": max(r["tau"] for r in rows),
        "tau_raised_episodes": sum(1 for r in rows if r["tau"] > 0.8),
        "clusters_with_tau_raised": sorted({r["cluster"] for r in rows if r["tau"] > 0.8})}

    # ---- G. novel 주입 분리
    nv = [r for r in ok if r["is_novel_injection"]]
    rg = [r for r in ok if not r["is_novel_injection"]]
    out["novel_injection"] = {
        "n_novel": len(nv), "rate": round(len(nv) / len(ok), 4),
        "novel_r_V": round(sum(1 for r in nv if r["executor"] == "vla") / len(nv), 4),
        "regular_r_V": round(sum(1 for r in rg if r["executor"] == "vla") / len(rg), 4),
        "novel_success": round(sum(1 for r in nv if r["outcome"] == "success") / len(nv), 4),
        "regular_success": round(sum(1 for r in rg if r["outcome"] == "success") / len(rg), 4),
        "novel_clusters_matured": sorted(cl for cl, v in life.items()
                                         if v["is_novel_pool"] and v["final_state"] == "M")}

    # ---- H. 그림자 관할 반사실 (§5 예측치 대조)
    # 단위: 예측치와 동일 basis로 산출해야 대조가 성립한다. §5 예측치의 7.32 ms는
    # E1 ratios.basis.conservative_floor = anchor2 + anchor3 (habit 질의 + gate를 chunk마다
    # 계상한 보수 하한)이고, 31.51 ms는 그 위에 라우팅 비율 x로 VLA(anchor1)를 섞은 값이다.
    # → 대상 = **발화 후보 질의**, 척도 = **질의(chunk)당**. 에피소드 총지연으로 재면
    # 스텝 수가 지배해 예측치와 비교 불가능해진다.
    pred = load_prereg_prediction()
    have = [r for r in ok if r.get("shadow_jur")]
    rej_fire = [r for r in fire if r.get("shadow_jur") and not r["shadow_jur"]["accept"]]
    x_rej = len(rej_fire) / max(len(fire), 1)
    q_off = ms_act + ms_gate                              # 관할 OFF = 성숙도 게이트만
    q_on = x_rej * ms_vla + (1 - x_rej) * q_off           # 관할 ON = 기각분만 VLA로
    basis_check = round(abs(q_off - pred["latency_off_ms"]), 3)
    out["shadow_jurisdiction_counterfactual"] = {
        "prereg_prediction": pred,
        "unit_basis": {
            "scale": "질의(chunk)당 ms, 대상 = 발화 후보",
            "off_formula": "anchor2(ACT) + anchor3(gate) = conservative_floor",
            "on_formula": "x·anchor1(OFT) + (1−x)·off, x = 발화 중 관할 기각 비율",
            "off_reconstruction_error_ms": basis_check,
            "verified": basis_check < 0.05},
        "coverage": {"n_with_score": len(have), "n_without_score": len(ok) - len(have),
                     "reason_without": "관할 참조 분포(μ_k, Σ_k) 미적합 클러스터 — 원리상 산출 불가"},
        "observed": {
            "fire_shadow_reject_n": len(rej_fire),
            "fire_shadow_reject_rate": round(x_rej, 4),
            "r_V_off": round(len(vla) / len(ok), 4),
            "r_V_on": round((len(vla) + len(rej_fire)) / len(ok), 4),
            "vla_routing_increase_pp": round(100 * len(rej_fire) / len(ok), 2),
            "query_latency_off_ms": round(q_off, 2),
            "query_latency_on_ms": round(q_on, 2),
            "latency_ratio": round(q_on / q_off, 3)},
        "prediction_vs_observed": {
            "routing_increase_pp": {"predicted": pred["vla_routing_increase_pp"],
                                    "observed": round(100 * len(rej_fire) / len(ok), 2)},
            "latency_ratio": {"predicted": pred["latency_ratio"],
                              "observed": round(q_on / q_off, 3)}},
        "note": "추가 rollout 0 — shadow_jur 로그만으로 산출 (§5 등재)"}

    # 예측 빗나감의 원인 규명: 예측은 E4-R **원 보정** q 기준 기각률,
    # E5 그림자는 **재보정 절차** q(§5 2026-08-16) — 보정 절차가 다르면 기각률이 다르다.
    try:
        cm = json.load(open(os.path.join(HABIT2, "results", "e4", "e4r_competence_map.json")))
        e4r_rej = cm["reading"]["rule2_alignment"]["reject_rate_at_learned_w"]
    except Exception:
        e4r_rej = None
    sd = json.load(open(os.path.join(HABIT2, "results", "e4", "e4_scorer_diag.json"))) \
        if os.path.exists(os.path.join(HABIT2, "results", "e4", "e4_scorer_diag.json")) else {}
    recal = next((v for k, v in sd.get("rows", {}).items() if "recalibrated" in k), {})
    out["shadow_jurisdiction_counterfactual"]["divergence_diagnosis"] = {
        "e4r_reject_rate_at_w001_original_q": e4r_rej,
        "observed_stream_reject_rate": round(x_rej, 4),
        "hypothesis": "예측치는 E4-R 원 보정 q의 기각률에서 유도, E5 그림자는 재보정 q "
                      "(§5 2026-08-16: known held-out 50:50 분할, FR 0.264→0.06) 사용 — "
                      "보정 절차 불일치가 예측·실측 격차의 후보 원인",
        "recalibrated_fr_reference": {k: recal.get(k) for k in ("mean_fr", "mean_fr_before") if k in recal}}

    # ---- H2. r_V 구조적 하한 분해 (사후 분해 — 판정 근거 아님, 설명용)
    # 최종 상태를 고정했을 때 스트림 구성이 허용하는 최소 VLA 비율. M이 아닌 클러스터의
    # 노출은 전량 VLA이고, M 클러스터도 성숙 이전 노출은 VLA다.
    tail = [r for r in rows[-1000:] if r["outcome"] != "infra_error"]
    tail_vla = [r for r in tail if r["executor"] == "vla"]
    by_state = {s: sum(1 for r in tail_vla if r["lifecycle_state"] == s)
                for s in sorted({r["lifecycle_state"] for r in tail_vla})}
    # 잔여 VLA 호출이 "습관이 원리상 담당할 수 없는 상황"인지, "아직 형성 중"인지 분리
    out["r_V_tail_decomposition"] = {
        "kind": "post-hoc decomposition — 판정 근거 아님(H4a는 사전등록 단측 검정)",
        "window": "last 1000 ep",
        "r_V_observed": round(len(tail_vla) / len(tail), 4),
        "vla_calls_by_lifecycle_state": by_state,
        "share_of_window": {s: round(n / len(tail), 4) for s, n in by_state.items()},
        "state_meaning": {"X": "습관 부적격(R_max 소진) — 형성 실패로 확정, 재도전 없음",
                          "I": "기지-미성숙 — BC 풀 축적 중이거나 재학습 기회 소진",
                          "U": "미지 — 클러스터 신설 직후"},
        "ceiling_if_all_eligible_matured": round(
            1 - sum(v["exposures"] for v in life.values()) / len(rows)
            + sum(v["exposures"] for v in life.values() if v["final_state"] == "M") / len(rows), 4),
        "note": "후반 r_V 정체가 습관 성능 저하 때문인지, 잔여 VLA가 X·I 클러스터에 "
                "고정된 결과인지 구분하기 위한 분해"}

    # ---- H3. 강등(M→I) 사건 전수 + 성숙 초기 취약성
    dem = []
    for cl in clusters:
        cr = [r for r in rows if r["cluster"] == cl]
        for a, b in zip(cr, cr[1:]):
            if a["lifecycle_state"] == "M" and b["lifecycle_state"] == "I":
                prior = [r for r in cr if r["t"] <= a["t"] and r["decision_reason"] == "fire"]
                dem.append({"cluster": cl, "t": b["t"],
                            "fires_before_demotion": len(prior),
                            "fire_fails_before": sum(1 for r in prior if r["outcome"] == "fail"),
                            "sigma": a["sigma_k"], "phi": a["phi_k"],
                            "p_ge_tau": a["p_ge_tau"], "tau": a["tau"],
                            "regained_M": any(r["lifecycle_state"] == "M" for r in cr if r["t"] > b["t"])})
    out["demotions"] = {
        "n_demotions": len(dem), "events": dem,
        "n_regained": sum(1 for d in dem if d["regained_M"]),
        "median_fires_before_demotion": int(np.median([d["fires_before_demotion"] for d in dem])) if dem else None,
        "note": "성숙 직후 σ가 작을 때 단일 실패 + ACI τ 상향이 강등을 촉발하는지 — "
                "재초기화 계수 c=0.25가 σ를 압축하므로 성숙 초기가 구조적으로 취약하다는 가설의 시험"}

    # ---- I. H4-b 비열등 (CF 필요)
    cfp = os.path.join(rd, f"cf_{args.seed_idx}.jsonl")
    if os.path.exists(cfp):
        cf = {}
        for l in open(cfp):
            d = json.loads(l)
            if "teacher_success" in d:
                cf[d["uid"]] = d["teacher_success"]
        sys_s, vla_s, missing = [], [], 0
        for r in ok:
            s = r["outcome"] == "success"
            if r["executor"] == "habit":
                if r["uid"] not in cf:
                    missing += 1
                    continue
                sys_s.append(s), vla_s.append(cf[r["uid"]])
            else:
                sys_s.append(s), vla_s.append(s)   # 비발화는 VLA 실측이 곧 기준선 (§3)
        boot = paired_bootstrap(sys_s, vla_s)
        # CF가 미완이면 발화 에피소드가 표본에서 빠져 baseline이 system 쪽으로 붕괴한다
        # (누락분은 전부 "system=VLA=실측"인 비발화). 부분 표본 판정 금지 — 완료 시에만 확정.
        complete = missing == 0
        out["H4b_noninferiority"] = {
            **boot, "n_cf_missing": missing, "cf_complete": complete,
            "verdict": ("PASS" if boot["noninferior"] else "FAIL") if complete
                       else "PARTIAL — CF 미완, 판정 보류(발화 표본 누락으로 편향)"}
        if complete:
            # 관할 ON 반사실의 성공률 절 (CF로 완성)
            gain = sum((1 if cf[r["uid"]] else 0) - (1 if r["outcome"] == "success" else 0)
                       for r in rej_fire if r["uid"] in cf)
            out["shadow_jurisdiction_counterfactual"]["observed"].update({
                "success_delta_total": gain,
                "conditional_gain_per_ep": round(gain / len(ok), 4),
                "success_rate_on": round(
                    (sum(r["outcome"] == "success" for r in ok) + gain) / len(ok), 4)})
    else:
        out["H4b_noninferiority"] = {"status": "PENDING — counterfactual 배치 진행 중"}

    # ---- 3장부
    out["ledgers"] = {"operational_s": summary["ledger_s"]["operational_s"],
                      "formation_s": summary["ledger_s"]["formation_s"],
                      "evaluation_s": "counterfactual 배치 (측정 아티팩트 — 비용 미보고)",
                      "total_wall_s": summary["total_wall_s"]}

    op = os.path.join(rd, f"reading_{args.seed_idx}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E5AN-DONE] {op}")
    print(json.dumps({k: out[k] for k in
                      ["overview", "H4a_call_rate_reduction", "risk_control",
                       "maturity_dual_report", "H4b_noninferiority"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
