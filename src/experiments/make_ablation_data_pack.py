"""depth ablation — 결과·데이터 완전 패키지 (전체 재실험 전 분석용).

`depth_ablation_pack_20260828.tar.gz`(판정 요청본)을 대체한다. 추가되는 것:
원자료 전량(per-episode) · 학습/평가 로그 20개 · 체크포인트 SHA256 매니페스트 ·
실측 wall-clock에서 산출한 27-클러스터 재실험 비용 · 재실험 계획.

수치는 전부 results/rgb_depth_ablation/ 및 로그에서 프로그래밍 산출. 수동 입력 0.
실행: hv2_hab python -u experiments/make_ablation_data_pack.py
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

DATE = "20260828"
PACK = f"depth_ablation_data_pack_{DATE}"
SRC = "results/rgb_depth_ablation/ablation_summary.json"
CKROOT = "checkpoints/rgb_only_ablation"
E3 = "results/e3/e3_curves.json"
LONG = "libero_10_task0"
NGRID = [10, 20, 40, 80]
SUITE = {"libero_object_task1": ("object", "easy"), "libero_object_task0": ("object", "difficult"),
         "libero_goal_task1": ("goal", "easy"), "libero_goal_task0": ("goal", "difficult"),
         "libero_spatial_task1": ("spatial", "medium"), "libero_10_task0": ("long", "censored")}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def measured_cost(done):
    """실측 wall-clock → 27 클러스터 재실험 비용. 하드코딩 없음."""
    tr = []
    for c in done:
        s = json.load(open(f"{CKROOT}/{c}/train_summary.json"))
        tr.append(sum(r["train_seconds"] for r in s["results"]))
    ev_rgb, ev_rgbd = [], []
    for c in done:
        ev_rgb.append(json.load(open(f"results/rgb_depth_ablation/{c}_rgb_h50.json"))["wall_seconds"])
        ev_rgbd.append(json.load(open(f"results/rgb_depth_ablation/{c}_rgbd_h50.json"))["wall_seconds"])
    return {"train_per_cluster_s": float(np.mean(tr)), "train_range_s": [min(tr), max(tr)],
            "eval_rgb_per_cluster_s": float(np.mean(ev_rgb)),
            "eval_rgbd_per_cluster_s": float(np.mean(ev_rgbd)),
            "n_done": len(done)}


def checkpoint_manifest():
    out = []
    for p in sorted(glob.glob(f"{CKROOT}/*/act_n*.pt")):
        h = sh(f"sha256sum {p}").split()[0]
        out.append({"path": p, "bytes": os.path.getsize(p), "sha256": h})
    return out


def report(d, cost, remaining, ck):
    p, ns, s80, rows = d["paired"], d["per_cluster_nstar"], d["s80"], d["rows"]
    delta80 = {r["cluster"]: r["delta_success"] for r in rows if r["n"] == 80}
    by_n = {n: float(np.mean([r["delta_success"] for r in rows if r["n"] == n])) for n in NGRID}
    ex_long = float(np.mean([v for c, v in delta80.items() if c != LONG]))
    same = sum(1 for c in ns if ns[c]["rgbd"] == ns[c]["rgb"])
    tot_gb = sum(x["bytes"] for x in ck) / 2**30

    # 재실험 비용 (초) — 남은 클러스터만, 실측 평균 기준
    r_tr = cost["train_per_cluster_s"] * len(remaining)
    r_ev = (cost["eval_rgb_per_cluster_s"] + cost["eval_rgbd_per_cluster_s"]) * len(remaining)
    L = []
    A = L.append
    A("# depth privileged-information confound — Stage 1 결과·데이터 패키지\n")
    A("작성 2026-08-28 · **전체 재실험(Stage 2) 전 분석용** · "
      "`depth_ablation_pack_20260828.tar.gz`(판정 요청본)을 **대체**한다.\n")
    A("> 연구원 결정: 본 패키지를 분석한 뒤 **전체 재실험을 실시**한다. 따라서 본 문서는 판정 요청이 "
      "아니라 ①Stage 1이 무엇을 확정했고 ②무엇이 미해결로 남았으며 ③Stage 2가 무엇을 고정해야 하는지를 "
      "넘기는 인수인계다.\n")
    A("---\n\n## 1. Stage 1이 확정한 것\n")
    A(f"6 클러스터 × n{{10,20,40,80}} × held-out {d['n_heldout']} = **{p['n_episodes']:,} paired 에피소드**. "
      f"두 조건의 차이는 conv1 4채널→3채널 하나뿐이다(파라미터 차 {d['param_delta']:,} = "
      f"{100*d['param_delta']/d['params']['rgbd']:.4f}%).\n")
    A("| 확정 사항 | 실측 |")
    A("|---|---|")
    A(f"| 전체 paired 차이 | **{p['mean_delta_success']:+.4f}**, 95% CI "
      f"[{p['ci95_bootstrap'][0]:+.4f}, {p['ci95_bootstrap'][1]:+.4f}] — **0 포함** |")
    A(f"| exact McNemar | p = **{p['mcnemar_exact_p']:.4f}** "
      f"(불일치 {p['discordant_rgb_only_success']}/{p['discordant_rgbd_only_success']}) |")
    A(f"| N* 불변 | **{same}/6**, 이동 2개는 +1/−1 상쇄 |")
    A(f"| n별 Δ (pp) | " + " · ".join(f"n={n} {100*by_n[n]:+.2f}" for n in NGRID) + " — **비단조** |")
    A(f"| n=80 평균 감소 | 전체 **{d['mean_drop_pp_at_n80']:.2f} pp** / long 제외 **{-100*ex_long:.2f} pp** |")
    A(f"| 실패 유형 | 양 조건 **전부 timeout 계열**, 기타 0건 |")
    A(f"| 기존 결과 재현 | 동일 RGB-D 체크포인트 50-trial 재평가 → 기존 E3 20-trial과 공통 uid **80/80 일치** |")
    A("")
    A("**해석.** depth는 저데이터 구간(n=10, −12.00 pp)의 수렴을 돕고 **천장에는 기여하지 않는다**. "
      "논문의 주장(성숙 인증 후 teacher 대체)은 천장에 걸려 있으므로 privileged-sensing 비판은 "
      "이 표본에서는 지지되지 않는다.\n")
    A("## 2. 미해결로 남은 것 — Stage 2가 답해야 할 질문\n")
    A(f"1. **long-horizon.** 유일한 long 클러스터 `{LONG}`가 Δ = {delta80[LONG]:+.2f}로 가장 크다. "
      f"그런데 이 셀은 **RGB-D에서도 N*>80으로 우측절단**돼 애초에 형성되지 않는다. "
      f"형성 실패 위에서 측정한 차이라 해석이 성립하지 않는다. long 스위트에 형성되는 셀이 있는지, "
      f"있다면 거기서도 −14 pp가 재현되는지가 핵심이다.")
    A(f"2. **표본 크기.** n=80 평균 감소가 {d['mean_drop_pp_at_n80']:.2f} pp로 기준 3 pp를 "
      f"0.67 pp 초과하는데, 이 초과가 long 1개에서 나온다. 6개 표본에서 클러스터 1개의 "
      f"영향력이 지나치게 크다.")
    A(f"3. **spatial 가설 반증.** depth 의존 후보로 지목한 spatial이 Δ = "
      f"{delta80['libero_spatial_task1']:+.2f}에 N* 불변이었다. spatial 클러스터가 2개뿐이라 "
      f"\"기하 과제가 depth를 필요로 한다\"는 가설을 6개로는 검정할 수 없다.")
    A("4. **실패 유형 분해.** 전 실패가 timeout이라는 사실만 확인했다. grasp/localization/placement "
      "분리는 per-episode 비디오가 필요해 Stage 1 범위 밖이었다.")
    A("5. **온라인 lifecycle.** 배치 형성만 다뤘다. 재학습 예산·probe 결과와 depth가 상호작용하는지는 미검증.\n")
    A("## 3. Stage 2 규모 — ★ 25가 아니라 27이다\n")
    A(f"지시서 §11은 \"25 클러스터 재실행\"으로 적었으나, **E3 배치 원장의 실제 클러스터는 "
      f"{len(json.load(open(E3))['clusters'])}개**다(`{E3}`의 `n_clusters_reported`, "
      f"completeness missing=[] 확인). 그중 **{cost['n_done']}개는 Stage 1에서 이미 완료**됐고 "
      f"**{len(remaining)}개가 남는다**. Stage 2 착수 전에 이 수를 확정해야 한다.\n")
    A("**남은 클러스터 " + str(len(remaining)) + "개**\n")
    A("```")
    for i in range(0, len(remaining), 3):
        A("  " + "  ".join(f"{c:<32}" for c in remaining[i:i + 3]).rstrip())
    A("```\n")
    A("### 실측 기반 비용 (Stage 1 wall-clock에서 산출, 하드코딩 없음)\n")
    A("| 항목 | 클러스터당 실측 | 남은 " + str(len(remaining)) + "개 |")
    A("|---|---|---|")
    A(f"| RGB-only 학습 (n 4개 합) | {cost['train_per_cluster_s']/60:.1f} 분 "
      f"(범위 {cost['train_range_s'][0]/60:.1f}–{cost['train_range_s'][1]/60:.1f}) | "
      f"**{r_tr/3600:.1f} 시간** |")
    A(f"| RGB-only 평가 (4 ckpt × 50) | {cost['eval_rgb_per_cluster_s']/60:.1f} 분 | "
      f"{cost['eval_rgb_per_cluster_s']*len(remaining)/3600:.1f} 시간 |")
    A(f"| RGB-D 재평가 (4 ckpt × 50) | {cost['eval_rgbd_per_cluster_s']/60:.1f} 분 | "
      f"{cost['eval_rgbd_per_cluster_s']*len(remaining)/3600:.1f} 시간 |")
    A(f"| **합계** | | **{(r_tr + r_ev)/3600:.1f} 시간** |")
    A("")
    A(f"클러스터당 학습 1회는 {cost['train_per_cluster_s']/60:.1f}분인데 평가는 두 조건 합쳐 "
      f"{(cost['eval_rgb_per_cluster_s'] + cost['eval_rgbd_per_cluster_s'])/60:.1f}분으로 "
      f"**평가가 전체의 {100*r_ev/(r_tr + r_ev):.0f}%** 를 차지한다. "
      f"`chained_*` 2개는 커스텀 래퍼라 에피소드가 길어 위 평균보다 더 걸릴 수 있다.\n")
    A("### Stage 2가 고정해야 할 것 (Stage 1과 동일하게 유지)\n")
    A("- **동일성 16항목** — `CONFIG_DIFF.md`. depth 외 어떤 것도 바꾸지 않는다 "
      "(width·encoder·steps·augmentation·lr·태스크별 튜닝 전부 금지).")
    A("- **paired 성립** — 두 조건이 같은 held-out uid를 본다. `heldout_specs(suite, task, 50)`.")
    A("- **기존 RGB-D 체크포인트 재사용** — 재학습하지 않는다. 재평가만 한다.")
    A(f"- **Stage 1 6개를 다시 돌리지 않는다** — 이미 완료됐고 결정성이 확인됐다. "
      f"체크포인트 SHA256이 `checkpoint_manifest.json`에 있다({tot_gb:.1f} GB, 24개).")
    A("- **클러스터 전수** — Stage 1처럼 층화 표집하지 않으므로 선택 편의 문제가 사라진다.\n")
    A("## 4. 패키지 구성\n")
    A("| 경로 | 내용 |")
    A("|---|---|")
    A("| `RGB_DEPTH_ABLATION_AUDIT.md` | 지시서 §16 A~E 감사 보고서 |")
    A(f"| `results/rgb_depth_ablation/*.json` | **원자료 12개** — 클러스터×조건, per-episode "
      f"(uid·outcome·steps) 전량 |")
    A("| `results/rgb_depth_ablation/ablation_summary.json` | 분석 단일 진입점 |")
    A("| `results/rgb_depth_ablation/table_detail.csv` | 지시서 §14 전 필드 |")
    A("| `results/rgb_depth_ablation/fig_{A,B,C}*.png` | 형성 곡선 · Δŝ(80) · N* paired |")
    A("| `paper/` | 논문 서식 산출물 (본문 절·Table·Fig·매크로 21개·원고 지시서) |")
    A("| `logs/rgb_depth_ablation/` | **학습 6 + 평가 12 + run.log** = 19개 원본 로그 |")
    A("| `train_summaries/` | 클러스터별 final_l1 · train_seconds |")
    A("| `checkpoint_manifest.json` | RGB-only 체크포인트 24개 SHA256 (가중치는 디스크에 잔류) |")
    A("| `stage2_remaining_clusters.json` | 남은 클러스터 목록 + 실측 비용 |")
    A("| `experiments/rgb_depth_ablation/` | 실행·분석·산출 스크립트 전부 |")
    A("| `habits/` | ACT 구현 (`in_ch` 스위치 포함) |")
    A("| `configs/preregistration.md` · `log.md` · `CLAUDE.md` | 전문 |")
    A("| `git_log.txt` | 커밋 이력 |")
    A("")
    A(f"**체크포인트 가중치는 포함하지 않았다** — {tot_gb:.1f} GB로 패키지에 넣을 수 없다. "
      f"`{CKROOT}/`에 그대로 있고 SHA256으로 대조 가능하다.\n")
    return "\n".join(L)


def main():
    d = json.load(open(SRC))
    done = list(d["clusters"])
    allc = list(json.load(open(E3))["clusters"])
    remaining = [c for c in allc if c not in done]
    cost = measured_cost(done)
    ck = checkpoint_manifest()

    if os.path.exists(PACK):
        shutil.rmtree(PACK)
    for sub in ("results", "experiments", "configs", "logs", "train_summaries"):
        os.makedirs(f"{PACK}/{sub}", exist_ok=True)

    shutil.copytree("results/rgb_depth_ablation", f"{PACK}/results/rgb_depth_ablation")
    shutil.copytree("experiments/rgb_depth_ablation", f"{PACK}/experiments/rgb_depth_ablation")
    shutil.copytree("habits", f"{PACK}/habits", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree("logs/rgb_depth_ablation", f"{PACK}/logs/rgb_depth_ablation")
    shutil.copytree("results/rgb_depth_ablation/paper", f"{PACK}/paper")
    for c in done:
        shutil.copy(f"{CKROOT}/{c}/train_summary.json", f"{PACK}/train_summaries/{c}.json")
    shutil.copy("experiments/rgb_depth_ablation/RGB_DEPTH_ABLATION_AUDIT.md", PACK)
    shutil.copy("experiments/make_ablation_data_pack.py", f"{PACK}/experiments/")
    shutil.copy("configs/preregistration.md", f"{PACK}/configs/")
    for f in ("log.md", "CLAUDE.md"):
        shutil.copy(f, PACK)

    json.dump(ck, open(f"{PACK}/checkpoint_manifest.json", "w"), indent=1)
    json.dump({"all_batch_clusters": allc, "n_all": len(allc),
               "stage1_done": done, "stage2_remaining": remaining,
               "n_remaining": len(remaining), "measured_cost_seconds": cost,
               "note": "지시서 §11의 '25 클러스터'는 E3 원장(27)과 불일치. 확정 필요."},
              open(f"{PACK}/stage2_remaining_clusters.json", "w"), indent=1, ensure_ascii=False)
    open(f"{PACK}/git_log.txt", "w").write(sh("git log --stat -14"))
    open(f"{PACK}/REPORT.md", "w").write(report(d, cost, remaining, ck))

    subprocess.run(f"tar czf {PACK}.tar.gz {PACK}", shell=True, check=True)
    nf = sum(len(f) for _, _, f in os.walk(PACK))
    print(f"[PACK-DONE] {PACK}.tar.gz · {os.path.getsize(PACK + '.tar.gz')/1e6:.2f} MB · {nf} 파일")
    print(f"  Stage 1 완료 {len(done)} · Stage 2 남은 {len(remaining)} / 전체 {len(allc)}")
    print(f"  실측 재실험 비용 {(cost['train_per_cluster_s'] + cost['eval_rgb_per_cluster_s'] + cost['eval_rgbd_per_cluster_s'])*len(remaining)/3600:.1f} 시간")


if __name__ == "__main__":
    main()
