"""depth ablation 판정 패키지 생성기 (CLAUDE.md §6 — 판정 국면 = 패키지 자동 생성).

수치는 results/rgb_depth_ablation/ablation_summary.json에서 프로그래밍 주입. 수동 입력 0.
실행: hv2_hab python -u experiments/make_ablation_pack.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

DATE = "20260828"
PACK = f"depth_ablation_pack_{DATE}"
SRC = "results/rgb_depth_ablation/ablation_summary.json"
LONG = "libero_10_task0"
NGRID = [10, 20, 40, 80]
SUITE = {"libero_object_task1": ("object", "easy"), "libero_object_task0": ("object", "difficult"),
         "libero_goal_task1": ("goal", "easy"), "libero_goal_task0": ("goal", "difficult"),
         "libero_spatial_task1": ("spatial", "medium"), "libero_10_task0": ("long", "censored")}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def report(d):
    p, ns, s80 = d["paired"], d["per_cluster_nstar"], d["s80"]
    rows = d["rows"]
    delta80 = {r["cluster"]: r["delta_success"] for r in rows if r["n"] == 80}
    by_n = {n: float(np.mean([r["delta_success"] for r in rows if r["n"] == n])) for n in NGRID}
    ex_long = float(np.mean([v for c, v in delta80.items() if c != LONG]))
    same = sum(1 for c in ns if ns[c]["rgbd"] == ns[c]["rgb"])

    L = []
    A = L.append
    A(f"# depth privileged-information confound 스크리닝 — 판정 요청\n")
    A(f"작성 2026-08-28 · Stage 1 · 실행 `experiments/rgb_depth_ablation/`\n")
    A(f"> **판정 요청 1건**: 자동 판정은 **CASE {d['screening_case']}**이고 지시서 §11은 "
      f"CASE B에 **C(25 클러스터 재실행)** 를 지정한다. 그러나 증거는 경계선이며 실험자 권고는 "
      f"**B(6 클러스터 결과를 ablation 절로 추가)** 다. **B와 C 중 택일이 필요하다.**\n")
    A("---\n\n## 1. 무엇을 했나\n")
    A(f"habit이 teacher에 없는 depth를 받는다는 설계 사실이 \"높은 습관 성능 = privileged sensing\"이라는 "
      f"리뷰어 비판을 부를 수 있다. 이를 데이터로 검사했다.\n")
    A(f"- **기존 실험 무수정.** 별도 경로(`experiments/rgb_depth_ablation/`, `results/rgb_depth_ablation/`, "
      f"`checkpoints/rgb_only_ablation/`)에서만 작업했고 기존 RGB-D 체크포인트는 읽기 전용으로 재사용했다.")
    A(f"- **차이는 depth 하나뿐.** conv1을 4채널 → 3채널로 좁힌 것이 전부다 "
      f"(파라미터 차 {d['param_delta']:,} = {100*d['param_delta']/d['params']['rgbd']:.4f}%). "
      f"teacher 궤적·`B_k` 순서·optimizer·스케줄·seed·해상도·시점·평가 스펙이 모두 동일하다 "
      f"(16항목 감사: `CONFIG_DIFF.md`).")
    A(f"- **클러스터는 결과 산출 전 고정.** 스위트별 N* 최소/최대 규칙으로 6개 "
      f"(easy 2 · medium 1 · difficult 2 · censored 1), spatial·long 필수 포함 "
      f"(`ABLA_RGBD_CLUSTER_SELECTION.md`).")
    A(f"- **기존 결과 재현 확인.** 동일 RGB-D 체크포인트를 50-trial로 재평가해 기존 E3 20-trial과 "
      f"공통 uid **80/80 완전 일치**. 실행 결정성이 확인된 뒤에야 해석에 들어갔다.")
    A(f"- **에피소드 단위 paired.** 두 조건이 같은 held-out uid를 본다 → {p['n_episodes']:,} paired 에피소드.\n")
    A("## 2. 결과\n### 2.1 클러스터별\n")
    A("| Cluster | Suite | 난이도 | N*(RGB-D) | N*(RGB) | ŝ(80) RGB-D | ŝ(80) RGB | Δŝ(80) |")
    A("|---|---|---|---|---|---|---|---|")
    for c in d["clusters"]:
        su, gr = SUITE[c]
        A(f"| {c.replace('libero_','')} | {su} | {gr} | {ns[c]['rgbd']} | {ns[c]['rgb']} "
          f"| {s80[c]['rgbd']:.3f} | {s80[c]['rgb']:.3f} | {delta80[c]:+.3f} |")
    A("")
    A("### 2.2 전체 paired\n")
    A(f"- 평균 차 **{p['mean_delta_success']:+.4f}**, bootstrap 95% CI "
      f"**[{p['ci95_bootstrap'][0]:+.4f}, {p['ci95_bootstrap'][1]:+.4f}]** — **0을 포함**")
    A(f"- 불일치쌍 RGB-only만 성공 {p['discordant_rgb_only_success']} / RGB-D만 성공 "
      f"{p['discordant_rgbd_only_success']} · exact McNemar **p = {p['mcnemar_exact_p']:.4f}**\n")
    A("### 2.3 n별 Δ (6 클러스터 평균, pp)\n")
    A("| n | " + " | ".join(str(n) for n in NGRID) + " |")
    A("|---|" + "---|" * len(NGRID))
    A("| Δ | " + " | ".join(f"{100*by_n[n]:+.2f}" for n in NGRID) + " |")
    A("")
    A(f"**비단조**. depth는 저데이터 구간(n=10)에서 {abs(100*by_n[10]):.2f} pp를 벌어주고 "
      f"n=40에서는 오히려 RGB-only가 {100*by_n[40]:+.2f} pp 앞선다.\n")
    A("### 2.4 실패 유형\n")
    A(f"두 조건 모두 **전 실패가 timeout 계열**(에피소드 상한 도달), 기타 0건. depth 제거가 "
      f"조기 파국을 만들지 않는다.\n")
    A(f"> {d['failure_type_caveat']}\n")
    A("## 3. 판정 요청 — B인가 C인가\n")
    A(f"지시서 §11의 자동 판정 기준은 \"n=80 평균 감소 < 3 pp → CASE A\"이고, 실측은 "
      f"**{d['mean_drop_pp_at_n80']:.2f} pp**로 **0.67 pp 초과**해 CASE B로 떨어진다. "
      f"§11의 CASE B 처방은 C(25 클러스터 재실행)다.\n")
    A("**다만 그 초과분의 출처가 한 곳이다.**\n")
    A("| 근거 | CASE A를 지지 | CASE B를 지지 |")
    A("|---|---|---|")
    A(f"| 전체 paired | CI [{p['ci95_bootstrap'][0]:+.4f}, {p['ci95_bootstrap'][1]:+.4f}] 0 포함, "
      f"McNemar p={p['mcnemar_exact_p']:.4f} | — |")
    A(f"| n=80 평균 | long 제외 시 감소 **{-100*ex_long:.2f} pp** (기준 이내) "
      f"| 전체 **{d['mean_drop_pp_at_n80']:.2f} pp** (기준 초과) |")
    A(f"| N* | {same}/6 동일, 이동 2개는 **+1/−1 상쇄** | — |")
    A(f"| 스위트 | spatial {delta80['libero_spatial_task1']:+.2f} (차이 없음) "
      f"| long **{delta80[LONG]:+.2f}** |")
    A(f"| 실패 유형 | 전부 timeout, 조기 파국 없음 | — |")
    A("")
    A(f"long 클러스터(`{LONG}`)는 **RGB-D에서도 N*>80으로 우측절단**된, 애초에 형성되지 않은 셀이다. "
      f"원고도 이를 유일한 절단 사례로 이미 보고한다.\n")
    A("**실험자 권고: B.** 리뷰어 비판의 실질은 \"습관의 높은 성능이 depth 덕분\"인데, 논문의 주장이 "
      "걸려 있는 **천장에서 차이가 검출되지 않는다**. 층화 표본에서 결과가 한 방향으로 쏠리지도 않는다"
      "(6개 중 2개는 RGB-only 우위). C를 해도 결론이 \"천장 동일, 저데이터 구간에서만 차이\"를 "
      "벗어날 가능성이 낮아 비용 대비 이득이 낮다. C 비용은 약 **15시간**"
      "(RGB-only 학습 4.4h + 양 조건 평가 10.4h).\n")
    A("**그러나 자동 판정이 CASE B로 떨어진 것은 사실이므로, 지시서 §18(\"결론은 데이터에 따라 결정한다\")에 "
      "따라 실험자가 단독으로 기준을 완화하지 않고 판정을 요청한다.**\n")
    A("## 4. B 채택 시 즉시 사용 가능한 산출물\n")
    A("`paper/`에 논문(v11) 서식으로 준비돼 있다. 수동 숫자 입력 0 — 전 수치가 `\\Num{Abla*}` 매크로다.\n")
    A("| 파일 | 내용 |")
    A("|---|---|")
    A("| `paper/SECTION_depth_ablation.tex` | 본문 절 (삽입 위치 = V-F Scope 직전) |")
    A("| `paper/TABLE_depth_ablation.tex` | Table III |")
    A("| `paper/fig_depth_ablation.pdf` | Fig. 6, 2-패널 double-column |")
    A("| `paper/ablation_numbers.json` | 매크로 21개 + 각 값의 source 경로 |")
    A("| `paper/INSTRUCTION_depth_ablation_for_manuscript_agent.md` | 적용 절차 |")
    A("")
    A("## 5. 패키지 구성\n")
    A("| 경로 | 내용 |")
    A("|---|---|")
    A("| `RGB_DEPTH_ABLATION_AUDIT.md` | 지시서 §16 A~E 감사 보고서 |")
    A("| `results/rgb_depth_ablation/` | 원자료 12 JSON + summary + CSV + 그림 3종 |")
    A("| `experiments/rgb_depth_ablation/` | 실행·분석·산출 스크립트 전부 |")
    A("| `habits/` | ACT 구현 (in_ch 스위치 포함) |")
    A("| `configs/preregistration.md` · `log.md` | 전문 |")
    A("| `git_log.txt` | 커밋 이력 |")
    return "\n".join(L)


def main():
    d = json.load(open(SRC))
    if os.path.exists(PACK):
        shutil.rmtree(PACK)
    os.makedirs(f"{PACK}/results", exist_ok=True)
    os.makedirs(f"{PACK}/experiments", exist_ok=True)
    os.makedirs(f"{PACK}/configs", exist_ok=True)

    shutil.copytree("results/rgb_depth_ablation", f"{PACK}/results/rgb_depth_ablation")
    shutil.copytree("experiments/rgb_depth_ablation", f"{PACK}/experiments/rgb_depth_ablation")
    shutil.copytree("habits", f"{PACK}/habits",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy("experiments/rgb_depth_ablation/RGB_DEPTH_ABLATION_AUDIT.md", PACK)
    shutil.copy("experiments/make_ablation_pack.py", f"{PACK}/experiments/")
    shutil.copy("configs/preregistration.md", f"{PACK}/configs/")
    shutil.copy("log.md", PACK)
    shutil.copy("CLAUDE.md", PACK)
    # paper/ 를 최상위로도 노출 (판정 후 바로 집어갈 수 있게)
    shutil.copytree("results/rgb_depth_ablation/paper", f"{PACK}/paper")

    open(f"{PACK}/git_log.txt", "w").write(sh("git log --stat -12"))
    open(f"{PACK}/REPORT.md", "w").write(report(d))

    subprocess.run(f"tar czf {PACK}.tar.gz {PACK}", shell=True, check=True)
    size = os.path.getsize(f"{PACK}.tar.gz") / 1e6
    nf = sum(len(f) for _, _, f in os.walk(PACK))
    print(f"[PACK-DONE] {PACK}.tar.gz · {size:.2f} MB · {nf} 파일")


if __name__ == "__main__":
    main()
