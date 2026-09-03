"""실물 로봇 실험 인수인계 패키지 — 전체 프로젝트 + 이식 가이드.

대상: RA-L 논문(v11)의 실물 검증을 맡을 연구원.
포함: 전 소스 · 실험 결과 전량 · 원자료 · 그림 · 사전등록/log 전문 · 이식 가이드 · 논문.
제외: checkpoints(93G) · data HDF5(8.8G) · third_party(재설치) — 사유와 복원 경로를 문서에 명시.

산출: realrobot_handoff_<날짜>/ + .tar.gz
실행: hv2_hab python -u experiments/make_realrobot_pack.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(HABIT2)

CODE_DIRS = ("envs", "habits", "gates", "teacher", "experiments", "tools", "configs")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.joblib", "*.pt", "*.hdf5")


def main():
    stamp = datetime.now().strftime("%Y%m%d")
    pk = os.path.join(HABIT2, f"realrobot_handoff_{stamp}")
    shutil.rmtree(pk, ignore_errors=True)
    for d in ("code", "results", "raw", "figures", "prereg", "docs", "paper"):
        os.makedirs(os.path.join(pk, d), exist_ok=True)

    # ---- 코드 전량 (저장소 구조 보존)
    for d in CODE_DIRS:
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(pk, "code", d), ignore=IGNORE)

    # ---- 결과 전량 (무효·증거 디렉토리 제외, 그림은 figures/로)
    for stage in ("e0", "e1", "e2", "e3", "e4", "e5"):
        src = os.path.join("results", stage)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(pk, "results", stage)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            s = os.path.join(src, f)
            if not os.path.isfile(s):
                continue
            shutil.copy2(s, os.path.join(pk, "figures", f) if f.endswith(".png")
                         else os.path.join(dst, f))

    # ---- 원자료 (3 seed 스트림·CF)
    for i in (0, 1, 2):
        for kind in ("stream", "cf", "cf_queue"):
            src = f"results/e5/{kind}_{i}.jsonl"
            if os.path.exists(src):
                subprocess.run(["gzip", "-c", src],
                               stdout=open(os.path.join(pk, "raw", f"e5_{kind}_{i}.jsonl.gz"), "wb"))

    # ---- 문서
    shutil.copy2("docs/REAL_ROBOT_PORTING_GUIDE.md", os.path.join(pk, "01_PORTING_GUIDE.md"))
    for f in sorted(os.listdir("docs")):
        shutil.copy2(os.path.join("docs", f), os.path.join(pk, "docs", f))
    for f in ("configs/preregistration.md", "log.md", "CLAUDE.md"):
        shutil.copy2(f, os.path.join(pk, "prereg", os.path.basename(f)))
    with open(os.path.join(pk, "prereg", "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-80"],
                               capture_output=True, text=True).stdout)
    # 논문 PDF는 저장소에 없다 — 자리만 만들고 안내
    open(os.path.join(pk, "paper", "PUT_PAPER_PDF_HERE.txt"), "w").write(
        "RA-L 투고본 PDF(HabitVLA2_RAL_initial_anonymous_fixed.pdf)를 이 디렉토리에 넣어 전달할 것.\n"
        "저장소에는 원고 소스가 없어 자동 포함이 불가하다(원고 작업은 별도 환경).\n")

    # ---- 시뮬 기준선 요약 (실물 결과와 대조할 값)
    syn = json.load(open("results/e5/seed_synthesis.json"))
    lat = json.load(open("results/e1/e1_latency.json"))
    import numpy as _np
    _ex = [json.load(open(f"results/e5/reading_{i}.json"))["maturity_dual_report"]
           ["exposures_to_maturity_median"] for i in (0, 1, 2)]
    md = {"exposures_to_maturity_median_per_seed": _ex,
          "exposures_to_maturity_mean": round(float(_np.mean(_ex)), 1),
          "exposures_to_maturity_sd": round(float(_np.std(_ex, ddof=1)), 1)}
    base = {
        "note": "실물 결과와 대조할 시뮬 기준선. 전 수치는 results/ JSON에서 프로그래밍 산출.",
        "H4a_routing": syn["H4a_call_rate"],
        "H4b_noninferiority": syn["H4b_noninferiority"],
        "risk": syn["risk_control"],
        "formation": syn["formation"],
        "maturity_exposures": md,
        "latency_ms": {"teacher_chunk": lat["anchor1_oft_chunk_forward"]["median_ms"],
                       "habit_chunk": lat["anchor2_act_forward"]["median_ms"],
                       "gate_once_per_episode": lat["anchor3_gate_path"]["median_ms"],
                       "attn": lat["attn"], "gpu": lat["gpu"]},
        "constants": {"tau0": 0.8, "delta": 0.1, "epsilon": 0.2, "gamma": 0.02,
                      "clip": [0.8, 0.99], "retrain_trigger": [20, 80], "R_max": 2,
                      "P_probe": 20, "c_reinit": 0.25, "chunk_K": 8,
                      "train_steps": {"n20": 10000, "n80": 28000}},
    }
    json.dump(base, open(os.path.join(pk, "02_sim_baseline.json"), "w"), indent=2, ensure_ascii=False)

    # ---- START HERE
    a, b, r = syn["H4a_call_rate"], syn["H4b_noninferiority"], syn["risk_control"]
    L = []
    A = L.append
    A("# HabitVLA-2 실물 로봇 실험 — 인수인계 패키지\n")
    A(f"생성 {datetime.now():%Y-%m-%d}. 시뮬레이션(LIBERO) 구현·실행 일체와 이식 가이드를 포함한다.\n")
    A("## 읽는 순서\n")
    A("| 순서 | 파일 | 내용 |")
    A("|---|---|---|")
    A("| 1 | `paper/` (PDF를 넣어 전달) | RA-L 투고본 — 주장과 future work |")
    A("| 2 | **`01_PORTING_GUIDE.md`** | **실물 이식 시 무엇이 깨지는지 — 먼저 읽을 것** |")
    A("| 3 | `02_sim_baseline.json` | 실물 결과와 대조할 시뮬 기준선·상수 |")
    A("| 4 | `code/` | 전체 소스 (저장소 구조 그대로) |")
    A("| 5 | `prereg/preregistration.md` | 동결 상수와 그 근거·변경 이력 |")
    A("| 6 | `prereg/log.md` | 실패 이력 — **통독 말고 검색해 쓸 것** |")
    A("| 7 | `results/`, `raw/`, `figures/` | 시뮬 결과 전량 |\n")
    A("## 시뮬 결과 요약 (3 seed, 12,000 ep)\n")
    A(f"- VLA 호출률 **{a['first1000']['mean']}±{a['first1000']['sd']} → {a['last1000']['mean']}±{a['last1000']['sd']}** "
      f"(Δ {a['delta']['mean']}±{a['delta']['sd']})")
    A(f"- 비열등 diff **{b['diff']['mean']:+.4f}±{b['diff']['sd']:.4f}** (margin {b['margin']}, "
      f"paired {syn['counterfactual']['total_paired']:,} ep)")
    A(f"- 발화 위험 Pr(fail|fire) **{r['pr_fail_given_fire']['mean']}±{r['pr_fail_given_fire']['sd']}** (ε={r['epsilon']})")
    A(f"- 성숙 {syn['formation']['n_matured']['mean']:.1f}±{syn['formation']['n_matured']['sd']:.1f}/33, "
      f"소요 노출 중앙값 {md['exposures_to_maturity_mean']}±{md['exposures_to_maturity_sd']}회\n")
    A("## ★ 착수 전 반드시 결정할 두 가지\n")
    A("1. **성공 판정을 무엇으로 할 것인가** — 시뮬의 무료·정확한 predicate가 실물엔 없다. "
      "이 신호가 학습 데이터·인증·위험 통제 셋에 동시에 들어가므로, 판정기 오차를 먼저 측정해야 한다. "
      "(가이드 §1)")
    A("2. **H4b 검정을 어떻게 다시 설계할 것인가** — 실물은 초기상태를 재현할 수 없어 "
      "paired replay가 불가능하다. 논문의 \"paired full-VLA replay under identical episode "
      "specifications\" 문구는 실물에서 그대로 쓸 수 없다. (가이드 §2)\n")
    A("## 제외된 것과 복원 방법\n")
    A("| 항목 | 크기 | 복원 |")
    A("|---|---|---|")
    A("| `checkpoints/` | 93 GB | 재학습으로 재생성 (spec 결정적) |")
    A("| `data/` HDF5 | 8.8 GB | teacher 재수집으로 재생성 |")
    A("| `third_party/` (LIBERO, openvla-oft) | — | 공개 저장소에서 설치 |")
    A("| OpenVLA-OFT 가중치 | ~16 GB | `moojink/openvla-7b-oft-finetuned-libero-*` |\n")
    A("## 코드를 돌려보려면\n")
    A("```bash")
    A("cp -r code/* <새-저장소>/     # code/ 내용을 저장소 루트로 옮기면 경로가 맞는다")
    A("#   (스크립트가 파일 기준 상위를 프로젝트 루트로 잡으므로 code/ 하위에서는 동작하지 않는다)")
    A("git clone <LIBERO>  third_party/LIBERO")
    A("git clone <openvla-oft> third_party/openvla-oft")
    A("```\n")
    A("## 환경\n")
    A("conda env 2개: `hv2_oft`(OpenVLA-OFT 추론) · `hv2_hab`(ACT 학습·분석). "
      "`export HF_HOME=<repo>/.hf_cache`로 캐시를 격리한다(공용 캐시 오염 방지). "
      "flash-attn은 sm_120 미빌드이므로 **attn=sdpa**를 쓰며, 모든 지연 수치에 이를 명기해야 한다.\n")
    open(os.path.join(pk, "00_START_HERE.md"), "w").write("\n".join(L))

    subprocess.run(["tar", "czf", f"{pk}.tar.gz", "-C", HABIT2, os.path.basename(pk)])
    n = sum(len(fs) for _, _, fs in os.walk(pk))
    print(f"[RRPACK] {pk}.tar.gz  ({n} files)")


if __name__ == "__main__":
    main()
