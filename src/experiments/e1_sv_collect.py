"""E1-a 집계: 스위트별·태스크별 S_V + Wilson 95% CI + go 판정.

go 기준 (configs/preregistration.md §1·§3):
  PASS   = S_V ≥ 0.85
  RELAX  = 0.75 ≤ S_V < 0.85 → 해당 스위트 임계 완화를 §5 변경 이력에 기록 후 진행 (연구원 승인)
  REDESIGN = S_V < 0.75 → 해당 셀 제외 재설계

부가 산출 (E2+ 설계 입력):
  - 태스크별 성공률: C-L0/C-L1 대표 클러스터 선정과 스트림 혼합비의 기초 자료
  - 성공/실패 에피소드 wall-clock 분리: E0-7 예산 캐비앳 1(성공-only 편향) 해소
"""
import json
import os
import re
from datetime import datetime
from math import sqrt

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
LOGDIR = os.path.join(HABIT2, "logs", "e1_sv")
OUT = os.path.join(HABIT2, "results", "e1", "e1_sv.json")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TRIALS = 25
TS = re.compile(r"^(\d{2})/(\d{2}) \[(\d{2}):(\d{2}):(\d{2})\]")
YEAR = 2026


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def parse_suite(path):
    """태스크별 (성공 수, 에피소드 수)와 성공/실패 분리 wall-clock을 추출."""
    tasks = []  # [(task_desc, [(success, dur_s), ...])]
    cur_task, cur_eps = None, []
    t_last = t_start = None
    for line in open(path, errors="replace"):
        m = TS.match(line)
        if m:
            mo, dy, hh, mm, ss = map(int, m.groups())
            t_last = datetime(YEAR, mo, dy, hh, mm, ss)
        if "Task:" in line and "task_description" not in line:
            desc = line.split("Task:", 1)[1].strip().split("  ")[0].strip()
            if desc and desc != cur_task:
                if cur_task is not None:
                    tasks.append((cur_task, cur_eps))
                cur_task, cur_eps = desc, []
        elif "Starting episode" in line and t_last is not None:
            t_start = t_last
        elif "Success:" in line and t_start is not None:
            succ = "Success: True" in line
            dur = (t_last - t_start).total_seconds() if t_last else None
            cur_eps.append((succ, dur))
            t_start = None
    if cur_task is not None:
        tasks.append((cur_task, cur_eps))
    return tasks


def main():
    report = {"trials_per_task": TRIALS, "suites": {}, "status": "FAIL"}
    verdicts = []
    for suite in SUITES:
        path = os.path.join(LOGDIR, f"console_{suite}.log")
        if not os.path.exists(path):
            report["suites"][suite] = {"status": "MISSING"}
            verdicts.append("MISSING")
            continue
        text = open(path, errors="replace").read()
        m_ep = re.findall(r"Total episodes: (\d+)", text)
        m_su = re.findall(r"Total successes: (\d+)", text)
        tasks = parse_suite(path)
        if not (m_ep and m_su):
            report["suites"][suite] = {"status": "INCOMPLETE", "reason": "no final totals — crashed"}
            verdicts.append("INCOMPLETE")
            continue
        n, k = int(m_ep[-1]), int(m_su[-1])
        expected = TRIALS * 10
        if n != expected:
            report["suites"][suite] = {
                "status": "INCOMPLETE",
                "reason": f"episodes {n} != expected {expected}",
            }
            verdicts.append("INCOMPLETE")
            continue
        sv = k / n
        lo, hi = wilson(k, n)
        if sv >= 0.85:
            st = "PASS"
        elif sv >= 0.75:
            st = "RELAX"  # 사전등록 완화 조항 — 변경 이력 기록 후 진행 (연구원 승인)
        else:
            st = "REDESIGN"
        succ_d = [d for _, eps in tasks for s, d in eps if s and d is not None]
        fail_d = [d for _, eps in tasks for s, d in eps if not s and d is not None]
        report["suites"][suite] = {
            "episodes": n,
            "successes": k,
            "S_V": round(sv, 4),
            "wilson_95": [round(lo, 4), round(hi, 4)],
            "status": st,
            "per_task": [
                {
                    "task": t,
                    "n": len(eps),
                    "k": sum(1 for s, _ in eps if s),
                    "rate": round(sum(1 for s, _ in eps if s) / len(eps), 3) if eps else None,
                }
                for t, eps in tasks
            ],
            "walltime": {
                "success_mean_s": round(sum(succ_d) / len(succ_d), 1) if succ_d else None,
                "success_n": len(succ_d),
                "fail_mean_s": round(sum(fail_d) / len(fail_d), 1) if fail_d else None,
                "fail_n": len(fail_d),
            },
        }
        verdicts.append(st)

    if any(v in ("MISSING", "INCOMPLETE") for v in verdicts):
        report["status"] = "FAIL"
    elif all(v == "PASS" for v in verdicts):
        report["status"] = "PASS"
    elif any(v == "REDESIGN" for v in verdicts):
        report["status"] = "REDESIGN"
    else:
        report["status"] = "RELAX"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                s: {kk: v[kk] for kk in ("S_V", "wilson_95", "status") if kk in v}
                for s, v in report["suites"].items()
            },
            indent=2,
        )
    )
    print(f"[E1-PASS] item=E1-SV status={report['status']} json=results/e1/e1_sv.json")


if __name__ == "__main__":
    main()
