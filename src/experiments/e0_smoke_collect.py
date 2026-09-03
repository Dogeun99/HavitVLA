"""E0-5 집계: logs/e0_5/console_*.log에서 최종 성공률을 파싱해 JSON + PASS 마커 출력.

go 기준(사전등록, 설계서 §5): 스위트별 성공률이 공개 보고치 ±10 %p 이내.
공개 보고치(스위트별 독립 체크포인트): spatial 97.6 / object 98.4 / goal 97.9 / long(10) 94.5.
  출처: OpenVLA-OFT 논문(Kim et al., 2025) LIBERO 결과 표 — 평균 97.1%는 로컬
  third_party/openvla-oft/LIBERO.md:41에서 교차 확인, 스위트별 수치의 원본은 논문 표이며
  이 dict가 프로젝트 내 유일 기록이므로 configs/preregistration.md §4에도 등재(검증 워크플로우 발견 반영).

판정 3단 (ISSUE-7 규칙 — configs/preregistration.md §3 등재, 연구원 승인 대기):
  PASS    = 완결(n=10 또는 20) AND |rate − pub| ≤ 0.10
  RECHECK = 완결 n=10 AND 밴드 밖 AND 실패 ≤ 2 → 해당 스위트만 20 ep(추가 init state) 재확인
  FAIL    = 그 외 (중도 크래시·표본 수 불일치 포함 — 부분 로그는 절대 PASS 불가)
"""
import json
import os
import re

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
LOGDIR = os.path.join(HABIT2, "logs", "e0_5")
OUT = os.path.join(HABIT2, "results", "e0", "e0_5_smoke.json")

PUBLISHED = {"libero_spatial": 0.976, "libero_object": 0.984, "libero_goal": 0.979, "libero_10": 0.945}

# 완결성 요구 표본 크기: 1차 = 10 (10 태스크 × 1), 재확인 = 20 (10 태스크 × 2, ISSUE-7)
EXPECTED_N = (10, 20)

report = {"published": PUBLISHED, "expected_n": list(EXPECTED_N), "suites": {}, "status": "FAIL"}
statuses = []
for suite, pub in PUBLISHED.items():
    path = os.path.join(LOGDIR, f"console_{suite}.log")
    if not os.path.exists(path):
        report["suites"][suite] = {"status": "MISSING"}
        statuses.append("FAIL")
        continue
    text = open(path, errors="replace").read()
    m_ep = re.findall(r"Total episodes: (\d+)", text)
    m_su = re.findall(r"Total successes: (\d+)", text)
    succ_lines = re.findall(r"Success: (True|False)", text)
    incomplete = None
    if m_ep and m_su:
        n, k = int(m_ep[-1]), int(m_su[-1])
    else:
        # 최종 집계 라인 부재 = 중도 크래시. 부분 표본으로 판정하지 않는다 (검증 워크플로우 발견).
        n, k = len(succ_lines), sum(s == "True" for s in succ_lines)
        incomplete = "no final totals line — run crashed mid-suite"
    diff = (k / n if n else 0.0) - pub
    failures = n - k
    if incomplete or n not in EXPECTED_N:
        st = "FAIL"  # 완결성 미달 — 판정 불가를 PASS로 오독하지 않음
        incomplete = incomplete or f"episode count {n} not in expected {EXPECTED_N}"
    elif abs(diff) <= 0.10:
        st = "PASS"
    elif n == 10 and failures <= 2:
        # ISSUE-7 문서 규칙 그대로: ±10%p 밖 + 실패 ≤ 2 → 해당 스위트만 20 ep 재확인.
        # 주의: 파이프라인이 결정적(greedy·seed 고정)이므로 재확인은 반드시 추가 init state
        # (num_trials_per_task=2 → init_states[0..1])로만 유효하다. 동일 10 ep 재실행은 무정보.
        st = "RECHECK"
    else:
        st = "FAIL"
    report["suites"][suite] = {
        "episodes": n,
        "successes": k,
        "rate": round(k / n, 3) if n else 0.0,
        "published": pub,
        "diff_pp": round(diff * 100, 1),
        "status": st,
        "incomplete": incomplete,
        "per_episode": succ_lines,
    }
    statuses.append(st)

if all(s == "PASS" for s in statuses):
    report["status"] = "PASS"
elif any(s == "FAIL" for s in statuses):
    report["status"] = "FAIL"
else:
    report["status"] = "RECHECK"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in report["suites"].items()}, indent=2, ensure_ascii=False))
print(f"[E0-PASS] item=E0-5 status={report['status']} json=results/e0/e0_5_smoke.json")
