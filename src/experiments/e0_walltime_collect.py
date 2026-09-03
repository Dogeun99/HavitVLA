"""E0-7: 스모크 콘솔 로그의 타임스탬프에서 에피소드당 wall-clock을 추출해 예산을 정밀화한다.

입력: logs/e0_5/console_<suite>.log (run_libero_eval의 asctime 로그)
      results/e0/e0_4_ckpt.json (로드 시간·forward 시간)
출력: results/e0/e0_7_walltime.json — 스위트별 에피소드당 초 + E1/E2/E3/E5 예산 환산.

예산 모형 (출처 명시 — 검증 워크플로우 발견 반영):
  E1 = 40 태스크 × 25 ep = 1,000 ep (teacher)          [설계서 §5 E1 행]
  E2 = 2 클러스터 (C-L0 + C-L1 대표 1 — 설계서 §5 E2 행) × (120 수집 + 4 ckpt × 50 평가)
  E3 = 28 클러스터 (preregistration.md §4 파생상수) × (120 수집 + 4 × 20 평가)
  E5 = 2,000 ep × 3 seed                                [설계서 §4.4]

★ 추정치 편향 주의 (JSON의 budget_caveats에도 기록):
  1. 스모크가 전 성공(40/40)이라 실패 에피소드(max_steps 완주, 성공 대비 2~4배)가 미반영 — 하향 편향.
  2. 측정 구간('Starting episode'→'Success:')이 save_rollout_video()의 mp4 인코딩을 포함 —
     실전 rollout(비디오 저장 없음)에는 없는 비용 — 상향 편향.
  3. rich 로그의 초 단위·간헐 타임스탬프 상속 — 에피소드당 ±1~2s 오차.
  4. ACT 학습/추론 비용 미포함 — E1 레이턴시 앵커 ⑤ 측정 후 갱신.
따라서 "_h" 값들은 계획 수립용 개략치이며 상한 보장이 아니다. E1 완료 시 성공/실패 분리 실측으로 대체.
"""
import json
import os
import re
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
LOGDIR = os.path.join(HABIT2, "logs", "e0_5")
OUT = os.path.join(HABIT2, "results", "e0", "e0_7_walltime.json")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# rich 로그 형식: 초가 바뀔 때만 "08/15 [00:29:17]" 접두사, 같은 초의 후속 라인은 공백.
TS = re.compile(r"^(\d{2})/(\d{2}) \[(\d{2}):(\d{2}):(\d{2})\]")
YEAR = 2026


def parse_suite(path):
    """'Starting episode' → 다음 'Success:' 라인까지의 간격(초). 접두사 없는 라인은 직전 초를 상속."""
    episodes = []
    t_last, t_start = None, None
    for line in open(path, errors="replace"):
        m = TS.match(line)
        if m:
            mo, dy, hh, mm, ss = map(int, m.groups())
            t_last = datetime(YEAR, mo, dy, hh, mm, ss)
        if t_last is None:
            continue
        if "Starting episode" in line:
            t_start = t_last
        elif "Success:" in line and t_start is not None:
            episodes.append((t_last - t_start).total_seconds())
            t_start = None
    return episodes


def main():
    report = {"per_suite": {}, "budget": {}}
    all_eps = []
    for suite in SUITES:
        path = os.path.join(LOGDIR, f"console_{suite}.log")
        if not os.path.exists(path):
            report["per_suite"][suite] = {"status": "MISSING"}
            continue
        eps = parse_suite(path)
        all_eps += eps
        report["per_suite"][suite] = {
            "n": len(eps),
            "mean_s": round(sum(eps) / len(eps), 1) if eps else None,
            "min_s": round(min(eps), 1) if eps else None,
            "max_s": round(max(eps), 1) if eps else None,
        }

    ckpt_path = os.path.join(HABIT2, "results", "e0", "e0_4_ckpt.json")
    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        report["model_load_seconds"] = {k: v["load_seconds"] for k, v in ck.items()}

    if all_eps:
        mean_ep = sum(all_eps) / len(all_eps)
        h = lambda n_ep: round(n_ep * mean_ep / 3600, 1)
        report["mean_teacher_episode_s"] = round(mean_ep, 1)
        report["budget_caveats"] = [
            "success-only sample (40/40) — 실패 에피소드(max_steps 완주) 미반영: 하향 편향",
            "구간에 save_rollout_video mp4 인코딩 포함 — 실전 rollout엔 없음: 상향 편향",
            "초 단위 간헐 타임스탬프 — 에피소드당 ±1~2s",
            "ACT 학습/추론 비용 미포함 — E1 앵커 ⑤ 후 갱신",
        ]
        report["budget"] = {
            "assumption": "teacher-ep 개략치 (budget_caveats 참조; 상한 보장 아님)",
            "E1_1000ep_h": h(1000),
            "E2_collect_240ep_h": h(2 * 120),
            "E2_eval_400ep_h": h(2 * 4 * 50),
            "E3_collect_3360ep_h": h(28 * 120),
            "E3_eval_2240ep_h": h(28 * 4 * 20),
            "E5_6000ep_h": h(6000),
            "total_h_estimate": h(1000 + 240 + 400 + 3360 + 2240 + 6000),
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = all_eps and all(v.get("n", 0) > 0 for v in report["per_suite"].values())
    print(f"[E0-PASS] item=E0-7 status={'PASS' if ok else 'FAIL'} json=results/e0/e0_7_walltime.json")


if __name__ == "__main__":
    main()
