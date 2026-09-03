"""무인 실행 감시자 — 세션과 무관하게 살아서 이상을 잡아내고, seed가 끝나면 즉시 분석한다.

감시 항목 (5분 주기):
  1. 오케스트레이터 생존       — 죽었는데 미완 stage가 남아 있으면 ALERT
  2. 스트림 진행 정체          — 드라이버는 살아 있는데 45분간 에피소드가 늘지 않으면 ALERT
                                (재학습 n=80 ≈ 11분 + probe 20회 ≈ 5분을 고려한 여유값)
  3. FAILED_JOBS.json 증가     — 새 실패 job ALERT
  4. 새 DONE_SEED/DONE_REPLAY  — 즉시 해당 분석을 돌려 중간 산출물을 만들어 둔다
                                (오케스트레이터의 최종 분석은 marker 기반이라 중복돼도 무해)
기록: results/<RUN>/ALERTS.log  (한 줄 = 한 사건, [ALERT]/[EVENT] 접두)
실행: setsid nohup hv2_hab python -u experiments/rgb_only_rerun/watchdog.py &
"""
import json
import os
import subprocess
import sys
import time

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import FAILED, MARKERS, PY_HAB, ROOT  # noqa: E402

PERIOD = 300
STALL_LIMIT = 45 * 60
ALERTS = f"{ROOT}/ALERTS.log"
RR = "experiments/rgb_only_rerun"


def log(kind, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{kind}] {msg}"
    with open(ALERTS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def alive(pat):
    return bool(subprocess.run(["pgrep", "-f", pat], capture_output=True,
                               text=True).stdout.strip())


def lines(p):
    if not os.path.exists(p):
        return 0
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def run(script, tag):
    log("EVENT", f"{tag} 분석 실행 시작")
    r = subprocess.run([PY_HAB, "-u", f"{RR}/{script}"], cwd=HABIT2,
                       capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()
    log("EVENT" if r.returncode == 0 else "ALERT",
        f"{tag} 분석 종료 exit={r.returncode} :: {tail[-1] if tail else ''}")


def main():
    log("EVENT", "watchdog 기동")
    seen_markers = set(os.listdir(MARKERS)) if os.path.isdir(MARKERS) else set()
    seen_failures = len(json.load(open(FAILED))) if os.path.exists(FAILED) else 0
    last_counts, last_change = {}, {}

    while True:
        try:
            # --- 1. 오케스트레이터 생존
            orch = alive(f"{RR}/run_all.py")
            done_all = os.path.exists(f"{MARKERS}/DONE_PACKAGE")
            if not orch and not done_all:
                log("ALERT", "오케스트레이터 프로세스 없음 — 미완 stage가 남아 있다")
                time.sleep(PERIOD)
                continue

            # --- 2. 스트림 정체
            drv = alive("experiments/e5_driver.py")
            for s in (0, 1, 2):
                p = f"{ROOT}/0{s + 2}_online_seed{s}/stream_{s}.jsonl"
                n = lines(p)
                if n == 0:
                    continue
                if last_counts.get(s) != n:
                    last_counts[s] = n
                    last_change[s] = time.time()
                elif drv and n < 4000 and time.time() - last_change.get(s, time.time()) > STALL_LIMIT:
                    log("ALERT", f"seed{s} 정체 — {n}/4000에서 "
                                 f"{(time.time() - last_change[s]) / 60:.0f}분간 변화 없음")
                    last_change[s] = time.time()      # 반복 알림 억제

            # --- 3. 새 실패 job
            if os.path.exists(FAILED):
                cur = json.load(open(FAILED))
                if len(cur) > seen_failures:
                    for f_ in cur[seen_failures:]:
                        log("ALERT", f"job 실패: {f_['job']} exit={f_['exit_code']} "
                                     f"log={f_['log']}")
                    seen_failures = len(cur)

            # --- 4. 새 완료 marker → 즉시 분석
            now = set(os.listdir(MARKERS)) if os.path.isdir(MARKERS) else set()
            for m in sorted(now - seen_markers):
                log("EVENT", f"marker {m}")
                if m.startswith("DONE_SEED"):
                    run("analyze_online.py", m)
                elif m.startswith("DONE_REPLAY"):
                    run("analyze_replay.py", m)
                    run("integrity_audit.py", m)
                    # seed 하나가 온전히 끝날 때마다 패키지를 다시 만들어 둔다.
                    # 뒤 단계에서 무슨 일이 나도 완결된 seed까지는 항상 손에 남는다.
                    run("make_package.py", m)
            seen_markers = now

            if done_all:
                log("EVENT", "DONE_PACKAGE 확인 — watchdog 종료")
                return
        except Exception as e:                        # 감시자가 죽으면 안 된다
            log("ALERT", f"watchdog 내부 예외 (계속 진행): {type(e).__name__}: {e}")
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()
