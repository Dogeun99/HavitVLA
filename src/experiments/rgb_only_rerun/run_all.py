"""RGB-only full rerun 마스터 오케스트레이터 (§16 무인 주말 실행).

marker 기반 resume — 이미 끝난 stage는 건너뛴다. 한 stage가 실패해도
**독립적인 다음 작업은 계속**한다 (§16). 각 stage 뒤 WEEKEND_RUN_STATUS를 갱신한다.

실행: setsid nohup hv2_hab python -u experiments/rgb_only_rerun/run_all.py \
        > logs/rgb_only_rerun/run_all.log 2>&1 < /dev/null &
"""
import os
import subprocess
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import (PY_HAB, PY_OFT, ROOT,  # noqa: E402
                                               has_marker, run_job, write_marker)

CK_ONLINE = "checkpoints/rgb_only_rerun/online"
DATA_ONLINE = "data/rgb_only_rerun/online"
RR = "experiments/rgb_only_rerun"


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def update_status():
    subprocess.run([PY_HAB, "-u", f"{RR}/status.py"], cwd=HABIT2,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stage_batch():
    return run_job("batch", [PY_HAB, "-u", f"{RR}/run_batch.py"],
                   marker="DONE_BATCH_STAGE", log="stage_batch.log")


def stage_online(s):
    """4,000 ep 스트림. 기술적 오류가 없는 한 성능과 무관하게 완주한다 (§8)."""
    os.makedirs(seed_dir(s), exist_ok=True)
    return run_job(
        f"online_seed{s}",
        [PY_OFT, "-u", "experiments/e5_driver.py", "--seed-idx", str(s), "--n", "4000",
         "--no-depth", "--out-root", seed_dir(s), "--ck-root", CK_ONLINE,
         "--data-root", DATA_ONLINE],
        marker=f"DONE_SEED{s}", log=f"online_seed{s}.log", success_marker_str="[E5-DONE]")


def stage_replay(s):
    """§11 PAIRED FULL-VLA REPLAY — 새 RGB-only run의 발화 집합에서 새로 추출."""
    os.makedirs(f"{ROOT}/05_paired_replay", exist_ok=True)
    return run_job(
        f"replay_seed{s}",
        [PY_OFT, "-u", "experiments/e5_counterfactual.py", "--seed-idx", str(s),
         "--queue-root", seed_dir(s), "--out-root", f"{ROOT}/05_paired_replay"],
        marker=f"DONE_REPLAY{s}", log=f"replay_seed{s}.log", success_marker_str="[E5CF-DONE]")


ANALYSIS = [
    ("batch_statistics", f"{RR}/analyze_batch.py", "DONE_BATCH_STATS", "[BATCH-STATS-DONE]"),
    ("online_summary", f"{RR}/analyze_online.py", "DONE_ONLINE_SUMMARY", "[ONLINE-SUMMARY-DONE]"),
    ("paired_summary", f"{RR}/analyze_replay.py", "DONE_PAIRED_SUMMARY", "[PAIRED-DONE]"),
    ("familiarity", f"{RR}/analyze_familiarity.py", "DONE_FAMILIARITY", "[FAMILIARITY-DONE]"),
    ("latency", f"{RR}/measure_latency.py", "DONE_LATENCY", "[LATENCY-DONE]"),
    ("old_vs_new", f"{RR}/old_vs_new.py", "DONE_OLDVSNEW", "[OLDVSNEW-DONE]"),
    ("integrity", f"{RR}/integrity_audit.py", "DONE_INTEGRITY", "[INTEGRITY-DONE]"),
    ("package", f"{RR}/make_package.py", "DONE_PACKAGE", "[PACKAGE-DONE]"),
]


def wait_for_external_batch():
    """별도로 선행 기동된 run_batch.py가 있으면 끝날 때까지 기다린다.

    같은 클러스터를 두 프로세스가 동시에 잡으면 marker 경합이 생긴다. GPU도 하나뿐이라
    동시 실행은 어차피 이득이 없다."""
    import time
    while True:
        r = subprocess.run(["pgrep", "-f", "rgb_only_rerun/run_batch.py"],
                           capture_output=True, text=True)
        pids = [x for x in r.stdout.split() if x and int(x) != os.getpid()]
        if not pids:
            return
        print(f"[WAIT] 선행 batch 프로세스 {pids} 종료 대기…", flush=True)
        time.sleep(60)


def main():
    print("=== RGB-only full rerun 오케스트레이터 시작 ===", flush=True)
    wait_for_external_batch()
    update_status()

    # 1. 배치 형성 (별도 프로세스로 선행 기동됐을 수 있다 — marker로 흡수)
    if not has_marker("DONE_BATCH"):
        stage_batch()
    update_status()

    # 2. 온라인 3 seed + paired replay (seed마다 replay를 이어 붙여, 도중에 멈춰도
    #    완결된 seed는 paired 기준선까지 갖춘 상태가 되게 한다)
    for s in (0, 1, 2):
        if stage_online(s):
            update_status()
            stage_replay(s)
        else:
            print(f"[SKIP-REPLAY] seed {s} 스트림 미완료 — replay 보류", flush=True)
        update_status()

    # 3. 분석 (§7·§10·§11·§12·§13·§14·§15). 하나가 실패해도 나머지는 계속한다.
    for name, script, marker, pass_str in ANALYSIS:
        if not os.path.exists(script):
            print(f"[SKIP] {name} — 스크립트 없음 {script}", flush=True)
            continue
        py = PY_OFT if name == "latency" else PY_HAB
        run_job(name, [py, "-u", script], marker=marker, log=f"{name}.log",
                success_marker_str=pass_str)
        update_status()

    update_status()
    print("=== 오케스트레이터 종료 ===", flush=True)


if __name__ == "__main__":
    main()
