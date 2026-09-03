"""§16 무인 실행 프레임워크 — resume · 완료 marker · retry 1회 · 작업 원장 · 상태 파일.

원칙 (§16):
  - 완료 marker가 있으면 건너뛴다 (resume).
  - transient 실패에 한해 **동일 command** 1회 재시도. batch size·precision·seed·steps·lr·
    model setting은 자동 변경하지 않는다.
  - 재시도도 실패하면 FAILED_JOBS.json에 기록하고 **독립적인 다음 작업은 계속**한다.
"""
import json
import os
import subprocess
import time

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_ID = "rgb_only_full_rerun_20260828"
ROOT = os.path.join(HABIT2, "results", RUN_ID)
MARKERS = os.path.join(ROOT, "markers")
LOGS = os.path.join(ROOT, "logs")
JOBS_LEDGER = os.path.join(ROOT, "JOBS_LEDGER.jsonl")
FAILED = os.path.join(ROOT, "FAILED_JOBS.json")

# release: interpreters of the two conda envs (override with HV2_HAB_PY / HV2_OFT_PY)
PY_HAB = os.environ.get("HV2_HAB_PY", os.path.expanduser("~/miniconda3/envs/hv2_hab/bin/python"))
PY_OFT = os.environ.get("HV2_OFT_PY", os.path.expanduser("~/miniconda3/envs/hv2_oft/bin/python"))


def marker_path(name):
    return os.path.join(MARKERS, name)


def has_marker(name):
    return os.path.exists(marker_path(name))


def write_marker(name, payload=None):
    os.makedirs(MARKERS, exist_ok=True)
    with open(marker_path(name), "w") as f:
        json.dump(payload or {"done_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  indent=1, ensure_ascii=False)


def _append_ledger(rec):
    os.makedirs(ROOT, exist_ok=True)
    with open(JOBS_LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _record_failure(job, rec):
    cur = json.load(open(FAILED)) if os.path.exists(FAILED) else []
    cur.append(rec)
    json.dump(cur, open(FAILED, "w"), indent=1, ensure_ascii=False)


def run_job(name, cmd, marker=None, log=None, cwd=HABIT2, env=None, max_retry=1,
            success_marker_str=None):
    """command 1개 실행. 반환 True = 성공(또는 이미 완료).

    success_marker_str: stdout/로그에 반드시 있어야 하는 명시적 PASS 마커
    (CLAUDE.md §6 — "error" grep 금지, 명시 마커로 판정).
    """
    if marker and has_marker(marker):
        print(f"[SKIP] {name} (marker={marker})", flush=True)
        return True
    os.makedirs(LOGS, exist_ok=True)
    logp = os.path.join(LOGS, log or f"{name}.log")
    e = dict(os.environ)
    e.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
    e.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))
    e.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
    e.setdefault("MUJOCO_GL", "egl")
    if env:
        e.update(env)

    for attempt in range(max_retry + 1):
        t0 = time.time()
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[RUN] {name} attempt={attempt} → {os.path.relpath(logp, HABIT2)}", flush=True)
        with open(logp, "a") as lf:
            lf.write(f"\n===== {name} attempt={attempt} start={start} =====\n")
            lf.write(" ".join(cmd) + "\n")
            lf.flush()
            r = subprocess.run(cmd, cwd=cwd, env=e, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = r.returncode == 0
        if ok and success_marker_str:
            ok = success_marker_str in open(logp, errors="replace").read()
        rec = {"job": name, "attempt": attempt, "cmd": cmd, "start": start,
               "end": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(dt, 1),
               "exit_code": r.returncode, "log": os.path.relpath(logp, HABIT2),
               "pass_marker": success_marker_str, "ok": ok}
        _append_ledger(rec)
        if ok:
            if marker:
                write_marker(marker, rec)
            print(f"[OK] {name} ({dt/60:.1f} min)", flush=True)
            return True
        print(f"[FAIL] {name} attempt={attempt} exit={r.returncode} ({dt/60:.1f} min)", flush=True)
        if attempt == max_retry:
            _record_failure(name, rec)
    return False
