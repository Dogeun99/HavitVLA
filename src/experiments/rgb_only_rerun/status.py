"""§17 STATUS FILE — WEEKEND_RUN_STATUS.{json,md} 갱신.

원장(JOBS_LEDGER.jsonl) + marker + 진행 파일의 실제 행 수만 읽는다. 추정 진행률 없음.
실행: hv2_hab python -u experiments/rgb_only_rerun/status.py
"""
import json
import os
import subprocess
import sys
import time

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import (FAILED, JOBS_LEDGER, MARKERS,  # noqa: E402
                                               ROOT, has_marker)

N_CLUSTERS, N_EPISODES = 27, 4000
STAGES = ["preflight", "smoke", "batch", "online_seed0", "replay_seed0",
          "online_seed1", "replay_seed1", "online_seed2", "replay_seed2",
          "batch_statistics", "online_summary", "familiarity", "latency",
          "integrity", "old_vs_new", "package"]


def _lines(p):
    if not os.path.exists(p):
        return 0
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def collect():
    st = {"run_id": os.path.basename(ROOT), "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
          "stages": {}, "progress": {}, "errors": [], "retries": []}

    # --- batch: 클러스터별 eval marker 수
    n_batch = len([f for f in os.listdir(MARKERS) if f.startswith("batch_eval_")]) \
        if os.path.isdir(MARKERS) else 0
    st["progress"]["batch"] = f"{n_batch}/{N_CLUSTERS}"

    # --- online / replay: 실제 기록 행 수
    for s in (0, 1, 2):
        ep = _lines(f"{ROOT}/0{s+2}_online_seed{s}/stream_{s}.jsonl")
        st["progress"][f"online_seed{s}"] = f"{ep}/{N_EPISODES}"
        q = _lines(f"{ROOT}/0{s+2}_online_seed{s}/cf_queue_{s}.jsonl")
        r = _lines(f"{ROOT}/05_paired_replay/cf_{s}.jsonl")
        st["progress"][f"replay_seed{s}"] = f"{r}/{q}" if q else "WAIT"

    # --- stage 상태
    marker_of = {"preflight": "DONE_PREFLIGHT", "smoke": "DONE_SMOKE", "batch": "DONE_BATCH",
                 "online_seed0": "DONE_SEED0", "online_seed1": "DONE_SEED1",
                 "online_seed2": "DONE_SEED2",
                 "replay_seed0": "DONE_REPLAY0", "replay_seed1": "DONE_REPLAY1",
                 "replay_seed2": "DONE_REPLAY2",
                 "batch_statistics": "DONE_BATCH_STATS", "online_summary": "DONE_ONLINE_SUMMARY",
                 "familiarity": "DONE_FAMILIARITY", "latency": "DONE_LATENCY",
                 "integrity": "DONE_INTEGRITY", "old_vs_new": "DONE_OLDVSNEW",
                 "package": "DONE_PACKAGE"}
    for s in STAGES:
        if has_marker(marker_of[s]):
            st["stages"][s] = "PASS"
        elif st["progress"].get(s, "").split("/")[0] not in ("0", "", "WAIT"):
            st["stages"][s] = "RUNNING"
        else:
            st["stages"][s] = "WAIT"

    # --- 원장에서 elapsed / retry / error
    if os.path.exists(JOBS_LEDGER):
        recs = [json.loads(l) for l in open(JOBS_LEDGER)]
        st["elapsed_total_h"] = round(sum(r["elapsed_s"] for r in recs) / 3600, 2)
        st["n_jobs"] = len(recs)
        st["retries"] = [{"job": r["job"], "attempt": r["attempt"], "exit": r["exit_code"]}
                         for r in recs if r["attempt"] > 0]
        st["last_jobs"] = [{"job": r["job"], "ok": r["ok"], "min": round(r["elapsed_s"] / 60, 1),
                            "end": r["end"]} for r in recs[-6:]]
        # ETA: 완료 job의 평균에서 남은 stage를 추정하지 않는다 — 실측 기반만 기록
        done_ep = sum(int(st["progress"][f"online_seed{s}"].split("/")[0]) for s in (0, 1, 2))
        st["online_episodes_done"] = f"{done_ep}/{3 * N_EPISODES}"
    if os.path.exists(FAILED):
        st["errors"] = json.load(open(FAILED))

    # --- ETA: 진행 중 seed는 자기 속도로, 남은 stage는 실측 단가로 추정한다.
    import datetime
    eta_h, notes = 0.0, []
    prior_seed_h, prior_replay_h = 16.6, 4.5      # 기존 RGB-D run 실측 (results/e5)
    # 본 run에서 완주한 seed가 있으면 그 실측값이 사전 단가보다 낫다.
    measured = []
    for s_ in (0, 1, 2):
        sp_ = f"{ROOT}/0{s_ + 2}_online_seed{s_}/summary_{s_}.json"
        if os.path.exists(sp_):
            try:
                measured.append(json.load(open(sp_))["total_wall_s"] / 3600)
            except Exception:
                pass
    if measured:
        prior_seed_h = round(sum(measured) / len(measured), 2)
        st["measured_seed_h"] = prior_seed_h
    done_seed = [v for v in st["stages"] if v.startswith("online_seed")
                 and st["stages"][v] == "PASS"]
    for s in (0, 1, 2):
        stage, prog = f"online_seed{s}", st["progress"].get(f"online_seed{s}", "0/4000")
        n_done = int(prog.split("/")[0])
        if st["stages"][stage] == "PASS":
            continue
        if n_done > 0:                              # 진행 중 — 자기 속도 사용
            p_ = f"{ROOT}/0{s + 2}_online_seed{s}/stream_{s}.jsonl"
            try:
                first = json.loads(open(p_).readline())["wall_clock_time"]
                el = (datetime.datetime.now()
                      - datetime.datetime.strptime(first, "%Y-%m-%d %H:%M:%S")).total_seconds()
                rate = n_done / (el / 3600)
                st[f"rate_ep_per_h_seed{s}"] = round(rate, 1)
                eta_h += (N_EPISODES - n_done) / rate
                notes.append(f"seed{s} 잔여 {(N_EPISODES - n_done) / rate:.1f}h (실측 {rate:.0f} ep/h)")
            except Exception:
                eta_h += prior_seed_h
        else:
            eta_h += prior_seed_h
            notes.append(f"seed{s} {prior_seed_h}h (사전 실측 단가)")
    for s in (0, 1, 2):
        if st["stages"][f"replay_seed{s}"] == "PASS":
            continue
        prog = st["progress"].get(f"replay_seed{s}", "WAIT")
        n_done = int(prog.split("/")[0]) if "/" in prog else 0
        n_tot = int(prog.split("/")[1]) if "/" in prog else 0
        if n_done > 0 and n_tot:
            # 프로세스 경과 시간으로 속도 산출 (파일 mtime은 쓰기마다 갱신돼 못 쓴다)
            try:
                pid = subprocess.run(["pgrep", "-f", "e5_counterfactual[.]py"],
                                     capture_output=True, text=True).stdout.split()[0]
                el = int(subprocess.run(["ps", "-o", "etimes=", "-p", pid],
                                        capture_output=True, text=True).stdout.strip())
                rate = n_done / (el / 3600)
                eta_h += (n_tot - n_done) / rate
                notes.append(f"replay{s} 잔여 {(n_tot - n_done) / rate:.1f}h (실측 {rate:.0f} ep/h)")
                continue
            except Exception:
                pass
        eta_h += prior_replay_h
    for stage in ("batch_statistics", "online_summary", "familiarity", "latency",
                  "integrity", "old_vs_new", "package"):
        if st["stages"][stage] != "PASS":
            eta_h += 0.25
    st["eta_remaining_h"] = round(eta_h, 1)
    st["eta_finish"] = (datetime.datetime.now()
                        + datetime.timedelta(hours=eta_h)).strftime("%Y-%m-%d %H:%M")
    st["eta_basis"] = notes + [f"미착수 seed {prior_seed_h}h · replay {prior_replay_h}h · 분석 0.25h/stage"]
    return st


def render_md(st):
    L = [f"# WEEKEND_RUN_STATUS — {st['run_id']}", "",
         f"갱신 {st['updated']}", "", "## Stages", "",
         "| stage | status | progress |", "|---|---|---|"]
    for s in STAGES:
        L.append(f"| {s} | {st['stages'][s]} | {st['progress'].get(s, '')} |")
    L += ["", "## 집계", "",
          f"- 잔여 예상: **{st.get('eta_remaining_h', '?')} h** → 완료 예상 "
          f"**{st.get('eta_finish', '?')}**",
          f"- 누적 실행 시간: **{st.get('elapsed_total_h', 0)} h** (job {st.get('n_jobs', 0)}개)",
          f"- 온라인 에피소드: **{st.get('online_episodes_done', '0/12000')}**",
          f"- 재시도: {len(st.get('retries', []))}건",
          f"- 실패 job: {len(st.get('errors', []))}건"]
    if st.get("eta_basis"):
        L += ["", "ETA 근거: " + " · ".join(st["eta_basis"])]
    if st.get("errors"):
        L += ["", "## 실패 job", ""]
        for e in st["errors"]:
            L.append(f"- `{e['job']}` exit={e['exit_code']} log=`{e['log']}`")
    if st.get("last_jobs"):
        L += ["", "## 최근 job", "", "| job | ok | min | end |", "|---|---|---|---|"]
        for j in st["last_jobs"]:
            L.append(f"| {j['job']} | {j['ok']} | {j['min']} | {j['end']} |")
    return "\n".join(L) + "\n"


def main():
    st = collect()
    json.dump(st, open(f"{ROOT}/WEEKEND_RUN_STATUS.json", "w"), indent=1, ensure_ascii=False)
    open(f"{ROOT}/WEEKEND_RUN_STATUS.md", "w").write(render_md(st))
    print(render_md(st))


if __name__ == "__main__":
    main()
