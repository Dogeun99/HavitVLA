"""§6 BATCH HABIT FORMATION 전체 재실행 (RGB-only ACT, 27 formation subjects).

프로토콜은 E3와 동일 — 변경은 depth 제거 하나뿐:
  - cluster source of truth = experiments/e3_collect.EXPECTED_CLUSTERS (25 표준 + 2 통제 체인)
  - teacher 성공 궤적 = 기존 data/{e2,e3}/<cluster>.hdf5 (동결)
  - n-grid {10,20,40,80}, warm-start 체인, steps_per_n {4000,6000,8000,10000}
  - held-out 평가 세트 = 기존과 동일 (E3 20 / object_task0·task5 50 / chained 50)
resume: 클러스터별 marker. 실패해도 다음 클러스터를 계속한다 (§16).
실행: hv2_hab python -u experiments/rgb_only_rerun/run_batch.py
"""
import os
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.e3_collect import EXPECTED_CLUSTERS, T2_CHAINED  # noqa: E402
from experiments.rgb_only_rerun.runner import PY_HAB, ROOT, run_job, write_marker  # noqa: E402

CKROOT = os.path.join(HABIT2, "checkpoints", "rgb_only_rerun", "batch")
CURVES = os.path.join(ROOT, "01_batch_formation", "curves")
# 기존 프로토콜의 held-out 규모 (results/e2·e3 실측에서 확인)
E2_CLUSTERS = ("libero_object_task0", "libero_object_task5")   # E2 프로토콜 = 50
N_HELDOUT = {c: 50 for c in list(E2_CLUSTERS) + list(T2_CHAINED)}


def parse(cluster):
    """cluster id → (suite, task, chained)."""
    if cluster.startswith("chained_"):
        base = cluster[len("chained_"):]
    else:
        base = cluster
    suite, task = base.rsplit("_task", 1)
    return suite, int(task), cluster.startswith("chained_")


def main():
    os.makedirs(CKROOT, exist_ok=True)
    os.makedirs(CURVES, exist_ok=True)
    done, failed = [], []
    for i, cl in enumerate(EXPECTED_CLUSTERS, 1):
        suite, task, chained = parse(cl)
        ddir = "e2" if cl in E2_CLUSTERS else "e3"
        h5 = os.path.join(HABIT2, "data", ddir, f"{cl}.hdf5")
        nh = N_HELDOUT.get(cl, 20)
        print(f"\n########## [{i}/{len(EXPECTED_CLUSTERS)}] {cl} "
              f"(suite={suite} task={task} chained={chained} heldout={nh}) ##########", flush=True)

        ok_tr = run_job(
            f"batch_train_{cl}",
            [PY_HAB, "-u", "habits/train.py", "--h5", h5, "--cluster", cl,
             "--n-grid", "10", "20", "40", "80", "--out", CKROOT, "--no-depth"],
            marker=f"batch_train_{cl}", log=f"batch_train_{cl}.log",
            success_marker_str="[TRAIN-PASS]")
        if not ok_tr:
            failed.append(cl)
            continue

        cmd = [PY_HAB, "-u", "habits/evaluate.py", "--cluster", cl, "--suite", suite,
               "--task", str(task), "--ckpt-dir", os.path.join(CKROOT, cl),
               "--n-grid", "10", "20", "40", "80", "--n-heldout", str(nh), "--out", CURVES]
        if chained:
            cmd.append("--chained")
        ok_ev = run_job(f"batch_eval_{cl}", cmd, marker=f"batch_eval_{cl}",
                        log=f"batch_eval_{cl}.log", success_marker_str="[EVAL-PASS]")
        (done if ok_ev else failed).append(cl)

    print(f"\n[BATCH-SUMMARY] 완료 {len(done)}/{len(EXPECTED_CLUSTERS)} · 실패 {failed}")
    if len(done) == len(EXPECTED_CLUSTERS):
        write_marker("DONE_BATCH", {"n_clusters": len(done), "clusters": done})
        print("[DONE_BATCH]")
    else:
        print(f"[BATCH-PARTIAL] {len(done)}/{len(EXPECTED_CLUSTERS)} — DONE_BATCH 미발급")


if __name__ == "__main__":
    main()
