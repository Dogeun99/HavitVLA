"""§5 SMOKE TEST — 기술적 정상성만 확인한다. 성능을 보고 설정을 바꾸지 않는다.

검사: training / inference / checkpoint save-load / success evaluation / logging /
      depth leakage 없음.
산출: results/<RUN_ID>/00_preflight/SMOKE.json
실행: hv2_hab python -u experiments/rgb_only_rerun/smoke.py
"""
import json
import os
import subprocess
import sys
import time

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import PY_HAB, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "00_preflight")
SMOKE_CK = os.path.join(HABIT2, "checkpoints", "rgb_only_rerun", "smoke")
SMOKE_RES = os.path.join(ROOT, "00_preflight", "smoke_eval")
CLUSTER, SUITE, TASK = "libero_goal_task1", "libero_goal", 1
STEPS, N_EVAL = 200, 3


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {"cluster": CLUSTER, "steps": STEPS, "n_eval": N_EVAL, "checks": {}}
    t0 = time.time()

    # 1. training
    r = subprocess.run(
        [PY_HAB, "-u", "habits/train.py", "--h5", f"data/e3/{CLUSTER}.hdf5",
         "--cluster", CLUSTER, "--n-grid", "10", "--out", SMOKE_CK,
         "--no-depth", "--steps", str(STEPS)],
        capture_output=True, text=True, cwd=HABIT2)
    rep["train_returncode"] = r.returncode
    rep["train_tail"] = r.stdout[-400:]
    rep["checks"]["training_runs"] = r.returncode == 0 and "[TRAIN-PASS]" in r.stdout

    ckpt = os.path.join(SMOKE_CK, CLUSTER, "act_n10.pt")
    rep["checks"]["checkpoint_saved"] = os.path.exists(ckpt)
    if not rep["checks"]["checkpoint_saved"]:
        json.dump(rep, open(f"{OUT}/SMOKE.json", "w"), indent=1, ensure_ascii=False)
        print("[SMOKE-FAIL] 체크포인트 미생성"); sys.exit(1)

    # 2. checkpoint load + depth leakage (런타임)
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    rep["checkpoint_meta"] = {k: sd[k] for k in ("use_depth", "in_ch", "steps", "n_episodes",
                                                 "n_params", "final_l1") if k in sd}
    rep["checks"]["ckpt_use_depth_false"] = sd.get("use_depth") is False
    rep["checks"]["ckpt_in_ch_3"] = sd.get("in_ch") == 3
    rep["checks"]["ckpt_steps_as_specified"] = sd.get("steps") == STEPS

    from habits.policy import HabitPolicy
    pol = HabitPolicy(ckpt)
    conv1 = pol.model.backbones[0][0]
    rep["runtime_policy"] = {"use_depth": pol.use_depth,
                             "conv1_in_channels": int(conv1.in_channels),
                             "n_backbones": len(pol.model.backbones)}
    rep["checks"]["policy_use_depth_false"] = pol.use_depth is False
    rep["checks"]["policy_conv1_is_3ch"] = int(conv1.in_channels) == 3
    del pol
    torch.cuda.empty_cache()

    # 3. inference + success evaluation + logging
    os.makedirs(SMOKE_RES, exist_ok=True)
    r2 = subprocess.run(
        [PY_HAB, "-u", "habits/evaluate.py", "--cluster", CLUSTER, "--suite", SUITE,
         "--task", str(TASK), "--ckpt-dir", os.path.join(SMOKE_CK, CLUSTER),
         "--n-grid", "10", "--n-heldout", str(N_EVAL), "--out", SMOKE_RES],
        capture_output=True, text=True, cwd=HABIT2)
    rep["eval_returncode"] = r2.returncode
    rep["eval_tail"] = r2.stdout[-400:]
    rep["checks"]["inference_runs"] = r2.returncode == 0 and "[EVAL-PASS]" in r2.stdout

    cp = os.path.join(SMOKE_RES, f"{CLUSTER}_curve.json")
    if os.path.exists(cp):
        c = json.load(open(cp))
        e = c["curve"][0]
        rep["eval_result"] = {"n_eval": e["n_eval"], "n_success": e["n_success"],
                              "s_hat": e["s_hat"], "n_infra_error": e["n_infra_error"],
                              "per_episode": e["per_episode"]}
        rep["checks"]["success_evaluation_works"] = e["n_eval"] == N_EVAL
        rep["checks"]["logging_per_episode"] = len(e["per_episode"]) == N_EVAL
        rep["checks"]["no_infra_error"] = e["n_infra_error"] == 0
    else:
        rep["checks"]["success_evaluation_works"] = False
        rep["checks"]["logging_per_episode"] = False

    rep["wall_s"] = round(time.time() - t0, 1)
    rep["verdict"] = "PASS" if all(rep["checks"].values()) else "FAIL"
    rep["note"] = ("기술적 정상성만 판정한다. s_hat 값은 200스텝 축소 학습의 산물이며 "
                   "설정 변경 근거로 쓰지 않는다 (§5).")
    json.dump(rep, open(f"{OUT}/SMOKE.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"checks": rep["checks"], "verdict": rep["verdict"],
                      "eval": rep.get("eval_result", {}).get("s_hat"),
                      "wall_min": round(rep["wall_s"] / 60, 1)}, indent=1))
    if rep["verdict"] != "PASS":
        print("[SMOKE-FAIL]"); sys.exit(1)
    print("[SMOKE-PASS]")


if __name__ == "__main__":
    main()
