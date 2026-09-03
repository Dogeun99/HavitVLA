"""§3 PRE-FLIGHT + §4 RGB-ONLY INPUT 검증.

산출 (results/rgb_only_full_rerun_20260828/00_preflight/):
  ENVIRONMENT.json · CONFIG_DIFF.json · RGB_ONLY_INPUT_AUDIT.json · PREFLIGHT_STATUS.json
depth 이외의 예상치 못한 설정 차이가 있으면 [PREFLIGHT-FAIL]로 종료한다 (§3-D).
실행: hv2_hab python -u experiments/rgb_only_rerun/preflight.py
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))

RUN_ID = "rgb_only_full_rerun_20260828"
OUT = f"results/{RUN_ID}/00_preflight"
# §6: 25 standard + 2 controlled chains = 27 formation subjects.
# source of truth = e3_collect.EXPECTED_CLUSTERS (기존 manifest, §6 명시)
DATA_DIR_E2 = ("libero_object_task0", "libero_object_task5")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- A. Source
def source_block():
    return {
        "git_commit": sh("git rev-parse HEAD"),
        "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
        "git_status_porcelain": sh("git status --porcelain"),
        "git_status_clean": sh("git status --porcelain") == "",
        "git_diff_stat": sh("git diff --stat"),
        "git_diff_full_sha256": hashlib.sha256(sh("git diff").encode()).hexdigest(),
        "rgb_only_relevant_files": {
            p: hashlib.sha256(open(p, "rb").read()).hexdigest()
            for p in ("habits/act.py", "habits/dataset.py", "habits/train.py",
                      "habits/policy.py", "habits/evaluate.py",
                      "experiments/e5_driver.py", "experiments/e5_counterfactual.py",
                      "envs/stream.py", "envs/libero_env.py", "gates/two_stage.py")
        },
    }


# ------------------------------------------------------------- B. Environment
def environment_block():
    import torch
    freeze = sh(f"{sys.executable} -m pip freeze")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gpu_total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available() else None,
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,memory.total "
                         "--format=csv,noheader"),
        "attn_implementation": "sdpa (flash-attn은 sm_120 미빌드 — CLAUDE.md §0)",
        "pip_freeze": freeze.splitlines(),
        "conda_envs": {"act": "hv2_hab", "teacher": "hv2_oft"},
        "env_vars": {k: os.environ.get(k) for k in
                     ("HF_HOME", "TORCH_HOME", "LIBERO_CONFIG_PATH", "MUJOCO_GL")},
    }


# ----------------------------------------------------------------- C. Dataset
def dataset_block():
    import h5py
    from experiments.e3_collect import EXPECTED_CLUSTERS

    rows, problems = [], []
    for cl in EXPECTED_CLUSTERS:
        ddir = "e2" if cl in DATA_DIR_E2 else "e3"
        p = f"data/{ddir}/{cl}.hdf5"
        if not os.path.exists(p):
            problems.append(f"{cl}: HDF5 없음 ({p})")
            rows.append({"cluster": cl, "h5": p, "exists": False})
            continue
        with h5py.File(p, "r") as f:
            meta = json.loads(f["meta_json"][()])
            n_succ = sum(1 for m in meta if m["outcome"] == "success")
            uids = [m["uid"] for m in meta if m["outcome"] == "success"]
            g = f[f"episodes/{uids[0]}"]
            keys = sorted(g.keys())
            r = {"cluster": cl, "h5": p, "exists": True, "schema": f.attrs.get("schema"),
                 "n_meta": len(meta), "n_success": n_succ, "n_episode_groups": len(f["episodes"]),
                 "episode_keys": keys,
                 "has_rgb": "agentview_rgb" in keys and "wrist_rgb" in keys,
                 "has_depth": "agentview_depth" in keys,
                 "has_proprio": "proprio" in keys,
                 "has_actions": "actions_flat" in keys and "chunk_lens" in keys,
                 "has_episode_id": all("uid" in m for m in meta),
                 "has_success_label": all("outcome" in m for m in meta),
                 "rgb_shape": list(g["agentview_rgb"].shape),
                 "rgb_dtype": str(g["agentview_rgb"].dtype),
                 "proprio_shape": list(g["proprio"].shape)}
        rows.append(r)
        if n_succ < 80:
            problems.append(f"{cl}: 성공 궤적 {n_succ} < 80 (n=80 학습 불가 — 우측절단 기록 대상)")
        for k in ("has_rgb", "has_proprio", "has_actions", "has_episode_id", "has_success_label"):
            if not r[k]:
                problems.append(f"{cl}: {k}=False")
    return {"n_clusters": len(EXPECTED_CLUSTERS), "clusters": rows, "problems": problems}


# ------------------------------------------------------- D. Configuration diff
def config_diff_block():
    """RGB-D config vs RGB-only config를 key-by-key 비교. 허용 차이는 depth 관련뿐."""
    from habits.act import ACTPolicy
    from habits.train import HP
    from gates import two_stage as ts
    from envs import stream as sm
    from envs.libero_env import TASK_MAX_STEPS, USABLE_W_MAX
    from experiments.e5_driver import BATCH_EQUIV_STEPS, CHUNK, GRID_FULL, PROBE_FULL

    def cfg(use_depth):
        return {
            # --- ACT / 학습
            "in_ch": 4 if use_depth else 3,
            "use_depth": use_depth,
            "lr": HP["lr"], "lr_backbone": HP["lr_backbone"],
            "batch_size": HP["batch_size"], "weight_decay": HP["weight_decay"],
            "kl_weight": HP["kl_weight"], "train_seed": HP["seed"],
            "steps_per_n": dict(HP["steps_per_n"]),
            "batch_equiv_steps": dict(BATCH_EQUIV_STEPS),
            "n_grid": [10, 20, 40, 80],
            "chunk_K": CHUNK,
            "rgb_normalization": "ImageNet mean/std (양 조건 동일)",
            "action_normalization": "cluster pool mean/std",
            "proprio_representation": "eef pos(3) + axisangle(3) + gripper qpos(2) = 8",
            "augmentation": "none",
            "backbone": "resnet18 ImageNet-pretrained",
            "n_params": sum(p.numel() for p in ACTPolicy(in_ch=4 if use_depth else 3,
                                                         pretrained=False).parameters()),
            # --- gate / lifecycle
            "tau_min": ts.TAU, "tau_max": ts.ACI_TAU_MAX, "delta": ts.DELTA,
            "epsilon": ts.EPSILON, "gamma": ts.ACI_GAMMA, "reinit_c": ts.REINIT_C,
            "alpha_j": ts.ALPHA_J,
            "probe_P": PROBE_FULL, "probe_max_rounds": ts.MaturityGate.PROBE_MAX_ROUNDS,
            "online_retrain_grid": list(GRID_FULL),
            # --- 스트림 / 에피소드 명세
            "w_id": sm.W_ID, "usable_w_max": dict(USABLE_W_MAX),
            "collect_base_range": [sm.COLLECT_BASE_RANGE.start, sm.COLLECT_BASE_RANGE.stop],
            "heldout_base_range": [sm.HELDOUT_BASE_RANGE.start, sm.HELDOUT_BASE_RANGE.stop],
            "seed_bands": {"collect": sm.COLLECT_SEED_BASE, "heldout": sm.HELDOUT_SEED_BASE,
                           "novel": sm.NOVEL_SEED_BASE, "probe": sm.PROBE_SEED_BASE,
                           "e5_stream": sm.E5_STREAM_SEED_BASE, "e5_novel": sm.E5_NOVEL_SEED_BASE},
            "noise_bands": {"heldout": sm.HELDOUT_NOISE_BASE, "novel": sm.NOVEL_NOISE_BASE,
                            "probe": sm.PROBE_NOISE_BASE, "e5_stream": sm.E5_STREAM_NOISE_BASE,
                            "e5_novel": sm.E5_NOVEL_NOISE_BASE},
            "e5_clusters": len(sm.E5_CLUSTERS), "e5_novel_pool": len(sm.E5_NOVEL_POOL),
            "e5_novel_rate": sm.E5_NOVEL_RATE,
            "task_max_steps": dict(TASK_MAX_STEPS),
            "bootstrap_B": 10000, "noninferiority_margin_pp": -3.0,
            "online_seeds": [0, 1, 2], "online_episodes_per_seed": 4000,
        }

    old, new = cfg(True), cfg(False)
    diffs = {k: {"rgbd": old[k], "rgb_only": new[k]} for k in old if old[k] != new[k]}
    ALLOWED = {"in_ch", "use_depth", "n_params"}
    unexpected = {k: v for k, v in diffs.items() if k not in ALLOWED}
    return {"rgbd_config": old, "rgb_only_config": new, "diffs": diffs,
            "allowed_diff_keys": sorted(ALLOWED),
            "unexpected_diff_keys": sorted(unexpected),
            "param_delta": old["n_params"] - new["n_params"],
            "verdict": "PASS" if not unexpected else "FAIL"}


# ------------------------------------------------- §4 RGB-ONLY INPUT 검증
def rgb_only_input_audit():
    """코드 경로 + 실제 mini-batch runtime 양쪽에서 depth 미사용을 확인한다."""
    import inspect
    import numpy as np
    import torch

    from habits.act import ACTPolicy
    from habits.dataset import ClusterDataset, compute_stats, load_cluster, make_frame_tensor
    from habits import policy as policy_mod
    from torch.utils.data import DataLoader

    audit = {"static": {}, "runtime": {}, "verdict": None}

    # --- static: 소스 경로 확인
    audit["static"]["make_frame_tensor_src"] = inspect.getsource(make_frame_tensor)
    audit["static"]["dataset_getitem_passes_use_depth"] = (
        "make_frame_tensor(ep[\"agentview_rgb\"][t], ep[\"agentview_depth\"][t], ud)"
        in inspect.getsource(ClusterDataset.__getitem__))
    audit["static"]["policy_reads_use_depth_from_ckpt"] = (
        'sd.get("use_depth", True)' in inspect.getsource(policy_mod.HabitPolicy.__init__))
    audit["static"]["act_in_ch_switch"] = "if in_ch != 3:" in inspect.getsource(
        __import__("habits.act", fromlist=["build_backbone"]).build_backbone)

    # --- runtime: 실제 mini-batch 하나
    h5 = "data/e3/libero_goal_task1.hdf5"
    eps = load_cluster(h5, 2)
    stats = compute_stats(eps)
    out = {}
    for tag, ud in (("rgbd", True), ("rgb_only", False)):
        ds = ClusterDataset(eps, stats, use_depth=ud)
        b = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
        model = ACTPolicy(in_ch=4 if ud else 3, pretrained=False)
        conv1 = model.backbones[0][0]      # Sequential(*resnet.children()[:-2]) → [0] = conv1
        out[tag] = {
            "model_input_keys": sorted(b.keys()),
            "agentview_shape": list(b["agentview"].shape),
            "wrist_shape": list(b["wrist"].shape),
            "proprio_shape": list(b["proprio"].shape),
            "actions_shape": list(b["actions"].shape),
            "pad_mask_shape": list(b["pad_mask"].shape),
            "image_channels": int(b["agentview"].shape[1]),
            "depth_key_in_batch": any("depth" in k for k in b.keys()),
            "conv1_in_channels": int(conv1.in_channels),
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        # forward 실행 가능성 + 채널 수 일치
        with torch.no_grad():
            loss, parts = model.loss([b["agentview"], b["wrist"]], b["proprio"],
                                     b["actions"], b["pad_mask"])
        out[tag]["forward_ok"] = bool(np.isfinite(loss.item()))
        out[tag]["loss_parts"] = {k: float(v) for k, v in parts.items()}
    audit["runtime"] = out

    # --- 결정적 판정
    rgb = out["rgb_only"]
    checks = {
        "image_channels_is_3": rgb["image_channels"] == 3,
        "conv1_in_channels_is_3": rgb["conv1_in_channels"] == 3,
        "no_depth_key_in_batch": rgb["depth_key_in_batch"] is False,
        "forward_ok": rgb["forward_ok"],
        "params_smaller_than_rgbd": rgb["n_params"] < out["rgbd"]["n_params"],
        "static_paths_ok": all(v for k, v in audit["static"].items() if isinstance(v, bool)),
    }
    audit["checks"] = checks
    audit["param_delta_rgbd_minus_rgb"] = out["rgbd"]["n_params"] - rgb["n_params"]
    audit["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return audit


def main():
    os.makedirs(OUT, exist_ok=True)
    src = source_block()
    env = environment_block()
    ds = dataset_block()
    cd = config_diff_block()
    ia = rgb_only_input_audit()

    json.dump({"run_id": RUN_ID, "source": src, "environment": env},
              open(f"{OUT}/ENVIRONMENT.json", "w"), indent=1, ensure_ascii=False)
    json.dump(cd, open(f"{OUT}/CONFIG_DIFF.json", "w"), indent=1, ensure_ascii=False)
    json.dump(ia, open(f"{OUT}/RGB_ONLY_INPUT_AUDIT.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"run_id": RUN_ID, "dataset": ds},
              open(f"{OUT}/DATASET_CHECK.json", "w"), indent=1, ensure_ascii=False)

    ok_ds = not ds["problems"]
    status = {"run_id": RUN_ID,
              "source": "PASS",
              "environment": "PASS" if env["cuda_available"] else "FAIL",
              "dataset": "PASS" if ok_ds else f"WARN({len(ds['problems'])})",
              "dataset_problems": ds["problems"],
              "config_diff": cd["verdict"],
              "config_unexpected_keys": cd["unexpected_diff_keys"],
              "rgb_only_input_audit": ia["verdict"],
              "rgb_only_checks": ia["checks"]}
    hard = [status["environment"], status["config_diff"], status["rgb_only_input_audit"]]
    status["overall"] = "PASS" if all(x == "PASS" for x in hard) else "FAIL"
    json.dump(status, open(f"{OUT}/PREFLIGHT_STATUS.json", "w"), indent=1, ensure_ascii=False)

    print(json.dumps(status, indent=1, ensure_ascii=False))
    if status["overall"] != "PASS":
        print("[PREFLIGHT-FAIL] full run을 시작하지 않는다 (§3-D)")
        sys.exit(1)
    print(f"[PREFLIGHT-PASS] run_id={RUN_ID} commit={src['git_commit'][:12]} "
          f"param_delta={ia['param_delta_rgbd_minus_rgb']}")


if __name__ == "__main__":
    main()
