"""E0-1/E0-2 산출물 생성.

- results/e0/e0_1_envs.json  : 실행 env의 버전·CUDA capability·GPU matmul 스모크 (env별로 1회씩 실행)
- results/e0/e0_2_libero.json: LIBERO 커밋·스위트 구성 (hv2_oft에서 1회)
- configs/task_registry.json : 4 스위트 × 10 태스크 지시어 전문 (§2.4 1층 클러스터링 원본)

실행:
  LIBERO_CONFIG_PATH=$HABIT2/.libero conda run -n hv2_oft python -u experiments/e0_env_registry.py --env hv2_oft --libero
  LIBERO_CONFIG_PATH=$HABIT2/.libero conda run -n hv2_hab python -u experiments/e0_env_registry.py --env hv2_hab
"""
import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E0 = os.path.join(ROOT, "results", "e0")

# LIBERO_CONFIG_PATH 미지정 실행이 공용 ~/.libero를 오염시키는 사고 방지 (ISSUE-13 가드)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(ROOT, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def env_report(env_name):
    import numpy as np
    import torch

    ok_cuda = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if ok_cuda else None
    matmul_ok = False
    if ok_cuda:
        a = torch.randn(512, 512, device="cuda")
        b = (a @ a).sum().item()
        matmul_ok = bool(abs(b) < 1e9 and b == b)  # finite
    return {
        "env": env_name,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": ok_cuda,
        "device_capability": list(cap) if cap else None,
        "gpu_matmul_ok": matmul_ok,
        "attn": "sdpa (flash-attn 미빌드, sm_120)",
        "numpy": np.__version__,
        "transformers": pkg("transformers"),
        "robosuite": pkg("robosuite"),
        "mujoco": pkg("mujoco"),
        "libero": pkg("libero"),
        "bddl": pkg("bddl"),
        "tensorflow": pkg("tensorflow"),
        "opencv-python": pkg("opencv-python"),
        "scikit-learn": pkg("scikit-learn"),
    }


def libero_report():
    from libero.libero import benchmark

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.join(ROOT, "third_party", "LIBERO"), text=True
    ).strip()
    d = benchmark.get_benchmark_dict()
    suites = {}
    registry = {}
    for name in SUITES:
        s = d[name]()
        tasks = []
        for i in range(s.n_tasks):
            t = s.get_task(i)
            init = s.get_task_init_states(i)
            tasks.append(
                {
                    "task_id": i,
                    "name": t.name,
                    "language": t.language,
                    "bddl_file": t.bddl_file,
                    "init_states_shape": list(init.shape),
                }
            )
        suites[name] = {"n_tasks": s.n_tasks, "tasks": tasks}
        registry[name] = {t["task_id"]: t["language"] for t in tasks}
    report = {
        "libero_commit": commit,
        "libero_local_patch": "configs/libero_local.patch (torch.load weights_only=False)",
        "task_order_index": 0,
        "suites": suites,
    }
    return report, registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--libero", action="store_true", help="also emit e0_2 + task registry")
    args = ap.parse_args()

    os.makedirs(E0, exist_ok=True)

    rep = env_report(args.env)
    path = os.path.join(E0, "e0_1_envs.json")
    merged = {}
    if os.path.exists(path):
        merged = json.load(open(path))
    merged[args.env] = rep
    json.dump(merged, open(path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))

    status = "PASS" if (rep["cuda_available"] and rep["gpu_matmul_ok"] and rep["device_capability"] == [12, 0]) else "FAIL"
    print(f"[E0-PASS] item=E0-1:{args.env} status={status} json=results/e0/e0_1_envs.json")

    if args.libero:
        lib, registry = libero_report()
        json.dump(lib, open(os.path.join(E0, "e0_2_libero.json"), "w"), indent=2, ensure_ascii=False)
        json.dump(registry, open(os.path.join(ROOT, "configs", "task_registry.json"), "w"), indent=2, ensure_ascii=False)
        n = sum(s["n_tasks"] for s in lib["suites"].values())
        ok = n == 40
        print(f"[E0-PASS] item=E0-2 status={'PASS' if ok else 'FAIL'} json=results/e0/e0_2_libero.json")
        if not ok:
            sys.exit(1)
    if status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
