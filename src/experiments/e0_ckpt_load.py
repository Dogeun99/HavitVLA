"""E0-4b: OFT 4 체크포인트 로드 + 더미 forward 검증 (순차 — VRAM 동시 로드 금지).

공식 initialize_model()과 sample observation pkl을 그대로 사용해 로드 경로를
run_libero_eval.py와 동일하게 유지한다.

실행(반드시 openvla-oft 루트가 cwd — experiments 패키지 해석):
  cd $HABIT2/third_party/openvla-oft && HF_HOME=$HABIT2/.hf_cache \
  conda run -n hv2_oft python -u $HABIT2/experiments/e0_ckpt_load.py
"""
import json
import os
import pickle
import sys
import time

import torch

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
OUT = os.path.join(HABIT2, "results", "e0", "e0_4_ckpt.json")
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))

# 공용 캐시/설정 오염 방지 가드 (ISSUE-13): 미지정 시 프로젝트 로컬로 강제
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

SUITES = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}


def main():
    assert os.environ.get("HF_HOME", "").startswith(HABIT2), "HF_HOME must be project-local"
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.openvla_utils import get_vla_action

    sample_path = os.path.join(
        HABIT2, "third_party", "openvla-oft", "experiments", "robot", "libero",
        "sample_libero_spatial_observation.pkl",
    )
    with open(sample_path, "rb") as f:
        observation = pickle.load(f)

    report = {}
    all_ok = True
    for suite, ckpt in SUITES.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = GenerateConfig(
            pretrained_checkpoint=ckpt,
            task_suite_name=suite,
            unnorm_key=suite,
        )
        t0 = time.time()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        load_s = time.time() - t0

        t0 = time.time()
        actions = get_vla_action(
            cfg, model, processor, observation,
            observation["task_description"], action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
        )
        fwd_s = time.time() - t0

        import numpy as np
        arr = np.asarray(actions, dtype=np.float64)
        finite = bool(np.isfinite(arr).all())
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        entry = {
            "checkpoint": ckpt,
            "load_seconds": round(load_s, 1),
            "first_forward_seconds": round(fwd_s, 3),
            "action_chunk_shape": list(arr.shape),
            "actions_finite": finite,
            "vram_peak_gb": round(vram_gb, 2),
            "dtype": str(next(model.parameters()).dtype),
            "attn": "sdpa (flash-attn 미빌드, sm_120)",
        }
        ok = finite and arr.shape[0] == 8 and arr.shape[-1] == 7
        entry["ok"] = bool(ok)
        all_ok &= ok
        report[suite] = entry
        print(json.dumps(entry, indent=2), flush=True)

        del model, action_head, proprio_projector, noisy_action_projector, processor
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    status = "PASS" if all_ok else "FAIL"
    print(f"[E0-PASS] item=E0-4 status={status} json=results/e0/e0_4_ckpt.json")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
