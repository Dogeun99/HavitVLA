"""E1-b: 레이턴시 앵커 측정 (설계서 §5 E1 — 5종 중 ①②③④, ⑤는 E2 후).

  ① OFT chunk forward (주 시스템 비용 단위 — "VLA-호출 등가"의 분모)
  ② ACT forward (무학습 모델 — 아키텍처 고정이므로 레이턴시는 가중치 무관)
  ③ DINOv2 embed + PCA + 2단 gate 판정 (주 gate 경로 총비용)
  ④ 히든 스테이트 추출 전용 (비교 arm의 비용 앵커)
  ⑤ ACT 학습 1회(n=40): E2 데이터 필요 → pending 기록, E2 직후 측정

방법: warmup 10회 후 100회 중앙값/평균/p95. torch.cuda.synchronize로 경계 고정.
전부 attn=sdpa (flash-attn 미빌드, sm_120) — 논문 각주 필수.
실행: cd $HABIT2/third_party/openvla-oft && conda run -n hv2_oft python -u $HABIT2/experiments/e1_latency.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative (was hardcoded)
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))

OUT = os.path.join(HABIT2, "results", "e1", "e1_latency.json")
N_WARM, N_MEAS = 10, 100


def timed(fn, n_warm=N_WARM, n_meas=N_MEAS):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_meas):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return {
        "median_ms": round(float(np.median(ts)), 2),
        "mean_ms": round(float(ts.mean()), 2),
        "p95_ms": round(float(np.percentile(ts, 95)), 2),
        "n": n_meas,
    }


def main():
    report = {"attn": "sdpa (flash-attn 미빌드, sm_120)", "gpu": torch.cuda.get_device_name()}

    # --- ① OFT chunk forward + ④ 히든 추출 ---
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.openvla_utils import get_vla_action

    cfg = GenerateConfig(
        pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
        task_suite_name="libero_spatial",
        unnorm_key="libero_spatial",
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    with open(
        os.path.join(HABIT2, "third_party", "openvla-oft", "experiments", "robot", "libero",
                     "sample_libero_spatial_observation.pkl"), "rb") as f:
        obs = pickle.load(f)
    task = obs["task_description"]

    report["anchor1_oft_chunk_forward"] = timed(
        lambda: get_vla_action(
            cfg, model, processor, obs, task, action_head=action_head,
            proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
        )
    )

    # ④ 히든 스테이트 추출: OFT predict_action이 이미 output_hidden_states=True로 forward하고
    # hidden_states[-1]을 사용(modeling_prismatic.py:910-916) — L32 히든은 VLA forward의 부산물.
    # 따라서 비교 arm의 "히든 추출 전용" 비용 = anchor1과 동일 등급 (별도 증분 ≈ 0).
    report["anchor4_hidden_extract"] = {
        "same_cost_as_anchor1": True,
        "evidence": "modeling_prismatic.py predict_action: output_hidden_states=True, hidden_states[-1] 사용",
        "implication": "히든 gate arm은 VLA forward 비용을 그대로 지불 — 주 gate 경로(anchor3) 대비 비용 축 비교의 앵커",
    }

    del model, action_head, proprio_projector, noisy_action_projector, processor
    torch.cuda.empty_cache()

    # --- ② ACT forward ---
    from habits.act import ACTPolicy

    act = ACTPolicy(pretrained=False).cuda().eval()
    ai = [torch.randn(1, 4, 128, 128, device="cuda"), torch.randn(1, 4, 128, 128, device="cuda")]
    ap = torch.randn(1, 8, device="cuda")
    report["anchor2_act_forward"] = timed(lambda: act.act(ai, ap))
    del act
    torch.cuda.empty_cache()

    # --- ③ DINOv2 + PCA + 2단 gate ---
    from gates.features import DinoFeatureExtractor, SharedPCA, prep_gate_rgb
    from gates.two_stage import JurisdictionGate, MaturityGate

    dino = DinoFeatureExtractor()
    rng = np.random.default_rng(0)
    fake_rgb = prep_gate_rgb(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8))
    pca = SharedPCA().fit(rng.normal(size=(200, 384)))
    jur = JurisdictionGate().fit(rng.normal(size=(40, 32)))
    mat = MaturityGate()
    for _ in range(30):
        mat.update(True, source="fire")

    def gate_path():
        e = dino.embed([fake_rgb])
        z = pca.transform(e)
        _ = jur.accepts(z[0]) and mat.accepts()

    report["anchor3_gate_path"] = timed(gate_path)

    report["anchor5_act_train_n40"] = {"status": "pending - E2 데이터 수집 직후 측정"}
    r1 = report["anchor1_oft_chunk_forward"]["median_ms"]
    r2 = report["anchor2_act_forward"]["median_ms"]
    r3 = report["anchor3_gate_path"]["median_ms"]
    report["ratios"] = {
        "act_over_oft": round(r2 / r1, 4),
        "gate_over_oft": round(r3 / r1, 4),
        "note": "VLA-호출 등가 환산의 분모 = anchor1 (per-chunk, 8 steps)",
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[E1-PASS] item=E1-LAT status=PASS json=results/e1/e1_latency.json")


if __name__ == "__main__":
    main()
