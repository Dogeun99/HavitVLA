"""E4-1: known 프레임 확보 — held-out 스펙 재생 렌더 패스 (연구원 지시 2026-08-16 §2).

- known = **held-out만** (수집 프레임은 μ/Σ fit에 쓰여 in-sample — 사용 금지, 검증 발견).
  관할 모델의 fit/calib 분할은 수집 프레임 유지 (§4c — 본 패스와 무관).
- 25 표준 클러스터 × held-out 20 스펙: E0-6 재현 프로토콜로 realize → I₀를
  **게이트 입력 규격**(prep_gate_rgb: 180° 회전 + 128)으로 저장.
- 산출: results/e4/known_frames/{cluster}.npz (frames u8 (20,128,128,3), uids json)
- 진행: [DUMP] <cluster> j/20 — heartbeat 파싱용. 종료 마커 [E4KNOWN-PASS|FAIL].

실행: hv2_hab python -u experiments/e4_known_frames.py
"""
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.libero_env import LiberoEpisodeEnv  # noqa: E402
from envs.stream import heldout_specs  # noqa: E402
from gates.features import prep_gate_rgb  # noqa: E402

OUT_DIR = os.path.join(HABIT2, "results", "e4", "known_frames")
STANDARD = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)
N_HELDOUT = 20


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for k, (suite, task) in enumerate(STANDARD):
        cl = f"{suite}_task{task}"
        out_p = os.path.join(OUT_DIR, f"{cl}.npz")
        if os.path.exists(out_p):
            print(f"[DUMP-SKIP] {cl}: 기존 산출물 존재 (경로 유일성 — 덮어쓰기 금지)", flush=True)
            continue
        env = LiberoEpisodeEnv(suite, task)
        frames, uids = [], []
        for j, spec in enumerate(heldout_specs(suite, task, N_HELDOUT)):
            obs = spec.realize(env)
            frames.append(prep_gate_rgb(obs["agentview_image"]))
            uids.append(spec.uid)
            print(f"[DUMP] {cl} {j + 1}/{N_HELDOUT}", flush=True)
        np.savez_compressed(out_p, frames=np.stack(frames).astype(np.uint8),
                            uids=json.dumps(uids))
        env.close()
        print(f"[DUMP-DONE] {cl} ({k + 1}/{len(STANDARD)})", flush=True)
    print("[E4KNOWN-PASS]")


if __name__ == "__main__":
    main()
