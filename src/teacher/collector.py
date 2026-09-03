"""Teacher(OpenVLA-OFT) 궤적 수집기 — 클러스터별 HDF5 저장 + 이중 장부.

설계 근거:
  - 공식 로드·전처리·행동 경로(initialize_model, get_vla_action, prepare_observation과 동일
    변환: 180° 회전, resize, proprio 구성)를 그대로 사용해 E0-5/E1과 동일한 teacher 동작 보장.
  - 에피소드 재현: envs.libero_env의 EpisodeSpec + begin_episode 프로토콜(E0-6).
  - **이중 장부 (설계서 §2.5):** 성공/실패 라벨과 에피소드 메타는 전부 기록하되,
    BC 학습 풀은 success=True 에피소드만. 실패 궤적의 프레임은 저장하지 않음(용량 절약) —
    통계(𝒟_k)는 메타로 충분.
  - **인프라 오류는 raise** — 정책 실패로 위장 금지 (InfraError 전파, 에피소드 무효 처리 후 기록).
  - 저장 해상도: RGB-D 128×128 (uint8 RGB + float16 depth), **requery 경계 프레임만**(8-step당 1장).
    teacher 입력은 256 렌더 원본을 공식 전처리로 사용, 저장본만 다운샘플 —
    클러스터당 ~0.4–0.7GB (28 클러스터 ~20GB; 검증 워크플로우 실측 산정).
  - 증분 저장: 성공 에피소드마다 즉시 HDF5 기록 + 메타는 finally에서 보존 (장애 시 소실 방지).

실행 예 (E2 수집):
  cd $HABIT2/third_party/openvla-oft && conda run -n hv2_oft python -u \
    $HABIT2/teacher/collector.py --suite libero_object --task 0 --n 120 --out $HABIT2/data/e2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.libero_env import InfraError, LiberoEpisodeEnv, proprio_vector  # noqa: E402
from envs.stream import collection_specs  # noqa: E402

SUITE_TO_CKPT = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}
STORE_RES = 128
CHUNK = 8  # OFT action chunk (num_open_loop_steps) — 공식 기본값과 동일


def load_teacher(suite_name):
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model

    cfg = GenerateConfig(
        pretrained_checkpoint=SUITE_TO_CKPT[suite_name],
        task_suite_name=suite_name,
        unnorm_key=suite_name,
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    return cfg, model, action_head, proprio_projector, noisy_action_projector, processor


def teacher_observation(obs, resize_size):
    """공식 prepare_observation과 동일 (run_libero_eval.py:243)."""
    from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image
    from experiments.robot.robot_utils import get_image_resize_size  # noqa: F401
    from experiments.robot.openvla_utils import resize_image_for_policy

    img = get_libero_image(obs)
    wrist = get_libero_wrist_image(obs)
    observation = {
        "full_image": resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist, resize_size),
        "state": proprio_vector(obs),
    }
    return observation


def store_frame(obs):
    """저장용 다운샘플 프레임: RGB uint8 + depth f16, 두 카메라. 방향은 teacher 학습 전처리와
    동일하게 180° 회전본을 저장한다(사용처 일관성)."""
    import cv2

    def prep_rgb(img):
        img = img[::-1, ::-1]
        return cv2.resize(img, (STORE_RES, STORE_RES), interpolation=cv2.INTER_AREA)

    def prep_depth(d):
        d = d[::-1, ::-1, 0]
        return cv2.resize(d, (STORE_RES, STORE_RES), interpolation=cv2.INTER_AREA).astype(np.float16)

    return {
        "agentview_rgb": prep_rgb(obs["agentview_image"]),
        "wrist_rgb": prep_rgb(obs["robot0_eye_in_hand_image"]),
        "agentview_depth": prep_depth(obs["agentview_depth"]),
        "wrist_depth": prep_depth(obs["robot0_eye_in_hand_depth"]),
        "proprio": proprio_vector(obs),
    }


def rollout_episode(spec, env, teacher, resize_size):
    """1 에피소드 rollout. 반환: (success, frames|None, actions|None, n_steps).
    frames는 requery 시점(8-step 경계)별 관측, actions는 실행된 per-step 행동 전체.

    ★ 공식 경로와 동일: get_vla_action → **process_action**(gripper [0,1]→{-1,+1} 이진화 +
    부호 반전, run_libero_eval.py:347) → env.step. 저장도 process 후 값 — BC 타깃이
    env 실행 공간과 일치해야 habit 실행(evaluate.py의 raw env.step)이 옳다.
    (검증 워크플로우 치명 발견 반영 — 누락 시 gripper 명령 공간 전체가 틀어짐.)

    모델측 예외(get_vla_action, TF resize)는 InfraError로 승격 — 정책 실패로 위장 금지.
    """
    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import get_vla_action

    from envs.chained_env import execute_chunk_with_boundary

    cfg, model, action_head, proprio_projector, noisy_action_projector, processor = teacher
    obs = spec.realize(env)
    frames, actions = [], []
    t, success, stale_discarded = 0, False, None
    while t < env.max_steps:
        try:
            t_obs = teacher_observation(obs, resize_size)
            frames.append(store_frame(obs))
            chunk = get_vla_action(
                cfg, model, processor, t_obs, env.language,
                action_head=action_head, proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
            )
        except InfraError:
            raise
        except Exception as e:
            raise InfraError(f"teacher inference failed: {type(e).__name__}: {e}") from e
        processed = [process_action(np.asarray(a, dtype=np.float32), cfg.model_family)
                     for a in chunk[:CHUNK]]
        # §4e 개정 α: 전환 감지 시 잔여 stale 행동 폐기 + 즉시 재질의 (diag5b 확증)
        obs, t, n_exec, stale = execute_chunk_with_boundary(env, processed, t, env.max_steps)
        actions.append(np.stack([np.asarray(a, dtype=np.float32) for a in processed[:n_exec]]))
        if stale is not None:
            stale_discarded = stale
        if env.check_success():
            success = True
            break
    return success, frames, actions, t, stale_discarded


def write_episode(f, uid, frames, actions):
    """성공 에피소드 1개를 즉시 기록 (증분 저장 — 장애 시 기수집분 보존)."""
    g = f.create_group(f"episodes/{uid}")
    for key in ("agentview_rgb", "wrist_rgb", "agentview_depth", "wrist_depth", "proprio"):
        g.create_dataset(
            key,
            data=np.stack([fr[key] for fr in frames]),
            compression="gzip",
            compression_opts=2,
        )
    flat = np.concatenate(actions)
    lens = np.array([len(a) for a in actions], dtype=np.int32)
    g.create_dataset("actions_flat", data=flat)
    g.create_dataset("chunk_lens", data=lens)
    f.flush()


def write_meta(f, episodes_meta):
    """메타는 attr 크기 한계를 피해 dataset으로 저장 (기존 있으면 교체)."""
    if "meta_json" in f:
        del f["meta_json"]
    f.create_dataset("meta_json", data=json.dumps(episodes_meta, ensure_ascii=False))
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chained", action="store_true",
                    help="C-T2 2연쇄 래퍼 (preregistration §4e) — 클러스터명 chained_*")
    args = ap.parse_args()

    import h5py

    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    set_seed_everywhere(7)  # 공식 eval 순서와 동일: initialize_model 이전 (run_libero_eval.py:468-471)
    teacher = load_teacher(args.suite)
    resize_size = get_image_resize_size(teacher[0])

    if args.chained:
        from envs.chained_env import ChainedEpisodeEnv, chained_collection_specs

        env = ChainedEpisodeEnv(args.suite, args.task)
        specs = chained_collection_specs(args.suite, args.task, args.n)
        cluster = f"chained_{args.suite}_task{args.task}"
    else:
        env = LiberoEpisodeEnv(args.suite, args.task)
        specs = collection_specs(args.suite, args.task, args.n)
        cluster = f"{args.suite}_task{args.task}"

    os.makedirs(args.out, exist_ok=True)
    out_h5 = os.path.join(args.out, f"{cluster}.hdf5")
    episodes_meta = []
    n_succ = n_fail = n_infra = 0
    t0 = time.time()
    completed = False
    with h5py.File(out_h5, "w") as f:
        f.attrs["schema"] = "habitvla2-teacher-v2"  # v2: process_action 적용, 증분 저장
        try:
            for i, spec in enumerate(specs):
                try:
                    success, frames, actions, steps, stale = rollout_episode(spec, env, teacher, resize_size)
                except InfraError as e:
                    # 인프라 오류: 라벨 오염 금지 — 별도 계정, 통계·학습 모두 제외
                    episodes_meta.append({**spec.to_dict(), "outcome": "infra_error", "error": str(e)})
                    n_infra += 1
                    print(f"[{i+1}/{len(specs)}] INFRA_ERROR {e}", flush=True)
                    continue
                outcome = "success" if success else "fail"
                entry = {**spec.to_dict(), "outcome": outcome, "steps": steps}
                if args.chained:
                    # 검증 ③·§6 분석용: stage 도달(2 = stage 1 성공·재배치 수행) + stage별 스텝
                    entry["stage"] = env.stage()
                    entry["stage_steps"] = env.stage_steps
                    entry["stale_discarded"] = stale  # §4e 개정 α — 전환 시 폐기된 stale 수 로깅
                episodes_meta.append(entry)
                if success:
                    write_episode(f, spec.uid, frames, actions)  # 이중 장부: 성공만 BC 풀 — 즉시 기록
                    n_succ += 1
                else:
                    n_fail += 1
                if (i + 1) % 10 == 0:
                    el = time.time() - t0
                    print(
                        f"[{i+1}/{len(specs)}] succ={n_succ} fail={n_fail} infra={n_infra} ({el:.0f}s)",
                        flush=True,
                    )
            completed = True
        finally:
            write_meta(f, episodes_meta)  # 장애 시에도 기수집분 메타 보존

    summary = {
        "cluster": cluster,
        "n_requested": args.n,
        "n_success": n_succ,
        "n_fail": n_fail,
        "n_infra_error": n_infra,
        "S_V_cluster": round(n_succ / max(n_succ + n_fail, 1), 4),
        "hdf5": out_h5,
        "partial": not completed,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, f"{cluster}_summary.json"), "w") as fj:
        json.dump(summary, fj, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    marker = "COLLECT-PASS" if completed else "COLLECT-PARTIAL"
    print(f"[{marker}] cluster={cluster} succ={n_succ}/{max(n_succ+n_fail,1)} json={out_h5}")
    env.close()


if __name__ == "__main__":
    main()
