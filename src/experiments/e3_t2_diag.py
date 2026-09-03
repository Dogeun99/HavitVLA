"""T2-SMOKE-FAIL 진단 (0/10, 전원 stage 2 도달 후 실패) — 원인 분리 실험.

가설 분리:
  A. 재배치 후 "상태 자체"가 teacher에게 해결 불가 (재배치 상태 결함)
  B. 상태는 해결 가능하나 "로봇 포즈 연속"이 문제 — teacher는 홈 포즈 시작 분포로
     학습됐고, stage 1 종료 직후(바스켓 위 팔 뻗은 포즈)에서 재시작을 못 함 (OOD)

실험 1 (본 재현 + 계기): chained 에피소드 1개 — stage 2 진입 시점과 이후의
  EE 위치·행동 크기·predicate·물체 위치를 로깅, 프레임 PNG 저장 (시각 검수용).
실험 2 (통제): 동일한 재배치 상태(물체 배치 동일)를 "로봇 홈 포즈"에서 시작하는
  일반 에피소드로 구성해 teacher 실행 → 성공하면 가설 B 확정.

실행: cd $HABIT2 && conda/hv2_oft python -u experiments/e3_t2_diag.py --task 0
출력: results/e3/t2_diag.json + 프레임 PNG (data/e3/t2_smoke/diag_frames/)
"""
import argparse
import json
import os
import sys

import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
sys.path.insert(0, os.path.join(HABIT2, "third_party", "openvla-oft"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(HABIT2, ".libero"))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.chained_env import ChainedEpisodeEnv, chained_collection_specs  # noqa: E402
from envs.libero_env import LiberoEpisodeEnv  # noqa: E402
from teacher.collector import load_teacher, teacher_observation  # noqa: E402

CHUNK = 8
FRAME_DIR = os.path.join(HABIT2, "data", "e3", "t2_smoke", "diag_frames")


def save_frame(obs, name):
    import cv2

    img = obs["agentview_image"][::-1, ::-1]
    cv2.imwrite(os.path.join(FRAME_DIR, f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def teacher_chunk(teacher, resize_size, obs, language):
    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import get_vla_action

    cfg, model, action_head, proprio_projector, noisy_action_projector, processor = teacher
    t_obs = teacher_observation(obs, resize_size)
    chunk = get_vla_action(
        cfg, model, processor, t_obs, language,
        action_head=action_head, proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
    )
    return [process_action(np.asarray(a, dtype=np.float32), cfg.model_family) for a in chunk[:CHUNK]]


def run_instrumented_chained(spec, env, teacher, resize_size, log):
    obs = spec.realize(env)
    save_frame(obs, "exp1_t000_start")
    t, success = 0, False
    stage2_trace = []
    prev_stage = 1
    while t < env.max_steps:
        actions = teacher_chunk(teacher, resize_size, obs, env.language)
        for a in actions:
            obs, _, done, _ = env.step(a.tolist())
            t += 1
            if done or t >= env.max_steps:
                break
        if prev_stage == 1 and env.stage() == 2:
            log["stage2_entry_step"] = t
            log["predicate_after_relocate"] = bool(env._env.check_success())
            save_frame(obs, f"exp1_t{t:03d}_stage2_entry")
            prev_stage = 2
        if env.stage() == 2:
            ee = obs.get("robot0_eef_pos")
            stage2_trace.append({
                "t": t,
                "ee": [round(float(x), 3) for x in ee] if ee is not None else None,
                "act_absmean": round(float(np.mean(np.abs(np.stack(actions)[:, :6]))), 4),
                "gripper_cmd": float(actions[-1][6]),
            })
            if len(stage2_trace) % 6 == 1:
                save_frame(obs, f"exp1_t{t:03d}_stage2")
        if env.check_success():
            success = True
            break
    save_frame(obs, f"exp1_t{t:03d}_end")
    log["exp1_success"] = success
    log["exp1_total_steps"] = t
    log["exp1_stage_steps"] = env.stage_steps
    log["exp1_stage2_trace"] = stage2_trace[:6] + stage2_trace[-6:] if len(stage2_trace) > 12 else stage2_trace
    return success


def run_control_from_home(spec, teacher, resize_size, log):
    """실험 2: 재배치 상태와 동일한 물체 배치 + 로봇 홈 포즈 (일반 에피소드 프로토콜)."""
    env = LiberoEpisodeEnv(spec.suite_name, spec.task_id)
    # chained_env.begin_chained_episode와 동일한 재배치 상태 구성 (동일 rng·동일 폭) —
    # perturbed_init_state는 EpisodeSpec.realize가 쓰는 공개 경로 (모델 상수 자체 초기화)
    rng = np.random.default_rng(spec.relocate_noise_seed)
    from envs.libero_env import USABLE_W_MAX

    w = min(spec.w, USABLE_W_MAX[spec.suite_name])
    s = env.perturbed_init_state(spec.relocate_base_idx, w, rng)
    obs = env.begin_episode(spec.seed, s)
    save_frame(obs, "exp2_t000_control_start")
    t, success = 0, False
    while t < env.max_steps:
        actions = teacher_chunk(teacher, resize_size, obs, env.language)
        for a in actions:
            obs, _, done, _ = env.step(a.tolist())
            t += 1
            if done or t >= env.max_steps:
                break
        if env.check_success():
            success = True
            break
    save_frame(obs, f"exp2_t{t:03d}_control_end")
    log["exp2_control_success"] = success
    log["exp2_control_steps"] = t
    env.close()
    return success


class HomeResetChainedEnv(ChainedEpisodeEnv):
    """진단 전용 probe (Option 2): stage 전환 = 물체 재배치 + **로봇 상태도 재배치 init의
    홈 포즈로 재설정** (set_init_state 전체 벡터 — begin_episode의 상태 의미론과 동일).
    파이프라인 채택 여부는 연구원 결정 사항 — 본 클래스는 e3_t2_diag 전용."""

    def _relocate_objects(self):
        obs = self._env.set_init_state(self._relocate_state)
        from envs.libero_env import DUMMY_ACTION, SETTLE_STEPS

        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self._env.step(DUMMY_ACTION)
            self._t += 1
        return obs


def run_probe_fix(task, n_eps, teacher, resize_size, log):
    """Option 2 probe: 홈 재설정 전환으로 n_eps 연쇄 에피소드 — 성공률 복원 확인."""
    specs = chained_collection_specs("libero_object", task, n_eps)
    results = []
    for i, spec in enumerate(specs):
        env = HomeResetChainedEnv("libero_object", task)
        obs = spec.realize(env)
        t, success = 0, False
        while t < env.max_steps:
            actions = teacher_chunk(teacher, resize_size, obs, env.language)
            for a in actions:
                obs, _, done, _ = env.step(a.tolist())
                t += 1
                if done or t >= env.max_steps:
                    break
            if env.check_success():
                success = True
                break
        results.append({"uid": spec.uid, "success": success, "stage_steps": env.stage_steps})
        print(f"[probe {i+1}/{n_eps}] success={success} stage_steps={env.stage_steps}", flush=True)
        env.close()
    log["probe_fix_option2"] = results
    log["probe_fix_success"] = sum(1 for r in results if r["success"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--spec-idx", type=int, default=0)
    ap.add_argument("--probe-fix", type=int, default=0,
                    help="Option 2(홈 재설정 전환) probe 에피소드 수 — 0이면 본 진단만")
    args = ap.parse_args()

    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    os.makedirs(FRAME_DIR, exist_ok=True)
    set_seed_everywhere(7)
    teacher = load_teacher("libero_object")
    resize_size = get_image_resize_size(teacher[0])

    spec = chained_collection_specs("libero_object", args.task, args.spec_idx + 1)[args.spec_idx]
    log = {"task": args.task, "spec_uid": spec.uid}

    if args.probe_fix > 0:
        run_probe_fix(args.task, args.probe_fix, teacher, resize_size, log)
        # R2 (검토 반영): 출력 경로에 task suffix 의무화 — 덮어쓰기 금지 렌즈
        out = os.path.join(HABIT2, "results", "e3", f"t2_diag_probe_task{args.task}.json")
        with open(out, "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"[T2-DIAG-PROBE] {log['probe_fix_success']}/{args.probe_fix} json={out}")
        return

    env = ChainedEpisodeEnv("libero_object", args.task)
    run_instrumented_chained(spec, env, teacher, resize_size, log)
    env.close()

    run_control_from_home(spec, teacher, resize_size, log)

    verdict = (
        "B: 로봇 포즈 OOD (상태는 홈 포즈에서 해결 가능)"
        if (not log["exp1_success"]) and log["exp2_control_success"]
        else "A: 재배치 상태 자체 결함" if not log["exp2_control_success"]
        else "재현 실패 — exp1 성공"
    )
    log["verdict"] = verdict
    # R2 (검토 반영): 출력 경로에 task suffix 의무화 — 덮어쓰기 금지 렌즈
    # (구 t2_diag.json은 task0 v1 실행이 task5 실행에 덮인 사고 — v1 원본은
    #  t2_diag_task0_v1.json으로 git 791977f에서 복구)
    out = os.path.join(HABIT2, "results", "e3", f"t2_diag_task{args.task}.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"[T2-DIAG] verdict={verdict} json={out} frames={FRAME_DIR}")


if __name__ == "__main__":
    main()
