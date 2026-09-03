"""T2 task5 진단 2차 — 재배치 상태 결함(가설 A)의 기전 분리.

통제 프레임 관찰: 종료 시 바스켓 안에 대상 추정 물체가 있는데 check_success=False.
→ 조건 4종으로 분리 + 물체·바스켓 좌표와 predicate를 동시 로깅:
  a) base0 + 수집대역 noise (sanity — 정상 에피소드, 성공 기대)
  b) base17 + 수집대역 noise (base 자체가 어려운가?)
  c) base17 + w=0 (섭동 없음 — 섭동 요인 제거)
  d) base17 + relocate대역 noise 500000 (스모크 재현 조건)
각: 최대 280 스텝, 매 chunk마다 (EE, tomato_sauce xyz, basket xyz, check_success),
    시작·종료 프레임 저장. 종료 시 전 free 물체의 xy가 바스켓 중심 ±10cm 내인지 검사
    (predicate와 기하의 불일치 검출).

실행: hv2_oft python -u experiments/e3_t2_diag2.py
"""
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

from envs.libero_env import LiberoEpisodeEnv  # noqa: E402
from teacher.collector import load_teacher, teacher_observation  # noqa: E402
from experiments.e3_t2_diag import teacher_chunk, save_frame, FRAME_DIR  # noqa: E402

TASK = 5
MAXS = 280


def body_pos(env, keyword):
    sim = env._env.env.sim
    for name in sim.model.body_names:
        if keyword in name:
            return name, np.array(sim.data.get_body_xpos(name))
    return None, None


def run_case(tag, env, state, seed, teacher, resize_size, out):
    obs = env.begin_episode(seed, state)
    save_frame(obs, f"d2_{tag}_start")
    t, success = 0, False
    trace = []
    while t < MAXS:
        actions = teacher_chunk(teacher, resize_size, obs, env.language)
        for a in actions:
            obs, _, done, _ = env.step(a.tolist())
            t += 1
            if done or t >= MAXS:
                break
        tn, tp = body_pos(env, "tomato_sauce")
        bn, bp = body_pos(env, "basket")
        trace.append({"t": t, "succ": bool(env.check_success()),
                      "tomato": None if tp is None else [round(float(x), 3) for x in tp],
                      "basket": None if bp is None else [round(float(x), 3) for x in bp]})
        if env.check_success():
            success = True
            break
    save_frame(obs, f"d2_{tag}_end")
    tn, tp = body_pos(env, "tomato_sauce")
    bn, bp = body_pos(env, "basket")
    geo_in = (tp is not None and bp is not None
              and abs(tp[0] - bp[0]) < 0.10 and abs(tp[1] - bp[1]) < 0.10)
    out[tag] = {
        "success": success, "steps": t,
        "tomato_body": tn, "basket_body": bn,
        "final_tomato": None if tp is None else [round(float(x), 3) for x in tp],
        "final_basket": None if bp is None else [round(float(x), 3) for x in bp],
        "tomato_geometrically_in_basket_xy": bool(geo_in),
        "predicate_final": bool(env.check_success()),
        "trace_tail": trace[-4:],
    }
    print(f"[{tag}] success={success} steps={t} geo_in={geo_in} "
          f"tomato={out[tag]['final_tomato']} basket={out[tag]['final_basket']}", flush=True)


def main():
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    os.makedirs(FRAME_DIR, exist_ok=True)
    set_seed_everywhere(7)
    teacher = load_teacher("libero_object")
    resize_size = get_image_resize_size(teacher[0])

    out = {}
    env = LiberoEpisodeEnv("libero_object", TASK)
    # 상태 구성 재료 준비 (perturbed_init_state가 모델 상수 자체 초기화)
    cases = [
        ("a_base0_collectnoise", 0, 0, 0.01),
        ("b_base17_collectnoise", 17, 17, 0.01),
        ("c_base17_w0", 17, None, 0.0),
        ("d_base17_relocnoise", 17, 500_000, 0.01),
    ]
    for tag, base, noise, w in cases:
        if w > 0:
            rng = np.random.default_rng(noise)
            state = env.perturbed_init_state(base, w, rng)
        else:
            state = env.init_states[base]
        run_case(tag, env, state, 10_000, teacher, resize_size, out)

    p = os.path.join(HABIT2, "results", "e3", "t2_diag2.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[T2-DIAG2] json={p}")


if __name__ == "__main__":
    main()
