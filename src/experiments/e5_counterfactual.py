"""E5 counterfactual completion 배치 (설계서 v0.3 §3, §4h 기준선).

발화(habit) 에피소드의 **동일 spec**을 teacher가 수행했다면? — 스펙 단위 paired 기준선.
스트림 종료 후 배치 실행(스트림 중 실행은 H4 측정 대상 오염 — 결정 2).
선행 관문: **결정성 사전 검증** — 표본 spec을 teacher로 2회 재실행해 결과 일치 확인,
불일치 시 즉시 정지·보고(§4h 명시).

시간 회계: 본 산출은 **평가 장부**(측정 아티팩트) — 지연 주장에 불산입(§4h 3장부).
산출: results/e5/cf_{seed}.jsonl + cf_summary_{seed}.json
마커: [E5CF-DET-PASS|FAIL] / [E5CF-EP] / [E5CF-DONE]
실행: hv2_oft python -u experiments/e5_counterfactual.py --seed-idx 0
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

from envs.chained_env import execute_chunk_with_boundary  # noqa: E402
from envs.libero_env import EpisodeSpec, InfraError, LiberoEpisodeEnv  # noqa: E402

CHUNK = 8
DET_SAMPLE = 5  # 결정성 사전 검증 표본 (설계서 §3: 5–10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    # RGB-only full rerun (2026-08-28): 순수 가산. 기본값은 기존 동작 그대로.
    ap.add_argument("--queue-root", default=None,
                    help="cf_queue_{seed}.jsonl이 있는 디렉터리 (기본 results/e5)")
    ap.add_argument("--out-root", default=None, help="산출 디렉터리 (기본 results/e5)")
    args = ap.parse_args()
    outdir = args.out_root or os.path.join(HABIT2, "results", "e5")
    qdir = args.queue_root or outdir
    os.makedirs(outdir, exist_ok=True)
    qpath = os.path.join(qdir, f"cf_queue_{args.seed_idx}.jsonl")
    opath = os.path.join(outdir, f"cf_{args.seed_idx}.jsonl")
    if os.path.exists(opath):
        raise SystemExit(f"[E5CF-FAIL] 출력 경로 존재 (덮어쓰기 금지): {opath}")
    queue = [json.loads(l) for l in open(qpath)]
    print(f"[E5CF] 큐 {len(queue)}건 (발화 에피소드 spec)", flush=True)

    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import get_vla_action
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    from teacher.collector import load_teacher, teacher_observation

    set_seed_everywhere(7)
    teachers, resize, envs = {}, {}, {}

    def teacher_of(suite):
        if suite not in teachers:
            for s in list(teachers):
                del teachers[s]
            import torch

            torch.cuda.empty_cache()
            teachers[suite] = load_teacher(suite)
            resize[suite] = get_image_resize_size(teachers[suite][0])
        return teachers[suite], resize[suite]

    def env_of(suite, task):
        k = (suite, task)
        if k not in envs:
            if len(envs) > 4:
                envs.popitem()[1].close()
            envs[k] = LiberoEpisodeEnv(suite, task)
        return envs[k]

    def run(spec_d):
        env = env_of(spec_d["suite"], spec_d["task_id"])
        spec = EpisodeSpec(spec_d["suite"], spec_d["task_id"], spec_d["seed"],
                           spec_d["base_idx"], spec_d["w"], spec_d["noise_seed"])
        obs = spec.realize(env)
        (cfg, model, ah, pp, nap, proc), rs = teacher_of(spec_d["suite"])
        t, success = 0, False
        while t < env.max_steps:
            t_obs = teacher_observation(obs, rs)
            chunk = get_vla_action(cfg, model, proc, t_obs, env.language, action_head=ah,
                                   proprio_projector=pp, noisy_action_projector=nap)
            acts = [process_action(np.asarray(a, np.float32), cfg.model_family)
                    for a in chunk[:CHUNK]]
            obs, t, _n, _s = execute_chunk_with_boundary(env, acts, t, env.max_steps)
            if env.check_success():
                success = True
                break
        return success, t

    # --- 선행 관문: 결정성 사전 검증 (2회 재실행 일치)
    det = []
    for d in queue[:DET_SAMPLE]:
        a, _ = run(d)
        b, _ = run(d)
        det.append({"uid": d["uid"], "run1": bool(a), "run2": bool(b), "match": a == b})
        print(f"[E5CF-DET] {d['uid']} {a} vs {b} {'OK' if a==b else '★불일치'}", flush=True)
    if not all(x["match"] for x in det):
        with open(os.path.join(outdir, f"cf_determinism_{args.seed_idx}.json"), "w") as f:
            json.dump({"status": "FAIL", "checks": det}, f, indent=2, ensure_ascii=False)
        raise SystemExit("[E5CF-DET-FAIL] 결정성 불일치 — 정지·보고 (§4h)")
    print(f"[E5CF-DET-PASS] {len(det)}/{len(det)} 일치", flush=True)

    # --- 본 배치
    t0 = time.time()
    n_match = 0
    with open(opath, "w") as f:
        for i, d in enumerate(queue):
            try:
                succ, steps = run(d)
            except InfraError as e:
                f.write(json.dumps({**d, "teacher_outcome": "infra_error", "error": str(e)},
                                   ensure_ascii=False) + "\n")
                continue
            n_match += int(succ == d["habit_success"])
            f.write(json.dumps({"uid": d["uid"], "cluster": d["cluster"],
                                "habit_success": d["habit_success"],
                                "teacher_success": bool(succ), "teacher_steps": steps},
                               ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[E5CF-EP] {i+1}/{len(queue)} 일치율 {n_match/(i+1):.3f}", flush=True)

    rows = [json.loads(l) for l in open(opath) if '"teacher_success"' in l]
    hs = sum(1 for r in rows if r["habit_success"])
    ts = sum(1 for r in rows if r["teacher_success"])
    b = sum(1 for r in rows if r["habit_success"] and not r["teacher_success"])   # 습관만 성공
    c = sum(1 for r in rows if not r["habit_success"] and r["teacher_success"])   # teacher만 성공
    summary = {"seed_idx": args.seed_idx, "n_paired": len(rows),
               "habit_success_rate": round(hs / len(rows), 4) if rows else None,
               "teacher_success_rate": round(ts / len(rows), 4) if rows else None,
               "discordant_habit_only": b, "discordant_teacher_only": c,
               "determinism_check": det, "ledger": "evaluation",
               "wall_s": round(time.time() - t0, 1),
               "note": "paired 기준선 — 발화 spec만. 비발화는 VLA 실측이 곧 기준선(설계서 §3)"}
    if b + c > 0:
        from scipy.stats import binomtest

        summary["mcnemar_exact_p"] = round(float(binomtest(b, b + c, 0.5).pvalue), 4)
    with open(os.path.join(outdir, f"cf_summary_{args.seed_idx}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for e in envs.values():
        e.close()
    print(f"[E5CF-DONE] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
