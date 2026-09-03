"""held-out 성숙 곡선 평가 — 체크포인트 간 동일 EpisodeSpec (paired 비교, 설계서 §4.2).

habit 실행 = K-step open-loop 후 requery (teacher와 동일 제어 패턴).
성공 판정 = LIBERO 공식 predicate. 인프라 오류는 별도 계정(성공/실패 어느 쪽도 오염 금지).

실행 (hv2_hab — ACT env):
  conda run -n hv2_hab python -u habits/evaluate.py \
    --cluster libero_object_task0 --suite libero_object --task 0 \
    --ckpt-dir checkpoints/libero_object_task0 --n-heldout 50 --out results/e2
"""
import argparse
import json
import os
import sys
import time

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)

from envs.libero_env import InfraError, LiberoEpisodeEnv  # noqa: E402
from envs.stream import heldout_specs  # noqa: E402
from habits.policy import HabitPolicy  # noqa: E402

CHUNK = 8


def eval_checkpoint(ckpt_path, env, specs):
    from envs.chained_env import execute_chunk_with_boundary

    policy = HabitPolicy(ckpt_path)
    per_episode = []
    n_succ = n_fail = n_infra = 0
    for spec in specs:
        try:
            obs = spec.realize(env)
            t, success, stale_discarded = 0, False, None
            while t < env.max_steps:
                chunk = policy.act_chunk(obs)
                # §4e 개정 α: 전환 감지 시 잔여 stale 행동 폐기 + 즉시 재질의 — 수집기와 동형
                obs, t, _n_exec, stale = execute_chunk_with_boundary(
                    env, list(chunk[:CHUNK]), t, env.max_steps)
                if stale is not None:
                    stale_discarded = stale
                if env.check_success():
                    success = True
                    break
            rec = {"uid": spec.uid, "outcome": "success" if success else "fail", "steps": t}
            if hasattr(env, "stage"):
                rec["stage"] = env.stage()  # chained: 실패의 stage 1/2 분해 (정직 보고)
                rec["stale_discarded"] = stale_discarded
            per_episode.append(rec)
            if success:
                n_succ += 1
            else:
                n_fail += 1
        except InfraError as e:
            per_episode.append({"uid": spec.uid, "outcome": "infra_error", "error": str(e)})
            n_infra += 1
    n_eval = n_succ + n_fail
    return {
        "ckpt": os.path.basename(ckpt_path),
        "n_eval": n_eval,
        "n_success": n_succ,
        "n_infra_error": n_infra,
        "s_hat": round(n_succ / n_eval, 4) if n_eval else None,
        "per_episode": per_episode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--n-grid", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--n-heldout", type=int, required=True)  # E2=50, E3=20 (preregistration §1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chained", action="store_true",
                    help="C-T2 2연쇄 래퍼 (preregistration §4e) — chained held-out 대역")
    ap.add_argument("--heldout-start", type=int, default=0,
                    help="스펙 시작 인덱스 — held-out 보충 평가용 (§4e 확대: 21–50 = start 20)")
    args = ap.parse_args()

    if args.chained:
        from envs.chained_env import ChainedEpisodeEnv, chained_heldout_specs

        env = ChainedEpisodeEnv(args.suite, args.task)
        specs = chained_heldout_specs(args.suite, args.task, args.n_heldout)
    else:
        env = LiberoEpisodeEnv(args.suite, args.task)
        specs = heldout_specs(args.suite, args.task, args.n_heldout)  # 체크포인트 간 동일 → paired
    specs = specs[args.heldout_start:]  # 보충 평가: 앞 구간(기평가분) 제외 — uid로 병합·중복 검증

    results = {"cluster": args.cluster, "n_heldout": args.n_heldout,
               "heldout_start": args.heldout_start, "curve": []}
    t0 = time.time()
    for n in sorted(args.n_grid):
        ckpt = os.path.join(args.ckpt_dir, f"act_n{n}.pt")
        if not os.path.exists(ckpt):
            results["curve"].append({"n": n, "missing": True})
            continue
        print(f"=== eval n={n} ===", flush=True)
        r = eval_checkpoint(ckpt, env, specs)
        r["n"] = n
        results["curve"].append(r)
        print(f"  n={n}: s_hat={r['s_hat']} ({r['n_success']}/{r['n_eval']}, infra={r['n_infra_error']})", flush=True)

    results["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.cluster}_curve.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[EVAL-PASS] cluster={args.cluster} json={out_path}")
    env.close()


if __name__ == "__main__":
    main()
