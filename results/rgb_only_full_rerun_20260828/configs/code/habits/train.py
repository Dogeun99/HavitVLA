"""클러스터별 ACT 학습 — n-grid {10,20,40,80} 체크포인트 (설계서 §2.3, preregistration §1).

- warm-start 허용(사전등록 명시): n=20은 n=10 체크포인트에서 이어 학습, 이하 동일.
- HP: C-L0에서만 튜닝 허용. 여기의 기본값이 C-L0 확정 전 초기값이며, 확정 후 동결.
- 산출: checkpoints/<cluster>/act_n{N}.pt (모델 + 정규화 통계 + 학습 메타)

실행:
  conda run -n hv2_hab python -u habits/train.py \
    --h5 data/e2/libero_object_task0.hdf5 --cluster libero_object_task0 \
    --n-grid 10 20 40 80 --out checkpoints
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))  # 공용 캐시 불침범

from habits.act import ACTPolicy  # noqa: E402
from habits.dataset import ClusterDataset, compute_stats, load_cluster  # noqa: E402

# 기본 HP (C-L0 튜닝 대상 — 확정 후 동결. ACT 공개 구현 표준에서 출발)
HP = {
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "batch_size": 8,
    "steps_per_n": {10: 4000, 20: 6000, 40: 8000, 80: 10000},  # warm-start 누적이 아닌 해당 n 총 스텝
    "weight_decay": 1e-4,
    "kl_weight": 10.0,
    "seed": 0,
}


def train_one(episodes, n, prev_ckpt, out_path, stats, device="cuda", hp=HP, log_every=200,
              use_depth=True):
    """stats: 클러스터 공용 정규화 통계 (max-n 풀에서 1회 산출 — warm-start가 정규화 공간을
    가로지르지 않도록 전 n-grid 단계 동결. preregistration §5 이력 기록)."""
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    subset = episodes[:n]
    if len(subset) < n:
        print(f"[WARN] requested n={n} but only {len(subset)} success episodes available")
    ds = ClusterDataset(subset, stats, use_depth=use_depth)
    dl = DataLoader(
        ds, batch_size=hp["batch_size"], shuffle=True, num_workers=2,
        drop_last=False, persistent_workers=True,
    )

    model = ACTPolicy(kl_weight=hp["kl_weight"], in_ch=4 if use_depth else 3).to(device)
    start_step = 0
    if prev_ckpt and os.path.exists(prev_ckpt):
        sd = torch.load(prev_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"warm-start from {prev_ckpt}")

    backbone_params = [p for p in model.backbones.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": hp["lr"]},
            {"params": backbone_params, "lr": hp["lr_backbone"]},
        ],
        weight_decay=hp["weight_decay"],
    )
    spn = hp["steps_per_n"]
    total = spn.get(n, 10000) if isinstance(spn, dict) else spn

    model.train()
    step, t0 = start_step, time.time()
    losses = []
    while step < total:
        for batch in dl:
            images = [batch["agentview"].to(device), batch["wrist"].to(device)]
            loss, parts = model.loss(
                images,
                batch["proprio"].to(device),
                batch["actions"].to(device),
                batch["pad_mask"].to(device),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(parts["l1"])
            step += 1
            if step % log_every == 0:
                print(
                    f"  step {step}/{total} l1={np.mean(losses[-log_every:]):.4f} "
                    f"kl={parts['kl']:.3f} ({time.time()-t0:.0f}s)",
                    flush=True,
                )
            if step >= total:
                break

    torch.save(
        {
            "model": model.state_dict(),
            "stats": stats,
            "hp": hp,
            "n_episodes": len(subset),
            "n_samples": len(ds),
            "steps": step,
            "final_l1": float(np.mean(losses[-200:])) if losses else None,
            "train_seconds": round(time.time() - t0, 1),
            "use_depth": use_depth,
            "in_ch": 4 if use_depth else 3,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        out_path,
    )
    return {"n": n, "final_l1": float(np.mean(losses[-200:])), "train_seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--n-grid", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--out", default=os.path.join(HABIT2, "checkpoints"))
    ap.add_argument("--no-warm-start", action="store_true")
    # E5 요건 (설계서 v0.3 §4): lazy 재학습은 호출이 분리되므로 이전 버전 체크포인트를
    # 명시적으로 승계해야 warm-start 규율(E2/E3와 동일)이 유지된다.
    ap.add_argument("--warm-from", default=None, help="시작 체크포인트 (lazy 재학습 승계)")
    ap.add_argument("--no-depth", action="store_true",
                    help="RGB-only ablation — depth 채널 제거 (기본: RGB-D)")
    ap.add_argument("--steps", type=int, default=None,
                    help="총 스텝 수 override. E5 재학습은 scratch이므로 배치 등가 총량을 "
                         "명시 지정한다(§5 HP 개정 2026-08-17). 스모크는 축소값.")
    args = ap.parse_args()
    if args.steps:
        HP["steps_per_n"] = args.steps

    episodes = load_cluster(args.h5)
    print(f"cluster={args.cluster} success episodes={len(episodes)}")
    outdir = os.path.join(args.out, args.cluster)
    os.makedirs(outdir, exist_ok=True)

    # 정규화 통계: max-n 풀에서 1회 산출, 전 단계 동결 (warm-start 정규화 공간 일관성)
    pool_stats = compute_stats(episodes[: max(args.n_grid)])

    results, prev = [], args.warm_from
    for n in sorted(args.n_grid):
        out_path = os.path.join(outdir, f"act_n{n}.pt")
        print(f"=== train n={n} -> {out_path} ===", flush=True)
        r = train_one(episodes, n, None if args.no_warm_start else prev, out_path, pool_stats,
                      use_depth=not args.no_depth)
        results.append(r)
        prev = out_path

    with open(os.path.join(outdir, "train_summary.json"), "w") as f:
        json.dump({"cluster": args.cluster, "results": results}, f, indent=2)
    print(f"[TRAIN-PASS] cluster={args.cluster} grid={args.n_grid} depth={not args.no_depth}")


if __name__ == "__main__":
    main()
