#!/usr/bin/env bash
# 릴리스 검증 오케스트레이터 — "이 폴더만 빌드해서 저장된 결과와 일치하게 동작하는가".
#
# 절차
#   0. 릴리스 src를 scratch(verify_runs/<stamp>/src_copy)에 복제하고 results/를 **복사본**으로 둔다
#      (릴리스의 results/는 건드리지 않는다). checkpoints/data/캐시는 원본 작업 디렉터리로 링크.
#   1. 단위 테스트 (gate_regression, executor_chunkbreak_test)
#   2. 패키지 자체 검증 (verify_package.py — 원장에서 요약 재계산, 52 검사)
#   3. RGB-D 본 실험 분석 재산출 (e2/e3/h2/e5 판독/seed 종합/사후분석/e4 scorer 표)
#   4. RGB-only rerun 분석 재산출 (batch/online/replay/old_vs_new)
#   5. 3·4 산출물을 저장 결과와 대조 (compare_outputs.py)
#   6. GPU: 학습·추론 스모크, 체크포인트 held-out 재평가(RGB-only·RGB-D 각 1 클러스터)를
#      저장된 에피소드별 결과와 대조, 레이턴시 재측정(teacher env), 무결성 감사(체크포인트 전수)
# 사용: verify/verify_release.sh [--no-gpu] [--run-dir=DIR]
#   환경변수 HV2_HAB_PY / HV2_OFT_PY 로 두 env의 python을 지정 (기본 ~/miniconda3/envs/hv2_{hab,oft}/bin/python)
#   ORIG = 대용량 자산(체크포인트·HDF5·HF 캐시)이 있는 원본 작업 디렉터리 (기본 /home/asmr/workspace/habitvla2)
set -uo pipefail
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REL=$(cd "$SRC/.." && pwd)
ORIG=${ORIG:-/home/asmr/workspace/habitvla2}
export HV2_HAB_PY=${HV2_HAB_PY:-$HOME/miniconda3/envs/hv2_hab/bin/python}
export HV2_OFT_PY=${HV2_OFT_PY:-$HOME/miniconda3/envs/hv2_oft/bin/python}
GPU=1; RUN=""
for a in "$@"; do case $a in --no-gpu) GPU=0;; --run-dir=*) RUN=${a#*=};; esac; done
RUN=${RUN:-$REL/verify_runs/$(date +%Y%m%d_%H%M%S)}
W=$RUN/src_copy; LOGS=$RUN/logs; REF=$REL/results
mkdir -p "$LOGS"
STEPS=$RUN/STEPS.tsv; : > "$STEPS"
echo "RUN=$RUN"; echo "HV2_HAB_PY=$HV2_HAB_PY"; echo "HV2_OFT_PY=$HV2_OFT_PY"

# ---------- 0. scratch 복제
rsync -a --exclude third_party --exclude results --exclude data --exclude checkpoints \
      --exclude .hf_cache --exclude .torch_cache --exclude .libero --exclude logs \
      --exclude __pycache__ --exclude verify_runs "$SRC/" "$W/"
ln -sfn "$SRC/third_party" "$W/third_party"
cp -r "$SRC/.libero" "$W/.libero"
mkdir -p "$W/logs"
cp -a "$REF" "$W/results"
bash "$W/setup/link_local_assets.sh" "$ORIG" > "$LOGS/link_assets.log" 2>&1 || { cat "$LOGS/link_assets.log"; exit 1; }
rm -f "$W/checkpoints/rgb_only_rerun/smoke" "$W/checkpoints/rgb_only_rerun/e5_smoke"   # 스모크 산출은 scratch에만
export HF_HOME=$W/.hf_cache TORCH_HOME=$W/.torch_cache LIBERO_CONFIG_PATH=$W/.libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
cd "$W"

step() {  # step <name> <marker|-> <cmd...>   → STEPS.tsv: name status exit elapsed marker
  local name=$1 marker=$2; shift 2
  local t0=$(date +%s)
  echo "[RUN] $name"
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$? ; local dt=$(( $(date +%s) - t0 )) ; local ok=$rc
  if [ "$rc" = 0 ] && [ "$marker" != "-" ] && ! grep -qF -- "$marker" "$LOGS/$name.log"; then ok=99; fi
  local st=PASS; [ "$ok" = 0 ] || st=FAIL
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$st" "$rc" "$dt" "$marker" >> "$STEPS"
  echo "[$st] $name (exit=$rc, ${dt}s, marker=$marker)"
}
PYH=$HV2_HAB_PY; PYO=$HV2_OFT_PY

# ---------- env
step env_hab - $PYH -c "import torch,libero,numpy,mujoco,robosuite,sys; print(sys.executable, torch.__version__, torch.cuda.is_available(), libero.__path__[0], numpy.__version__, mujoco.__version__, robosuite.__version__)"
step env_oft - $PYO -c "import torch,libero,prismatic,transformers,sys,os; print(sys.executable, torch.__version__, torch.cuda.is_available(), libero.__path__[0], os.path.dirname(prismatic.__file__), transformers.__version__)"

# ---------- 1. 단위 테스트
step unit_gate_regression "[GATE-REGRESSION-PASS]" $PYH -u experiments/gate_regression.py
step unit_executor_chunkbreak "[EXECUTOR-TEST-PASS]" $PYH -u experiments/executor_chunkbreak_test.py

# ---------- 2. 패키지 자체 검증
step verify_package "[PACKAGE-VERIFY-PASS]" $PYH -u experiments/rgb_only_rerun/verify_package.py --package results/rgb_only_full_rerun_20260828

# ---------- 3. RGB-D 본 실험 분석 재산출
step rederive_e2_gonogo - $PYH -u experiments/e2_collect.py
step rederive_e3_curves "[E3-CURVES]" $PYH -u experiments/e3_collect.py
step rederive_e3_h2 - $PYH -u experiments/e3_h2_analysis.py
for s in 0 1 2; do step rederive_e5_reading_$s - $PYH -u experiments/e5_analyze.py --seed-idx $s; done
step rederive_e5_seed_synthesis - $PYH -u experiments/e5_seed_synthesis.py
for s in 0 1 2; do step rederive_e5_postmortem_$s - $PYH -u experiments/e5_ineligible_postmortem.py --seed-idx $s; done
step rederive_e4_scorer_table - $PYH -u experiments/e4_scorer_table.py

# ---------- 4. RGB-only rerun 분석 재산출
step rederive_rr_batch "[BATCH-STATS-DONE]" $PYH -u experiments/rgb_only_rerun/analyze_batch.py
step rederive_rr_online "[ONLINE-SUMMARY-DONE]" $PYH -u experiments/rgb_only_rerun/analyze_online.py
step rederive_rr_replay "[PAIRED-DONE]" $PYH -u experiments/rgb_only_rerun/analyze_replay.py
step rederive_rr_old_vs_new "[OLDVSNEW-DONE]" $PYH -u experiments/rgb_only_rerun/old_vs_new.py

# ---------- 5. 대조 (CPU 산출물)
RR=rgb_only_full_rerun_20260828
CPU_FILES=(e2/e2_gonogo.json e3/e3_curves.json e3/h2_analysis.json
  e5/reading_0.json e5/reading_1.json e5/reading_2.json e5/seed_synthesis.json
  e5/ineligible_postmortem_0.json e5/ineligible_postmortem_1.json e5/ineligible_postmortem_2.json
  e4/e4_scorer_table.json
  $RR/01_batch_formation/batch_episode_results.csv $RR/01_batch_formation/batch_summary.csv
  $RR/01_batch_formation/NSTAR_RESULTS.csv $RR/01_batch_formation/batch_statistics.json
  $RR/08_statistics/rgb_only_e3_curves.json
  $RR/02_online_seed0/ONLINE_EPISODE_LEDGER_seed0.csv $RR/02_online_seed0/ONLINE_SUMMARY_seed0.json
  $RR/03_online_seed1/ONLINE_EPISODE_LEDGER_seed1.csv $RR/03_online_seed1/ONLINE_SUMMARY_seed1.json
  $RR/04_online_seed2/ONLINE_EPISODE_LEDGER_seed2.csv $RR/04_online_seed2/ONLINE_SUMMARY_seed2.json
  $RR/derived/ONLINE_SUMMARY_ALL_SEEDS.json $RR/derived/LIFECYCLE_EVENTS_LONG.csv $RR/derived/LIFECYCLE_CLUSTER_SUMMARY.csv
  $RR/05_paired_replay/PAIRED_REPLAY_EPISODES.csv $RR/05_paired_replay/PAIRED_REPLAY_SUMMARY.json
  $RR/05_paired_replay/bootstrap_seed0.npy $RR/05_paired_replay/bootstrap_seed1.npy $RR/05_paired_replay/bootstrap_seed2.npy
  $RR/05_paired_replay/bootstrap_fullstream_seed0.npy $RR/05_paired_replay/bootstrap_fullstream_seed1.npy
  $RR/05_paired_replay/bootstrap_fullstream_seed2.npy $RR/05_paired_replay/bootstrap_pooled.npy
  $RR/08_statistics/OLD_VS_NEW_NUMERIC.csv $RR/08_statistics/OLD_VS_NEW_NUMERIC.json)
step compare_cpu "[COMPARE-PASS]" $PYH -u verify/compare_outputs.py --ref "$REF" --new "$W/results" --files "${CPU_FILES[@]}" --out "$RUN/compare_cpu.json"

# ---------- 6. GPU
if [ "$GPU" = 1 ]; then
  step gpu_smoke_train_infer "[SMOKE-PASS]" $PYH -u experiments/rgb_only_rerun/smoke.py
  mkdir -p "$RUN/rollout_rgb_only" "$RUN/rollout_rgbd"
  step gpu_eval_rgb_only_goal_task1 "[EVAL-PASS]" $PYH -u habits/evaluate.py --cluster libero_goal_task1 --suite libero_goal --task 1 \
       --ckpt-dir checkpoints/rgb_only_rerun/batch/libero_goal_task1 --n-grid 10 80 --n-heldout 20 --out "$RUN/rollout_rgb_only"
  step gpu_eval_rgbd_object_task1 "[EVAL-PASS]" $PYH -u habits/evaluate.py --cluster libero_object_task1 --suite libero_object --task 1 \
       --ckpt-dir checkpoints/libero_object_task1 --n-grid 80 --n-heldout 20 --out "$RUN/rollout_rgbd"
  step gpu_latency_teacher_env "[LATENCY-DONE]" $PYO -u experiments/rgb_only_rerun/measure_latency.py
  step gpu_integrity_audit "[INTEGRITY-DONE]" $PYH -u experiments/rgb_only_rerun/integrity_audit.py
  step compare_gpu "[COMPARE-PASS]" $PYH -u verify/compare_outputs.py --ref "$REF" --new "$W/results" \
       --files $RR/09_integrity/DATA_INTEGRITY_AUDIT.json \
       --rollout "$RUN/rollout_rgb_only/libero_goal_task1_curve.json" "$REF/$RR/01_batch_formation/curves/libero_goal_task1_curve.json" \
       --rollout "$RUN/rollout_rgbd/libero_object_task1_curve.json" "$REF/e3/libero_object_task1_curve.json" \
       --out "$RUN/compare_gpu.json"
  # 레이턴시는 시간 측정이라 수치 동일성 대신 나란히 기록한다
  $PYH - "$REF/$RR/07_latency_cost/COMPUTE_SUMMARY.json" "$W/results/$RR/07_latency_cost/COMPUTE_SUMMARY.json" > "$RUN/latency_side_by_side.txt" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def walk(d, p=""):
    for k, v in d.items():
        q = f"{p}.{k}" if p else k
        if isinstance(v, dict): yield from walk(v, q)
        elif isinstance(v, (int, float)) and not isinstance(v, bool): yield q, v
A, B = dict(walk(a)), dict(walk(b))
print(f"{'metric':60s} {'stored':>14s} {'re-measured':>14s}")
for k in A:
    if k in B: print(f"{k:60s} {A[k]:>14.5g} {B[k]:>14.5g}")
PY
fi

# ---------- 요약
$PYH - "$STEPS" "$RUN" <<'PY'
import sys, json, csv
steps = [dict(zip(["name","status","exit","elapsed_s","marker"], r)) for r in csv.reader(open(sys.argv[1]), delimiter="\t")]
n_fail = sum(s["status"] != "PASS" for s in steps)
out = {"run_dir": sys.argv[2], "n_steps": len(steps), "n_fail": n_fail, "verdict": "PASS" if n_fail == 0 else "FAIL", "steps": steps}
json.dump(out, open(sys.argv[2] + "/SUMMARY.json", "w"), indent=1, ensure_ascii=False)
for s in steps: print(f"{s['status']:5s} {s['name']:36s} {s['elapsed_s']:>6s}s")
print(f"[VERIFY-RELEASE-{out['verdict']}] steps={len(steps)} fail={n_fail}  → {sys.argv[2]}/SUMMARY.json")
PY
