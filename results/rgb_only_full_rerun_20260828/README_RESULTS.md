# README_RESULTS — HabitVLA_RGB_only_full_rerun_20260828

RGB-only full rerun의 **데이터 패키지**다. 논문 해석문·그림·표는 들어 있지 않다.

## 1. run ID
- `rgb_only_full_rerun_20260828`
- 패키지: `HabitVLA_RGB_only_full_rerun_20260828`

## 2. git commit
- commit `2ffbc37aab0fc3cc27029ae4fb03b9c544f6b821`  branch `master`
- working tree clean: False
- RGB-only 관련 파일 sha256은 `ENVIRONMENT.json`의 `source.rgb_only_relevant_files`

## 3. 실행 환경
- NVIDIA GeForce RTX 5090 · driver/CUDA: `NVIDIA GeForce RTX 5090, 580.159.03, 32607 MiB`
- torch 2.7.0+cu128 (cuda 12.8) · python 3.11.15
- attention: sdpa (flash-attn은 sm_120 미빌드 — CLAUDE.md §0)
- conda env: ACT=`hv2_hab`, teacher=`hv2_oft`

## 4. RGB-only 변경 내용
- **depth 제거 하나뿐이다.** ACT 백본 conv1을 4채널 → 3채널로 좁혔다.
- teacher(OpenVLA-OFT)·teacher 궤적·클러스터 집합·train/eval split·에피소드 명세·
  seed 대역·n grid·재학습 지점·P=20·K=8·optimizer·lr·batch·steps·augmentation·
  RGB 정규화·action 정규화·proprio 표현·게이트 상수(τ·δ·γ·ε·c)는 전부 동결.
- key-by-key 대조는 `CONFIG_DIFF.json` (허용 차이 = depth 관련 키뿐).
- 런타임 depth 미사용 증명은 `RGB_ONLY_INPUT_AUDIT.json`.

## 5. 완료된 experiment

- preflight: **PASS** 
- smoke: **PASS** 
- batch: **PASS** 27/27
- online_seed0: **PASS** 4000/4000
- replay_seed0: **PASS** 1538/1538
- online_seed1: **PASS** 4000/4000
- replay_seed1: **PASS** 1664/1664
- online_seed2: **PASS** 4000/4000
- replay_seed2: **PASS** 1661/1661
- batch_statistics: **PASS** 
- online_summary: **PASS** 
- familiarity: **PASS** 
- latency: **PASS** 
- integrity: **PASS** 
- old_vs_new: **PASS** 
- package: **WAIT** 

## 6. 실패한 experiment

- 없음

## 7~8. 파일 위치와 1행의 단위

열 단위 정의는 `DATA_DICTIONARY.md`에 전부 있다. 주요 진입점:

| 파일 | 1행 = |
|---|---|
| `01_batch_formation/batch_episode_results.csv` | 배치 평가 에피소드 |
| `01_batch_formation/NSTAR_RESULTS.csv` | 클러스터 |
| `01_batch_formation/batch_statistics.json` | (통계 묶음) |
| `0X_online_seedS/ONLINE_EPISODE_LEDGER_seedS.csv` | 스트림 에피소드 |
| `derived/LIFECYCLE_EVENTS_LONG.csv` | 상태 전이 이벤트 |
| `derived/LIFECYCLE_CLUSTER_SUMMARY.csv` | (seed, 클러스터) |
| `derived/ONLINE_SUMMARY_ALL_SEEDS.json` | (3-seed 집계) |
| `05_paired_replay/PAIRED_REPLAY_EPISODES.csv` | 발화 에피소드의 paired 비교 |
| `06_familiarity/FAMILIARITY_EPISODES.csv` | 역량 지도 에피소드 |
| `07_latency_cost/LATENCY_RAW.csv` | 레이턴시 측정 샘플 |
| `07_latency_cost/FORMATION_TIMING_RAW.csv` | 재학습 이벤트 |
| `08_statistics/OLD_VS_NEW_NUMERIC.csv` | 대조 지표 |

원자료 JSONL(`raw/`)이 모든 CSV의 상위 출처다.

## 9. 재현 command

```bash
# 전 단계 (marker 기반 resume — 이미 끝난 stage는 건너뛴다)
hv2_hab python -u experiments/rgb_only_rerun/preflight.py
hv2_hab python -u experiments/rgb_only_rerun/smoke.py
hv2_hab python -u experiments/rgb_only_rerun/run_all.py

# 개별 stage
hv2_hab python -u experiments/rgb_only_rerun/run_batch.py
hv2_oft python -u experiments/e5_driver.py --seed-idx S --n 4000 --no-depth \
    --out-root <ROOT>/0X_online_seedS --ck-root checkpoints/rgb_only_rerun/online \
    --data-root data/rgb_only_rerun/online
hv2_oft python -u experiments/e5_counterfactual.py --seed-idx S \
    --queue-root <ROOT>/0X_online_seedS --out-root <ROOT>/05_paired_replay
```

## 10. 무결성 검사 결과

- **overall = VALID** (검사 47건, FAIL 0건)

## 11. old vs new 수치 대조
- `08_statistics/OLD_VS_NEW_NUMERIC.csv` (행마다 source_old·source_new 포함)
- **내부 검증용이다. 해석은 이 패키지를 받는 쪽에서 한다.**

## 12. 체크포인트

- 가중치 253개 · 89.9 GB 는 패키지에 넣지 않았다 (§21).
- `CHECKPOINT_MANIFEST.csv`의 path·sha256으로 원본 디렉터리에서 대조한다.
- 원본 경로: `checkpoints/rgb_only_rerun/batch/`, `checkpoints/rgb_only_rerun/online/`

## 13. 이 패키지가 하지 않은 것

논문 문장·LaTeX·PDF·Fig.·publication table을 만들지 않았다 (지시 §18·§23).
필요한 것은 이후 환경에서 이 데이터로 전부 재구성할 수 있다.
