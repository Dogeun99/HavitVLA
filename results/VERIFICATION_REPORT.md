# VERIFICATION_REPORT — 릴리스 `src/`만으로 빌드해 `results/`와 일치하는가

- 실행: `20260903_release_final` (verify/verify_release.sh) · 판정 **PASS** (27 단계, FAIL 0)
- 원본 저장소 커밋 `5c12f9b` · 릴리스 src 위치 `/home/asmr/workspace/habitvla2_release/src`
- 빌드: 릴리스의 `envs/setup_envs.sh`로 **별도 conda env**(`hv2r_oft`, `hv2r_hab`; 원본 env를 clone한 뒤 릴리스 `third_party/`로 editable 재설치)를 구성해 사용. 원본 env·원본 저장소는 수정하지 않았다.
  - hab env: `/home/asmr/miniconda3/envs/hv2r_hab/bin/python 2.7.0+cu128 True /home/asmr/workspace/habitvla2_release/src/third_party/LIBERO/libero 1.26.4 3.1.6 1.4.1`
  - oft env: `/home/asmr/miniconda3/envs/hv2r_oft/bin/python 2.7.0+cu128 True /home/asmr/workspace/habitvla2_release/src/third_party/LIBERO/libero /home/asmr/workspace/habitvla2_release/src/third_party/openvla-oft/prismatic 4.40.1`
- 방법: 릴리스 `src/`를 scratch에 복제하고 `results/`의 **복사본**을 붙인 뒤, 저장된 원자료(원장 JSONL/CSV, 곡선 JSON)에서 모든 요약·통계를 다시 계산해 저장본과 대조했다. GPU 단계는 실제 시뮬레이터 롤아웃으로 체크포인트를 재평가해 **에피소드별** 결과(uid → 성공/실패, 스텝 수)를 저장본과 대조했다. 체크포인트·HDF5·모델 캐시는 원본 디렉터리에 링크(읽기만).

## 1. 단계별 결과

| 단계 | 판정 | 소요(s) | 성공 마커 |
|---|---|---:|---|
| `env_hab` | PASS | 1 | `-` |
| `env_oft` | PASS | 7 | `-` |
| `unit_gate_regression` | PASS | 0 | `[GATE-REGRESSION-PASS]` |
| `unit_executor_chunkbreak` | PASS | 0 | `[EXECUTOR-TEST-PASS]` |
| `verify_package` | PASS | 0 | `[PACKAGE-VERIFY-PASS]` |
| `rederive_e2_gonogo` | PASS | 1 | `-` |
| `rederive_e3_curves` | PASS | 0 | `[E3-CURVES]` |
| `rederive_e3_h2` | PASS | 1 | `-` |
| `rederive_e5_reading_0` | PASS | 0 | `-` |
| `rederive_e5_reading_1` | PASS | 1 | `-` |
| `rederive_e5_reading_2` | PASS | 0 | `-` |
| `rederive_e5_seed_synthesis` | PASS | 0 | `-` |
| `rederive_e5_postmortem_0` | PASS | 1 | `-` |
| `rederive_e5_postmortem_1` | PASS | 0 | `-` |
| `rederive_e5_postmortem_2` | PASS | 1 | `-` |
| `rederive_e4_scorer_table` | PASS | 0 | `-` |
| `rederive_rr_batch` | PASS | 0 | `[BATCH-STATS-DONE]` |
| `rederive_rr_online` | PASS | 1 | `[ONLINE-SUMMARY-DONE]` |
| `rederive_rr_replay` | PASS | 1 | `[PAIRED-DONE]` |
| `rederive_rr_old_vs_new` | PASS | 0 | `[OLDVSNEW-DONE]` |
| `compare_cpu` | PASS | 0 | `[COMPARE-PASS]` |
| `gpu_smoke_train_infer` | PASS | 25 | `[SMOKE-PASS]` |
| `gpu_eval_rgb_only_goal_task1` | PASS | 83 | `[EVAL-PASS]` |
| `gpu_eval_rgbd_object_task1` | PASS | 49 | `[EVAL-PASS]` |
| `gpu_latency_teacher_env` | PASS | 29 | `[LATENCY-DONE]` |
| `gpu_integrity_audit` | PASS | 19 | `[INTEGRITY-DONE]` |
| `compare_gpu` | PASS | 0 | `[COMPARE-PASS]` |

## 2. 분석 재산출 ↔ 저장 결과 대조 (CPU)

- 패키지 자체 검증: `[PACKAGE-VERIFY-PASS] checks=52 fail=0` (레포 코드 import 없이 원장 CSV에서 요약 재계산)
- 대조 파일 36개: **완전 동일 34**, 휘발 필드(출처 절대경로·시간) 제외 동일 2, 불일치 0 → `PASS`

| 파일 | 판정 | 비고 |
|---|---|---|
| `e2/e2_gonogo.json` | IDENTICAL |  |
| `e3/e3_curves.json` | IDENTICAL |  |
| `e3/h2_analysis.json` | IDENTICAL |  |
| `e5/reading_0.json` | IDENTICAL |  |
| `e5/reading_1.json` | IDENTICAL |  |
| `e5/reading_2.json` | IDENTICAL |  |
| `e5/seed_synthesis.json` | IDENTICAL |  |
| `e5/ineligible_postmortem_0.json` | IDENTICAL |  |
| `e5/ineligible_postmortem_1.json` | IDENTICAL |  |
| `e5/ineligible_postmortem_2.json` | IDENTICAL |  |
| `e4/e4_scorer_table.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/01_batch_formation/batch_episode_results.csv` | IDENTICAL | 2640 rows |
| `rgb_only_full_rerun_20260828/01_batch_formation/batch_summary.csv` | IDENTICAL | 108 rows |
| `rgb_only_full_rerun_20260828/01_batch_formation/NSTAR_RESULTS.csv` | IDENTICAL | 27 rows |
| `rgb_only_full_rerun_20260828/01_batch_formation/batch_statistics.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/08_statistics/rgb_only_e3_curves.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/02_online_seed0/ONLINE_EPISODE_LEDGER_seed0.csv` | IDENTICAL | 4000 rows |
| `rgb_only_full_rerun_20260828/02_online_seed0/ONLINE_SUMMARY_seed0.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/03_online_seed1/ONLINE_EPISODE_LEDGER_seed1.csv` | IDENTICAL | 4000 rows |
| `rgb_only_full_rerun_20260828/03_online_seed1/ONLINE_SUMMARY_seed1.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/04_online_seed2/ONLINE_EPISODE_LEDGER_seed2.csv` | IDENTICAL | 4000 rows |
| `rgb_only_full_rerun_20260828/04_online_seed2/ONLINE_SUMMARY_seed2.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/derived/ONLINE_SUMMARY_ALL_SEEDS.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/derived/LIFECYCLE_EVENTS_LONG.csv` | IDENTICAL | 345 rows |
| `rgb_only_full_rerun_20260828/derived/LIFECYCLE_CLUSTER_SUMMARY.csv` | IDENTICAL | 99 rows |
| `rgb_only_full_rerun_20260828/05_paired_replay/PAIRED_REPLAY_EPISODES.csv` | IDENTICAL | 4863 rows |
| `rgb_only_full_rerun_20260828/05_paired_replay/PAIRED_REPLAY_SUMMARY.json` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_seed0.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_seed1.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_seed2.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_fullstream_seed0.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_fullstream_seed1.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_fullstream_seed2.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/05_paired_replay/bootstrap_pooled.npy` | IDENTICAL |  |
| `rgb_only_full_rerun_20260828/08_statistics/OLD_VS_NEW_NUMERIC.csv` | EQUAL_MODULO_VOLATILE | 159 rows, 무시한 필드: note, source_new, source_old |
| `rgb_only_full_rerun_20260828/08_statistics/OLD_VS_NEW_NUMERIC.json` | EQUAL_MODULO_VOLATILE | 무시한 필드: source_new |

## 3. GPU — 실제 롤아웃·학습·무결성

- 학습·추론 스모크 (`rgb_only_rerun/smoke.py`, libero_goal_task1 200스텝 학습 → held-out 3 ep 추론): **PASS** — 11/11 검사 통과 (training_runs, checkpoint_saved, ckpt_use_depth_false, ckpt_in_ch_3, ckpt_steps_as_specified, policy_use_depth_false, policy_conv1_is_3ch, inference_runs, success_evaluation_works, logging_per_episode, no_infra_error)
- 체크포인트 held-out 재평가 `libero_goal_task1` (rgb_only_full_rerun_20260828/01_batch_formation/curves/libero_goal_task1_curve.json 대비): **IDENTICAL**
  - n=10: 에피소드 20개 중 성공/실패 일치 20, 스텝 수 일치 20, ŝ 재측정 0.85 / 저장 0.85
  - n=80: 에피소드 20개 중 성공/실패 일치 20, 스텝 수 일치 20, ŝ 재측정 1.0 / 저장 1.0
- 체크포인트 held-out 재평가 `libero_object_task1` (e3/libero_object_task1_curve.json 대비): **IDENTICAL**
  - n=80: 에피소드 20개 중 성공/실패 일치 20, 스텝 수 일치 20, ŝ 재측정 0.75 / 저장 0.75
- 무결성 감사 재실행 (`integrity_audit.py`): **VALID** (47 검사, FAIL 0); RGB-only 체크포인트 253개 전수 depth 미사용 위반 0 — 저장된 감사 JSON과 IDENTICAL

### 3.1 레이턴시 재측정 (teacher env `hv2r_oft`, attn=sdpa; 시간 측정이라 동일성 대신 나란히 기록)

| metric | 저장값 | 재측정 |
|---|---:|---:|
| `act_forward_rgb_only.median_ms` | 3.35 | 3.375 |
| `act_forward_rgb_only.p95_ms` | 3.363 | 3.422 |
| `act_forward_rgb_only.n_params` | 9.503e+07 | 9.503e+07 |
| `act_forward_rgbd.median_ms` | 3.353 | 3.385 |
| `act_forward_rgbd.p95_ms` | 3.364 | 3.397 |
| `act_forward_rgbd.n_params` | 9.5036e+07 | 9.5036e+07 |
| `gate_path.median_ms` | 4.179 | 4.084 |
| `gate_path.p95_ms` | 5.011 | 4.714 |
| `teacher_oft_chunk_forward.median_ms` | 85.466 | 85.386 |
| `teacher_oft_chunk_forward.p95_ms` | 86.808 | 86.499 |
| `ratios.act_rgb_only_over_teacher` | 0.0392 | 0.03953 |
| `ratios.act_rgbd_over_teacher` | 0.03923 | 0.03964 |
| `ratios.gate_over_teacher` | 0.0489 | 0.04783 |
| `ratios.denominator_ms` | 85.466 | 85.386 |
| `formation_timing.training_by_n.20.median_s` | 226.6 | 226.6 |
| `formation_timing.training_by_n.80.median_s` | 627.9 | 627.9 |
| `formation_timing.formation_event_total.median_s` | 288.8 | 288.8 |
| `formation_timing.training_only.median_s` | 227 | 227 |
| `formation_timing.probe_and_prep.median_s` | 57.1 | 57.1 |
| `operational_time_summary.operational_h.median_s` | 9.2 | 9.2 |
| `operational_time_summary.formation_h.median_s` | 5.48 | 5.48 |

## 4. 해석

- 저장된 모든 요약·통계(E2 go/no-go, E3 27 곡선·N*, H2 분석, E5 3 seed 판독·종합·사후분석, E4 scorer 표, RGB-only rerun의 배치/온라인/paired replay/부트스트랩 분포/old-vs-new)는 릴리스 코드로 원자료에서 **비트 단위로 재산출**된다.
- 시뮬레이터 + ACT 정책 스택은 릴리스 폴더의 third_party(핀 커밋 + 패치)와 릴리스 env에서 저장된 롤아웃을 **에피소드 단위로 결정적으로 재현**한다 (RGB-only·RGB-D 체크포인트 각 1 클러스터, 성공/실패와 스텝 수까지 일치).
- 레이턴시는 하드웨어 시간 측정이므로 ms 단위 소수점에서만 다르고 순위·비율(ACT/teacher ≈ 0.039)은 같다.
- 검증하지 않은 것: 70 h짜리 전체 재실행(배치 27 클러스터 학습, 온라인 12,000 ep, paired replay)과 teacher 궤적 재수집. 이들은 결정적 에피소드 명세(§4h)와 seed 고정으로 재현 가능하도록 설계돼 있고, 위 체크포인트 재평가가 그 실행 경로(시뮬·정책·성공 판정)를 덮는다.
