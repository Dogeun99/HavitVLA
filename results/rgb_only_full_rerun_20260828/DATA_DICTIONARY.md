# DATA_DICTIONARY

각 파일의 **1행 = 무엇 하나**인지와 열의 의미. 값은 전부 실측이며 파생값은 출처를 표기한다.

## batch_episode_results.csv — 1행 = 배치 held-out 평가 에피소드 1개
| 열 | 의미 |
|---|---|
| cluster_id / suite / task_id | 클러스터 식별 |
| n | 학습에 쓴 teacher 성공 궤적 수 (10/20/40/80) |
| training_seed | ACT 학습 seed (전 클러스터 동일, 동결) |
| eval_episode_id / initial_state_id | EpisodeSpec uid (6원소 해시 — 초기상태 명세의 결정적 식별자) |
| success | 1/0, LIBERO 공식 predicate |
| outcome | success / fail / infra_error (인프라 오류는 성공·실패 어느 쪽에도 산입하지 않음) |
| steps | 소요 스텝 |
| checkpoint_path | 이 행을 만든 체크포인트 |
| n_heldout_protocol | 이 클러스터의 held-out 규모 (20 또는 50 — 기존 프로토콜 그대로) |
| in_e3_view | 1이면 E3 관점(앞 20 스펙) 집계에 포함된 행 |

## batch_summary.csv — 1행 = (클러스터, n)
num_trials·num_success·success_rate·n_infra_error. success_rate = num_success/num_trials.

## NSTAR_RESULTS.csv — 1행 = 클러스터
N_star = ŝ(n) ≥ 0.8을 처음 만족하는 n. 미달이면 ">80"(우측절단, right_censored=1).
formable = N_star가 그리드 안에서 정의됨. wilson_80_* = ŝ(80)의 Wilson 95% 구간.

## batch_statistics.json — §7 통계
decomposition_L(순위 분산 분해·Kruskal–Wallis) · regression_*(순위 OLS + 순열 p, B=10⁴) ·
horizon_T1_vs_T3(단측) · controlled_chain_product_baseline(곱 기준선 이항 + 모수 부트스트랩) ·
intermediate_inputs(재계산용 입력 테이블). 추정량은 기존 e3_* 스크립트 함수를 그대로 import.

## ONLINE_EPISODE_LEDGER_seedX.csv — 1행 = 스트림 에피소드 1개 (seed당 4,000)
| 열 | 의미 |
|---|---|
| seed / episode(t) | seed 인덱스, 스트림 내 0-based 위치 |
| cluster_id / suite / task_id | 클러스터 |
| cold_start | 이 클러스터가 novel 주입 풀(Spatial-b) 소속인가 |
| is_novel_injection | 이 에피소드가 novel 주입분인가 |
| spec_uid / initial_state_id / episode_seed / observation_noise_seed / perturbation_width | 에피소드 명세 6원소 (재현 키) |
| controller / executor | habit 또는 vla |
| decision_reason | fire / immature / unknown_cluster / habit_ineligible / infra |
| state_before / state_after | lifecycle 4상태 U·I·M·X |
| B_k_size (bc_pool) | 그 시점 BC 풀 크기 (teacher 성공 궤적 수) |
| training_triggered / training_round | 재학습 발생 여부와 정책 버전 |
| probe_triggered / probe_success_count / probe_failure_count | 재학습 직후 P=20 probe 결과 |
| sigma_k / phi_k | A_mat 사후 계수 (습관 출처 probe+fire만 산입) |
| tau_k | 그 시점 ACI 임계 |
| habit_fired / habit_success | 발화 여부와 그 결과 (비발화면 null) |
| teacher_used / teacher_success | teacher 실행 여부와 결과 (발화면 null) |
| demotion / rematuration / transition_to_X | 이 에피소드에서 일어난 상태 전이 |
| episode_success | 시스템 관점 성공 (infra_error면 null) |
| VLA_calls / habit_calls | 이 에피소드의 chunk 질의 횟수 |
| episode_latency / wall_s | 에피소드 벽시계 초 |
| wall_clock_time | 에피소드 시작 시각 |
| shadow_jur_* | 그림자 관할 기록 (불개입 — 발화 결정에 쓰이지 않음) |
| aci_* | ACI 누적 상태 |

## LIFECYCLE_EVENTS_LONG.csv — 1행 = 상태 전이 이벤트 1건
event_type ∈ {first_exposure, first_training, retraining, first_maturity, rematuration,
demotion, transition_X}. 에피소드 원장의 전이 플래그와 건수가 일치해야 한다(무결성 검사 항목).

## LIFECYCLE_CLUSTER_SUMMARY.csv — 1행 = (seed, 클러스터)
first_exposure/first_training/second_training/first_maturity/first_X 에피소드 인덱스,
num_firings·num_failures·num_demotions·num_rematurations, final_state, final_B_k_size.

## ONLINE_SUMMARY_seedX.json / ONLINE_SUMMARY_ALL_SEEDS.json — §10
vla_routing_rate(full/first_1000/last_1000/200-ep 창) · system_success · final_lifecycle(M/I/X/U) ·
lifecycle(ever mature·demotion·rematuration·probe 라운드별 통과) · risk(Pr(fail|fire)) ·
cold_start · late_traffic_last1000 · call_accounting. ALL_SEEDS는 완료된 seed의 mean/sd(ddof=1).

## PAIRED_REPLAY_EPISODES.csv — 1행 = 발화 에피소드 1개의 paired 비교
system_success · habit_success · full_vla_success · difference(system − full_vla) + 명세 6원소.

## PAIRED_REPLAY_SUMMARY.json
per_seed에 두 구성이 함께 있다:
  (a) 발화 집합 paired (§11 문면) — n_paired_episodes = 발화 수
  (b) full_stream_noninferiority — 발화분은 CF 재현, 비발화분은 VLA 실측 (논문 H4b와 동일 구성)
bootstrap B=10,000, seed 0. bootstrap_seed{S}.npy / bootstrap_fullstream_seed{S}.npy에
재표집 분포 전체를 저장했으므로 CI를 다시 그릴 수 있다.

## FAMILIARITY_*
DEPENDENCY_AUDIT = 지표별 (habit modality 의존 여부 / 재계산 여부 / 출처).
EPISODES.csv 1행 = 역량 지도 에피소드 1개 (섭동 폭 w, habit 성공, Mahalanobis 점수,
기각 여부, kNN k=5·10 점수, ID/boundary/OOD 라벨). teacher_hidden_score는 재계산 대상이
아니어서 null이며 원 출처는 audit에 기록돼 있다.

## LATENCY_RAW.csv — 1행 = 측정 샘플 1개 (anchor, sample_idx, ms)
anchor ∈ {act_forward_rgb_only, act_forward_rgbd, gate_path, teacher_oft_chunk_forward}.
warmup 10 + 측정 100, cuda.synchronize 경계, attn=sdpa.

## FORMATION_TIMING_RAW.csv — 1행 = 온라인 재학습 이벤트 1건
train_wall_s(학습만) · probe_and_prep_wall_s · formation_event_wall_s(합) ·
probe_success_count/probe_failure_count · n · probe_round.

## OLD_VS_NEW_NUMERIC.csv — 1행 = 지표 1개
metric · seed_or_cluster · old_rgbd · new_rgb · absolute_change · relative_change ·
source_old · source_new. **내부 검증용이며 해석 문장은 없다.**

## DATA_INTEGRITY_AUDIT.json — §14
검사별 status(PASS/FAIL)와 detail. FAIL이 하나라도 있으면 overall=INVALID.

## CHECKPOINT_MANIFEST.csv
가중치는 패키지에 넣지 않았다. path·size_bytes·sha256·seed·cluster·training_round로 대조한다.
