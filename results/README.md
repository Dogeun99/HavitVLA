# HabitVLA-2 — 결과 (릴리스 판)

원본 저장소(`/home/asmr/workspace/habitvla2`, 커밋 `5c12f9b`, 2026-08-31)에서 **git이 추적하던 결과 파일 전부**와
RGB-only 전체 재실행 데이터 패키지, 판정 보고서 문서를 모았다. 모든 수치의 출처는 이 폴더의 JSON/CSV이며
(수동 입력 금지 원칙, `../src/CLAUDE.md` §6), 아래 표의 값도 그 파일에서 옮겨 적은 것이다.
`../src/results`가 이 폴더를 가리키므로 `../src`의 분석 스크립트는 그대로 이 파일들을 읽고 쓴다.

검증(이 폴더의 결과가 `../src`만으로 재산출·재현되는가): **`VERIFICATION_REPORT.md`**.

## 1. 두 실험 라인

| 라인 | 기간 | 습관 입력 | 위치 |
|---|---|---|---|
| **본 실험 E0–E5** (사전등록 프로토콜, E6 다중 seed는 미실시 — E5 안에서 3 seed 수행) | 2026-08-15 → 08-21 (+ 08-28 depth ablation) | RGB-D (2-view RGB + depth + proprio) | `e0/ … e5/`, `rgb_depth_ablation/` |
| **RGB-only 전체 재실행** — teacher(RGB)와 habit의 modality mismatch 제거, 변경은 depth 제거 하나 | 2026-08-28 23:03 → 08-31 20:39 (≈70 h 무인) | RGB (2-view RGB + proprio) | `rgb_only_full_rerun_20260828/` |

## 2. 디렉터리

| 경로 | 내용 | 주요 파일 |
|---|---|---|
| `e0/` | 환경·depth·변이 폭·스모크·wall-clock go/no-go | `e0_5_smoke.json`(4 스위트 10/10), `e0_6_variation.json`(usable_w_max), `e0_3_depth.json` |
| `e1/` | teacher S_V(1,000 ep) + 레이턴시 앵커 | `e1_sv.json`, `e1_sv_per_task.json`, `e1_latency.json` |
| `e2/` | ★ 유일 치명 단계 — 형성 실증 (object task0/task5, held-out 50) | `e2_gonogo.json`, `*_curve.json` |
| `e3/` | 27 클러스터 성숙 곡선 ŝ_k(n), N*, H2(이중 해리) 분석, C-T2 진단 | `e3_curves.json`(단일 진입점), `h2_analysis.json`, `covariates.json`, `*_curve.json` |
| `e4/` | 관할 gate 오프라인 파일럿·scorer 표·재판정, E4-R 역량 지도, 작업공간 실측, novel 프레임 | `e4_scorer_table.json`, `e4r_competence_map.json`, `workspace_extent.json`, `fig_workspace_extent.png` |
| `e5/` | 온라인 lifecycle 스트림 3 seed × 4,000 ep, paired full-VLA replay, 사전등록 판독, 3-seed 종합 | `stream_{0,1,2}.jsonl`, `cf_{0,1,2}.jsonl`, `reading_{0,1,2}.json`, `seed_synthesis.json`, `fig_e5_s*_{behavior,mechanism}.png` |
| `e5/seed0_normstats_invalid/`, `e5/seed0_batchdata_invalid/`, `e5/smoke_run1_nocoverage/` | **무효화된 실행의 증거** (정규화 결함 등, 인용 금지 — log.md 2026-08-17·19) | — |
| `rgb_depth_ablation/` | depth privileged-information 스크리닝 Stage 1 (6 클러스터 × n{10,20,40,80} × held-out 50, paired) + 논문 서식 산출(`paper/`) | `ablation_summary.json`, `table_detail.csv`, `fig_{A,B,C}_*.png`, `paper/*.tex` |
| `rgb_only_full_rerun_20260828/` | RGB-only 전체 재실행 **데이터 패키지** (README_RESULTS.md · DATA_DICTIONARY.md 참조). 배치 27 클러스터, 온라인 3 seed, paired replay, familiarity, 레이턴시, 무결성, old-vs-new, 체크포인트 sha256 매니페스트, 실행 로그·마커·작업 원장 | `01_batch_formation/NSTAR_RESULTS.csv`, `derived/ONLINE_SUMMARY_ALL_SEEDS.json`, `05_paired_replay/PAIRED_REPLAY_SUMMARY.json`, `09_integrity/DATA_INTEGRITY_AUDIT.json`, `PACKAGE_VERIFICATION.json` |
| `videos/` | 롤아웃 영상 인덱스·매니페스트 (mp4 2.1 GB는 원본 `results/videos/`에만) | `index.json`, `manifest.json` |
| `figures/` | 위 그림들의 모음 (열람용 복사본) | |
| `reports/` | 판정 요청/보고 패키지의 문서 — 검토·회부·E4 갈래 판정/종결·E4-R·E5 설계·E5 판독(seed별)·원고 자료·v11 원고 패치·실물 로봇 인수인계·depth ablation 감사 — 및 원본 git 이력 | `GIT_HISTORY_original_repo.txt` |

## 3. 핵심 수치 (출처 파일 명시)

### 3.1 앵커 (E0/E1) — `e1/e1_sv.json`, `e1/e1_latency.json`
- teacher S_V: spatial 0.984 · object 0.984 · goal 0.996 · long(10) 0.968 (각 250 ep, Wilson CI는 파일 참조)
- 레이턴시 (RTX 5090, **attn=sdpa**): OFT chunk 85.07 ms · ACT 3.36 ms · gate 3.96 ms → per-chunk 25.3×, 보수 하한 11.6×

### 3.2 형성 H1 (E2/E3) — `e2/e2_gonogo.json`, `e3/e3_curves.json`
- E2 GO: object_task0 ŝ = 0.04/0.20/0.48/0.96 (n=10/20/40/80), object_task5 0.74/0.92/0.86/0.96
- E3 27 클러스터: N* 중앙값 object 10 · goal 20 · spatial 20; 우측절단 1 (long)

### 3.3 이중 해리 H2 (E3) — `e3/h2_analysis.json`
- 의미 레벨 L은 N*를 설명하지 못함 (between-share 0.0246, Kruskal p=0.772) → 경쟁 가설 **H2-L′ 채택**:
  운동·물리 난이도(median episode length β=3.721, perm p=0.0155)가 N*를 설명
- horizon 천장: T1 ŝ(80)=0.975 vs T3 0.8833, 단측 p=0.0968 (경계 — 방향만 서술)

### 3.4 게이트 H3 (E4/E4-R) — `e4/e4_scorer_table.json`, `e4/e4r_competence_map.json`
- 저비용 기하 관할은 임계(AUC 0.75) 미달, 히든 스테이트 gate가 우위이나 21× 비용 → "저비용 실시간 관할은 미해결"
- E5 그림자 관할 반사실: 사전 예측 +31.0%p·4.3× → 실측 +2.27%p·1.63× (닫힌 작업공간에서 관할 개입이 드묾)

### 3.5 시스템 상각 H4 (E5, RGB-D, 3 seed × 4,000 ep) — `e5/seed_synthesis.json`
- VLA 호출률 첫→끝 1,000 ep: **0.874±0.026 → 0.405±0.044** (H4a 3/3 PASS)
- 비열등 diff(margin −3%p): **−0.0021±0.0015**, paired 4,618 ep (H4b 3/3 PASS)
- Pr(fail|fire) **0.0285±0.0068** (ε=0.2), 성숙 도달 20.3±2.1 / 33 클러스터

### 3.6 depth ablation Stage 1 — `rgb_depth_ablation/ablation_summary.json`
- 6 클러스터 paired Δ(RGB − RGB-D) = **−0.015**, 95% CI [−0.039, +0.009] (0 포함), McNemar p=0.25
- n=80 평균 −3.67 pp (CASE B, long 1개가 −14 pp) → 연구원 결정: 전체 재실험

### 3.7 RGB-only 전체 재실행 — `rgb_only_full_rerun_20260828/derived/ONLINE_SUMMARY_ALL_SEEDS.json`, `05_paired_replay/PAIRED_REPLAY_SUMMARY.json`, `08_statistics/OLD_VS_NEW_NUMERIC.csv`
- 배치 27 클러스터 (2,640 ep): N* 동일 16/27, 형성셀 22 중앙값 15.0→10.0, 우측절단 1→1
- 온라인 3 seed (12,000 ep), RGB-D → RGB-only: r_V 0.6152±0.0190 → 0.5947±0.0180 · 시스템 성공 0.9608 → 0.9564 ·
  Pr(fail|fire) 0.0285 → 0.0377 · H4a 3/3 PASS
- H4b 전체스트림 −0.0065±0.0014 (3/3 PASS, margin의 1/5) · 발화집합만 Δ −0.0161±0.0036 (McNemar 3 seed 유의)
- 레이턴시(sdpa): ACT RGB-only 3.35 ms = RGB-D 3.35 ms · gate 4.18 · teacher 85.47
- 무결성 **VALID (47 검사, FAIL 0)** · 패키지-단독 숫자 복원 **PASS (52 검사)** · 실패 job 0 · 재시도 0

## 4. 제외된 것

체크포인트(192 GB), 궤적 HDF5(15 GB), OFT 가중치(60 GB), 롤아웃 mp4(2.1 GB), 무효 실행의 대용량 산출물,
E4 known 프레임(추적 안 됨). 복원 방법은 `../src/README.md` §4.
