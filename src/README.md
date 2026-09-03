# HabitVLA-2 — 소스 코드 (릴리스 판)

**Amortized Inference via Habit Formation.** VLA(OpenVLA-OFT) 교사가 반복 상황에서 성공 궤적을 쌓으면
클러스터별 경량 습관 정책(ACT)이 형성되고, 관할(Mahalanobis)·성숙도(Beta-Bernoulli)·ACI 위험 통제의
2단 gate가 VLA 호출을 선택적으로 생략한다. 플랫폼은 LIBERO(robosuite/MuJoCo). 실험 프로토콜은
`configs/preregistration.md`(사전등록 동결본)에, 프로젝트 맥락은 `CLAUDE.md`에, 전 과정 진행·이슈 기록은
`log.md`(append-only, 2026-08-15 → 08-31)에 있다.

- 원본 저장소 최종 커밋 `5c12f9b` (2026-08-31). 릴리스에서 바꾼 것은 경로 이식성뿐 → `RELEASE_CHANGES.md`.
- 결과는 형제 폴더 `../results/` (이 폴더의 `results` 심볼릭 링크가 그것을 가리킨다).
- 검증 보고서: `../results/VERIFICATION_REPORT.md`.

## 1. 디렉터리

```
src/
├── build.sh                 원클릭 빌드
├── envs/                    LIBERO 래퍼(depth 노출)·결정적 스트림 생성기·2연쇄 래퍼 · setup_envs.sh
├── teacher/                 OpenVLA-OFT 로드·rollout·궤적 HDF5 수집 (이중 장부)
├── habits/                  ACT 정의·데이터셋·학습(n-grid warm-start)·held-out 평가·실행 정책
├── gates/                   DINOv2→PCA 특징, 관할/성숙도/ACI 2단 gate
├── experiments/             e0 … e5 단계 스크립트, 판정 패키지 생성기, 영상, rgb_depth_ablation/, rgb_only_rerun/
├── tools/                   tmux 체인·heartbeat 등 장기 실행 운영 스크립트
├── configs/                 preregistration.md(수치 원본), task_registry.json, third_party 로컬 패치 2개
├── docs/                    E0 지시서, E5 드라이버 설계/체크리스트, 원고 패치 지시, 실물 로봇 이식 가이드
├── setup/                   pip lock·conda yml·하드웨어·자산 링크 스크립트
├── verify/                  릴리스 검증 스크립트
├── third_party/             LIBERO(핀 8f1084e) · openvla-oft(핀 e4287e9) — git submodule + configs/*.patch
├── CLAUDE.md · log.md       프로젝트 컨텍스트 · 진행 로그
└── results -> ../results    (링크) checkpoints/ data/ .hf_cache/ .torch_cache/ 는 로컬 링크(미포함, §4)
```

## 2. 빌드

검증된 스택: Ubuntu, **RTX 5090 (sm_120)**, 드라이버 580.159 / CUDA 13.0, torch 2.7.0+cu128, MuJoCo 3.1.6,
robosuite 1.4.1, LIBERO 핀 커밋, flash-attn 미빌드 → **attention = sdpa**. conda env 2개를 쓴다.

| env | 역할 | 핵심 버전 |
|---|---|---|
| `hv2_oft` (py3.10) | teacher 추론 + 시뮬 (E1/E2/E3 수집, E5 스트림 드라이버, paired replay, 레이턴시) | transformers 4.40.1 moojink 포크, TF 2.15, timm 0.9.10 |
| `hv2_hab` (py3.11) | ACT 학습·평가, gate, 모든 분석 | transformers 5.15, scikit-learn, scipy, h5py |

```bash
git clone --recurse-submodules <이 저장소>   # third_party 서브모듈 포함
cd habitvla2_release/src
bash build.sh                                # = envs/setup_envs.sh (핀/패치 → env 2개 → import 검증)
#   env 이름을 바꾸거나 기존 env를 복제해서 만들려면:
#   OFT_ENV=hv2r_oft HAB_ENV=hv2r_hab CLONE_OFT_FROM=hv2_oft CLONE_HAB_FROM=hv2_hab bash build.sh
#   대용량 자산(체크포인트·HDF5·모델 캐시)이 있는 원본 디렉터리를 링크하려면:
#   ORIG=/home/asmr/workspace/habitvla2 bash build.sh
```
마지막 줄 `[E0-SETUP-DONE]`이 성공 마커다. 정확한 패키지 버전 전체는 `setup/*.requirements.lock`.
처음부터 만들 때는 torch(cu128) 다운로드 ≈ 3 GB, OFT 체크포인트 4종 ≈ 60 GB(`experiments/e0_download_ckpts.py`,
`HF_HOME=<src>/.hf_cache` 필수)가 필요하다.

## 3. 검증 (이 폴더만으로 저장 결과가 재현되는가)

```bash
HV2_HAB_PY=~/miniconda3/envs/hv2_hab/bin/python HV2_OFT_PY=~/miniconda3/envs/hv2_oft/bin/python \
ORIG=/home/asmr/workspace/habitvla2 bash verify/verify_release.sh          # --no-gpu 로 CPU 단계만
```
`verify_runs/<stamp>/`에 scratch 복제본을 만들어 (릴리스 results는 건드리지 않고) 다음을 수행한다:
단위 테스트 2종 → 패키지 자체 검증(원장→요약 재계산 52검사) → RGB-D 본 실험 분석 재산출(e2/e3/H2/E5 판독·
seed 종합·사후분석·E4 표) → RGB-only rerun 분석 재산출(batch/online/replay/old-vs-new) → 저장 결과와 대조 →
GPU(학습·추론 스모크, 체크포인트 held-out 재평가 2 클러스터를 에피소드별로 대조, teacher env 레이턴시 재측정,
체크포인트 전수 무결성 감사). 결과는 `SUMMARY.json`, `compare_cpu.json`, `compare_gpu.json`,
`latency_side_by_side.txt`. 2026-09-03 실행 결과 → `../results/VERIFICATION_REPORT.md`.

## 4. 포함되지 않은 것과 복원

| 항목 | 크기 | 복원 |
|---|---|---|
| `checkpoints/` (E2/E3 RGB-D 27 클러스터, E5 3 seed, depth ablation, RGB-only rerun 253개) | 192 GB | 원본 디렉터리 링크(`setup/link_local_assets.sh`) 또는 재학습(seed 결정적). sha256은 `../results/rgb_only_full_rerun_20260828/CHECKPOINT_MANIFEST.csv` |
| `data/` teacher 궤적·스트림 HDF5 | 15 GB | teacher 재수집 (`teacher/collector.py`, 에피소드 명세 결정적) |
| `.hf_cache/` OFT 체크포인트 4종 + DINOv2 | 60 GB | `experiments/e0_download_ckpts.py` |
| `logs/` | — | 실행 시 생성. RGB-only rerun 로그는 결과 패키지에 포함 |

## 5. 실행 예시 (모두 `src/`에서, `HF_HOME`·`LIBERO_CONFIG_PATH`는 스크립트가 기본값을 잡는다)

```bash
HAB=~/miniconda3/envs/hv2_hab/bin/python; OFT=~/miniconda3/envs/hv2_oft/bin/python
$HAB -u experiments/gate_regression.py                      # gate 회귀 테스트 → [GATE-REGRESSION-PASS]
$HAB -u experiments/executor_chunkbreak_test.py             # 실행기 chunk-break 단위 테스트
$HAB -u experiments/rgb_only_rerun/verify_package.py --package results/rgb_only_full_rerun_20260828
$HAB -u experiments/e3_collect.py && $HAB -u experiments/e3_h2_analysis.py     # E3 곡선·H2 분석 재산출
for s in 0 1 2; do $HAB -u experiments/e5_analyze.py --seed-idx $s; done; $HAB -u experiments/e5_seed_synthesis.py
# 체크포인트 held-out 평가 (paired, 결정적)
$HAB -u habits/evaluate.py --cluster libero_goal_task1 --suite libero_goal --task 1 \
     --ckpt-dir checkpoints/rgb_only_rerun/batch/libero_goal_task1 --n-grid 10 20 40 80 --n-heldout 20 --out /tmp/eval
# 습관 학습 (n-grid warm-start 체인; --no-depth = RGB-only)
$HAB -u habits/train.py --h5 data/e3/libero_goal_task1.hdf5 --cluster libero_goal_task1 --n-grid 10 20 40 80 --out /tmp/ck
# teacher 궤적 수집 (hv2_oft)
$OFT -u teacher/collector.py --suite libero_goal --task 1 --n 120 --out data/e3
# E5 온라인 스트림 (hv2_oft, 4,000 ep ≈ 16 h/seed) + paired full-VLA replay
$OFT -u experiments/e5_driver.py --seed-idx 0 --n 4000 [--no-depth --out-root … --ck-root … --data-root …]
$OFT -u experiments/e5_counterfactual.py --seed-idx 0
# RGB-only 전체 재실행 파이프라인 (marker 기반 resume; 완료 stage는 건너뜀)
$HAB -u experiments/rgb_only_rerun/preflight.py && $HAB -u experiments/rgb_only_rerun/smoke.py && $HAB -u experiments/rgb_only_rerun/run_all.py
```

환경변수: `HV2_HAB_PY`/`HV2_OFT_PY`(두 env의 python), `HABIT2`(셸 스크립트의 루트 재지정), `HF_HOME`, `TORCH_HOME`,
`LIBERO_CONFIG_PATH`(기본 `<src>/.libero`), `MUJOCO_GL=egl`.

## 6. 모듈 인덱스

| 파일 | 역할 |
|---|---|
| `envs/libero_env.py` | LIBERO 래퍼 — depth 노출, `EpisodeSpec`(suite, task, base init, 섭동 폭 w, seed)로 결정적 초기상태 |
| `envs/stream.py` | 수집/held-out/novel/probe 스트림 명세 생성기 (seed 대역 6종 disjoint 보장) |
| `envs/chained_env.py` | C-T2 단일 태스크 2연쇄 래퍼 + `execute_chunk_with_boundary`(stage 전환 시 stale chunk 폐기) |
| `teacher/collector.py` | OFT 로드(`load_teacher`)·rollout·성공 궤적만 HDF5 저장 (이중 장부) |
| `habits/act.py` · `dataset.py` · `train.py` | ACT 정책, HDF5→학습셋(정규화 통계 자기 풀), n-grid {10,20,40,80} warm-start 학습, `--no-depth` |
| `habits/policy.py` · `evaluate.py` | 체크포인트 실행 정책(K=8 chunk), held-out 성숙 곡선 평가(paired) |
| `gates/features.py` · `two_stage.py` | DINOv2 ViT-S/14→PCA(32), `JurisdictionGate`(Mahalanobis, α_j=0.1) · `MaturityGate`(Beta, τ=0.8, δ=0.1, c=0.25) · `ACIRiskController`(ε=0.2) |
| `experiments/e0_*` | 환경·depth·변이 폭·스모크·wall-clock 검증 |
| `experiments/e1_*` | teacher S_V 재측정(1,000 ep), 레이턴시 앵커 5종 |
| `experiments/e2_*`, `e3_*` | 형성 곡선 수집·집계(27 클러스터), C-T2 검증/진단, H2 분석(`e3_h2_analysis.py`) |
| `experiments/e4_*`, `e4r_*` | 관할 gate 오프라인 파일럿·scorer 비교·재판정, E4-R 역량 지도·teacher w-사다리 |
| `experiments/e5_driver.py` · `e5_counterfactual.py` · `e5_analyze.py` · `e5_seed_synthesis.py` | 온라인 lifecycle 스트림, paired full-VLA replay, 사전등록 판독기, 3-seed 종합 |
| `experiments/rgb_depth_ablation/` | depth privileged-information 스크리닝 (6 클러스터, paired) + 논문 서식 산출 |
| `experiments/rgb_only_rerun/` | RGB-only 전체 재실행: runner(resume/retry/원장) · preflight · smoke · run_batch · run_all · analyze_{batch,online,replay,familiarity} · measure_latency · integrity_audit · old_vs_new · make_package · verify_package · watchdog · status |
| `experiments/make_*_pack.py` | 판정 패키지 생성기 (수치는 results JSON에서 프로그래밍 주입) |
| `experiments/video_*` | 롤아웃 영상 매니페스트·녹화·렌더 |
| `tools/` | tmux 체인(seed 연쇄 실행), heartbeat, 상태 조회 |
