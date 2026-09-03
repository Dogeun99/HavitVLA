# HabitVLA-2 — Amortized Inference via Habit Formation

> Claude Code가 매 세션 참조하는 프로젝트 컨텍스트.
> 원본 설계서 = "HabitVLA-2 최종 실험 설계서 v1.0" (2026-08-14, 사전등록 확정본).
> 본 문서는 그 설계서의 §0–§5 요약 + 작업 원칙이다. **수치의 원본은 `configs/preregistration.md`.**

---

## 0. 작업 환경 규칙 (공용 워크스테이션)

- 이 워크스테이션은 **학생 공용**이다. 모든 작업은 `~/workspace/habitvla2` + 전용 conda env 안으로 한정.
- 호스트 전역(전역 `apt`, `/etc`) 수정 금지. `sudo` 필요 명령은 **실행하지 말고 제안만**.
- **기존 env 불침범**: `habitvla`(Isaac), `phase1`(robosuite), `vla`, `vla_oft`는 건드리지 않는다.
  본 프로젝트는 신규 env 2개만 사용 (§9).
- **HF 캐시 격리**: `export HF_HOME=~/workspace/habitvla2/.hf_cache` 강제.
  전역 `~/.cache/huggingface`(43G, 공용)에는 **쓰지 않는다**. 기존 체크포인트 재사용은
  읽기 전용(symlink)만 허용.

### 하드웨어 / 스택
- GPU **RTX 5090 (Blackwell, sm_120)** 32GB VRAM / CPU Threadripper 9960X (24C/48T) / RAM 64GB
- Driver 580.159.03 · CUDA 13.0 (cu128 앱 하위호환 동작 확인됨)
- 디스크: `/` 1.8T, 여유 ~1.5T (2026-08-15 기준)
- Paper 1 교훈: **flash-attn은 sm_120에서 미빌드 → attention 구현은 `sdpa`**.
  모든 레이턴시 수치에 `attn=sdpa` 명기 + 구현 종속성 각주 필수.

---

## 1. 한 줄 주장

VLA 기반 로봇은 반복 경험을 통해 자신의 추론을 **상각(amortize)** 할 수 있다. 의미적으로 동일한
상황 클러스터에 VLA 성공 궤적이 축적되면 경량 습관 정책이 형성되고, **관할(익숙함)** 과
**성숙도(경험적 성공)** 의 2단 gate가 위험 보장 하에 VLA 호출을 선택적으로 생략한다.
→ 추론 비용은 배치 시간에 따라 감소하되 태스크 성공률은 유지된다.

**기여 3개**
1. **형성(Formation)** — VLA 성공 궤적만으로 클러스터별 경량 정책이 형성됨을 성숙 곡선 ŝ_k(n_k)로 실증.
2. **이중 해리(Double dissociation)** — 의미 복잡도 L은 형성 *속도*(N*(L) 우측 이동)를,
   horizon T는 도달 *천장*(compounding error O(εT²))을 지배함을 분리 실증.
3. **Lifecycle gating** — 관할+성숙도 2단 gate가 습관 생애주기(미지/미성숙/성숙)를 관리하며,
   온라인 스트림에서 VLA 호출률 감소와 시스템 성공률 비열등을 동시 달성.

**Paper 1과의 관계.** Paper 1 = *판별* 문제(성숙 완료된 고정 습관을 gate가 사전 판별하는가).
본 연구 = *형성* 문제(습관이 경험에서 형성되고 gate가 생애주기를 관리하는가).
Beta-Bernoulli·conformal 수학은 공유하되 **인용으로 처리**.

**캐싱과의 구분(related work 필수 단락).** 출력값 재사용(캐싱)이 아니라 **함수 학습(amortized
inference)**. VLA-Cache류(단일 에피소드 내 시각 토큰 재사용)와 층위가 다름 — 에피소드들에 *걸친*
정책 형성.

---

## 2. 가설과 반증 조건

| ID | 가설 | 반증 조건 |
|---|---|---|
| **H1** 형성 | 클러스터 k의 VLA 성공 궤적 n_k개로 학습한 π_k의 held-out 성공률 ŝ_k(n_k)가 n_k↑에 따라 상승, teacher 수준 근접 | n_k=80까지 미상승 또는 낮은 천장(<0.5) 정체 |
| **H2** 이중 해리 | L↑ → N*(L)↑ (우측 이동, 천장 유지) / T↑ → 천장↓ | L·T가 같은 방식으로 곡선을 변형하거나 효과 없음 |
| **H3** 2단 gate | 관할+성숙도 결합이 각 단독 대비 Pr(fail\|fire)↓ + coverage 유지 | 단독 대비 유의차 없음 |
| **H4** 시스템 상각 | 스트림에서 r_V(t) 감소 + 성공률이 full-VLA 대비 비열등(margin −3%p) | 호출률 감소가 성공률 붕괴 동반 |

---

## 3. 시스템 설계 요약

### 3.1 플랫폼 — LIBERO (robosuite/MuJoCo)
- 4 스위트(Spatial/Object/Goal/Long), 태스크당 언어 지시 고정.
- 관측: agentview RGB + wrist RGB + **depth(활성화)** + proprio.
- 성공 판정: LIBERO 공식 predicate (온라인 무료 라벨).
- **GT 클러스터 ID = 태스크 ID** (벤치마크 무료 제공).
- 초기상태: BDDL init 영역 + 공식 init state 파일. **변이 폭 파라미터화 가능 여부는 E0 확인 항목**.

### 3.2 Teacher — OpenVLA-OFT (공개 체크포인트)
- `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}`
- 2-image(3인칭+손목) + proprio, L1 regression head, action chunking, **greedy → 결정적 rollout**.
- 역할 3분기: (a) 탐험가(미지 처리·클러스터 신설) (b) 교사(성공 궤적 공급) (c) 대기(fallback).
- **히든 스테이트(L32) 추출은 비교 arm 전용** — 주 gate 경로에 VLA forward 없음(경량화 모순 회피).

### 3.3 습관 정책 — ACT (클러스터별 독립)
- (RGB-D, proprio) → action chunk. ACT 표준 구성.
- **전 단계(E5 포함) 클러스터별 독립 모델로 확정.** 공유 모델+클러스터 조건은 예비 부품.
- 학습 체크포인트 n_k ∈ {10, 20, 40, 80}. warm-start 허용.
- **HP 규율: C-L0(1 클러스터)에서만 튜닝, 이후 전 클러스터 동일 HP 동결.**

### 3.4 클러스터링 — 인수분해 2층
- **1층(태스크 정체성)**: 지시어 **정확 일치** → GT 태스크 ID 완전 정렬, 구현 비용 0.
  (패러프레이즈 강건성은 부록급 확장 — Paper 2로의 다리)
- **2층(태스크 내 관할)**: 성공 에피소드 **초기 특징** 분포 거리 판정.
  h(I₀) = DINOv2 ViT-S/14 → PCA(d=32), μ_k + Ledoit–Wolf 수축 공분산 Σ_k.

### 3.5 게이트 — 2단 + conformal
상황 x = (I₀, ℓ), 클러스터 배정 k(x) = task-id(ℓ).
- **1단 관할**: A_jur(x) = 𝟙[ d_M(h(I₀); μ_k, Σ_k) ≤ q_{k,1−α_j} ], α_j = 0.1 (calibration 분위수)
- **2단 성숙도**: A_mat(k) = 𝟙[ Pr(s_k ≥ τ | 𝒟_k) ≥ 1−δ ], s_k ~ Beta(1+σ_k, 1+φ_k), (τ,δ) = (0.8, 0.1)
- **발화**: g(x) = Habit ⟺ A_jur ∧ A_mat, 그 외 VLA.
- **lifecycle 3분기**: k 미존재 → 미지(VLA, 클러스터 신설) / A_mat=0 → 기지-미성숙(VLA, 𝒟_k·궤적 풀 축적)
  / A_jur ∧ A_mat → 발화.
- **정책 갱신 시 사후 재초기화**: 재학습 직후 Beta(1 + c·σ_k, 1 + c·φ_k), c = 0.25.
- **위험 통제**: Pr(fail|fire) ≤ ε = 0.2를 ACI(adaptive conformal)로 추적, 위반 시 τ 상향.
- **이중 장부**: 실패 에피소드는 𝒟_k(통계)엔 반영, BC 학습 풀에선 제외.

### 3.6 비교 arm (E4)
① gate 2×2 ablation {관할만/성숙도만/둘 다/없음} ② feasibility head(Paper 1 이월)
③ 히든 스테이트(L32) gate(비용 축 앵커) ④ oracle 관할(GT 태스크 ID + GT 초기상태 영역) = 상한

---

## 4. 실험 셀 (십자 설계) — E3 배치 대상 28 클러스터

| 셀 | 내용 | 클러스터 | 검증 |
|---|---|---|---|
| C-L0 | 단일 태스크 고정(Object 1개) | 1 | H1 기준점. **HP 튜닝 허용 유일 셀** |
| C-L1/2 | LIBERO-Object 전체 | 10 | 물체 선택·형상 |
| C-L3 | LIBERO-Goal 전체 | 10 | 목표 다양화 |
| C-L4a | Spatial 중 2태스크 고정 반복 | 2 | L4 존재 증명 |
| C-L4b | Spatial 나머지 | (8) | **E5 스트림 내 관측 전용**(배치 곡선 없음, novel 풀 겸용) |
| C-T2 | 단일 태스크 2연쇄(커스텀 래퍼) | 2–3 | 천장 하강(통제). 실패 시 Long 길이 층화로 대체 |
| C-T3 | LIBERO-Long 2–3태스크 | 2–3 | 천장 하강(생태 앵커) |

**D2(커스텀 자산) 기본 미사용** — 부족이 실측되면 그때만 커스텀 BDDL.

---

## 5. 프로토콜 요약

- **수집(E2/E3)**: 클러스터당 teacher rollout **120 ep**(초기상태=스트림 생성기, seed 고정)
  → 성공 궤적만 BC 학습(이중 장부).
- **평가**: **held-out 고정 초기상태**(학습 스트림과 분리, seed 고정, 체크포인트 간 동일 = paired).
  E2 = 50/클러스터, E3 = 20/클러스터.
- **성숙 정의(이원 보고)**: ① 클러스터 수준 N*(k) = min{n ∈ {10,20,40,80} : ŝ_k(n) ≥ 0.8}, 미도달 시 >80 우측절단
  ② 스트림 수준 성숙 소요 에피소드 수(재발률 × S_V 반영).
- **E5 온라인**: 스트림 2,000 ep × 3 seed, {C-L0, Object, Goal, Spatial-a} 혼합 + **10% novel 주입**(Spatial-b).
  **재학습 = 순차 배치(sequential interleaving)** — 트리거 시 스트림 일시정지, 학습 wall-clock 별도 계상.
  회계는 모두 **VLA-호출 등가**(E1 앵커). 관할 오류는 **양방향 로깅**(false accept = 위험, false reject = 효율).

### 단계별 go/no-go (요약, 원문 §5)
- **E0** 환경·depth·변이폭·스모크·wall-clock → **E1** S_V 재측정 + 레이턴시 앵커 5종
- **E2 ★ 유일 치명** — max_n ŝ ≥ 0.8 AND ŝ(80) > ŝ(10) (단측 two-proportion, α=0.05)
- **E3** 28 클러스터 곡선 → **E4** 관할 오프라인 파일럿(AUC ≥ 0.75) 선행 후 gate 비교
- **E5** r_V 감소 + 비열등(−3%p) → **E6** 다중 seed (5/5/3)

---

## 6. 작업 원칙 (Paper 1에서 이월 — 필수 준수)

- **문제 축소 금지 (★핵심).** 원래 사양을 임의로 단순화·축소하거나 검증 안 된 가정으로 우회 금지.
  구현이 어려우면 멈추고 "왜 어려운지" 보고. try/except로 에러 덮기, placeholder 대체,
  잘 되는 범위로 좁히기 **모두 금지**. 트레이드오프를 설명하고 연구원에게 물을 것.
- **사전등록 통계.** `configs/preregistration.md`의 수치는 동결. 변경 시 사유·일시 기록 필수.
- **수치 수동 입력 금지.** 모든 figure/표는 `results/`의 JSON/CSV 단일 진입점에서 프로그래밍 방식으로만 읽는다.
- **3자 워크플로우.** Claude Code = 실행 · 대화 Claude = 판정 · 연구원 = 결정.
- **판정 요청 = 패키지 자동 생성 (연구원 지시 2026-08-16).** 판정이 필요한 국면(갈래 판정,
  문구 확정, 설계 리뷰, 결과 해석 분기)에 도달하면 **요청을 기다리지 말고** 대화 Claude
  전달용 패키지를 즉시 생성한다. 구성: `<주제>_pack_<날짜>.tar.gz` = ①REPORT.md(수치는
  결과 JSON에서 **프로그래밍 주입**, 판정 요청 항목 명시) ②원자료 JSON ③관련 그림
  ④해당 스크립트 ⑤`configs/preregistration.md`·`log.md` 전문 ⑥git 이력. 생성기는
  `experiments/make_*_pack.py`로 남겨 재현 가능하게 하고, 패키지 자체는 gitignore한다.
- **장기 실행**: `python -u` + tmux(세션명 `habit2`). readiness 판정 = 프로세스 생존 + **명시적 PASS 마커**
  (로그 "error" grep 금지 — TF 배너 오탐 교훈).
- **진행/이슈 기록**: 루트 `log.md`에 계속 append (설계서 §9의 WORKLOG 역할).

---

## 7. 디렉토리

```
~/workspace/habitvla2/
├── CLAUDE.md                  # 본 문서
├── log.md                     # 진행/이슈 러닝 로그 (append-only)
├── configs/preregistration.md # §7 동결본 (수치 원본)
├── docs/                      # E0 지시서 등 작업 문서
├── envs/                      # LIBERO 래퍼(depth 노출), 스트림 생성기(변이 폭 파라미터)
├── teacher/                   # OFT 로드·rollout·궤적 수집
├── habits/                    # ACT 학습·체크포인트 (클러스터별)
├── gates/                     # 2층 클러스터링 + 2단 gate + conformal (Paper 1 수학 이월)
├── experiments/               # e0_smoke.py … e6_multiseed.py (figure 단일 진입점 포함)
├── results/                   # JSON/CSV 원자료 (figure는 여기서만 읽음)
└── logs/                      # 실행 로그 (tmux/nohup stdout)
```

## 8. 재사용 자산 (기존 워크스페이스)

- `~/workspace/habitvla/openvla-oft/` — OFT 레포 클론(2026-07). LIBERO eval 스크립트 포함.
- `~/workspace/habitvla/habitvla/gates/implementations.py` — Beta-Bernoulli / conformal / density gate 구현(Paper 1).
- `~/.cache/huggingface/` (공용, **읽기 전용 재사용만**) — `moojink/openvla-7b-oft-finetuned-libero-spatial`,
  `facebook/dinov2-small` 기보유.
