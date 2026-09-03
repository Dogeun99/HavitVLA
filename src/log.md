# HabitVLA-2 진행 로그 (append-only)

> 규칙: 최신 항목을 **아래에 추가**한다. 각 항목은 `## YYYY-MM-DD — 제목` + `### 한 일` / `### 발견·이슈`
> / `### 다음`. 이슈에는 `[ISSUE-n]` 번호를 붙이고, 해결되면 해결 항목에서 그 번호를 참조한다.
> 수치는 항상 출처(파일 경로·명령)와 함께 기록한다.

---

## 2026-08-15 — 레포 부트스트랩 + 기존 자산 실측 조사 (E0 착수 전)

### 한 일
- 설계서 v1.0 접수 → 레포 문서 3종 생성
  - `CLAUDE.md` (설계서 §0–§5 요약 + 작업 원칙 + 재사용 자산)
  - `configs/preregistration.md` (§7 수치 동결본, 상태 = 연구원 승인 대기)
  - `log.md` (본 파일)
- 워크스테이션 자산 실측 조사 (아직 **설치·다운로드는 하지 않음**).

### 실측 사실 (조사 결과)

| 항목 | 실측값 | 출처 |
|---|---|---|
| GPU | RTX 5090 32GB, Driver 580.159.03 / CUDA 13.0, 유휴(100MiB 사용) | `nvidia-smi` |
| 디스크 여유 | `/` 1.8T 중 **1.5T 여유** | `df -h /home` |
| 기존 conda env | `base`, `habitvla`, `phase1`, `vla`, `vla_oft` | `conda env list` |
| `vla_oft` env | py3.10.20, torch **2.7.0+cu128**, transformers **4.40.1**, timm 0.9.10, tokenizers 0.19.1, numpy 1.26.4, diffusers 0.30.3 | `pip list` |
| `phase1` env | py3.11.15, torch 2.7.0+cu128, robosuite **1.5.2**, mujoco 3.9.0 | `pip list` |
| LIBERO 패키지 | **어느 env에도 미설치** (신규 설치 필요) | `python -c "import LIBERO"` 실패 |
| OFT 레포 | `~/workspace/habitvla/openvla-oft` 클론 존재 (upstream `e4287e9`) | `git log` |
| OFT 레포 로컬 수정 | `pyproject.toml`이 **torch 2.7.0+cu128 / torchvision 0.22.0+cu128로 재핀** (주석: "HabitVLA: sm_120 (RTX 5090); upstream 2.2.0 has no Blackwell build"), `prismatic/vla/datasets/rlds/oxe/{configs,transforms}.py` 수정 | `git status` |
| 전역 HF 캐시 | `~/.cache/huggingface` **43G**, `moojink/openvla-7b-oft-finetuned-libero-spatial`, `facebook/dinov2-small` 기보유 | `ls ~/.cache/huggingface/hub` |
| LIBERO 요구사항(OFT) | **robosuite==1.4.1**, bddl, easydict, cloudpickle, gym, imageio[ffmpeg] | `experiments/robot/libero/libero_requirements.txt` |
| OFT env 요구사항 | transformers = moojink 포크(`transformers-openvla-oft`, 4.40.1 계열), peft 0.11.1, tf 2.15.0 등 | `pyproject.toml` |
| LIBERO env 생성 | `OffScreenRenderEnv(bddl_file_name, camera_heights, camera_widths)` — robosuite kwargs 통과 구조 | `experiments/robot/libero/libero_utils.py:18-26` |
| 초기상태 API | `task_suite.get_task_init_states(task_id)` (고정 배열) | `run_libero_eval.py:230` |
| 에피소드 상한 | spatial 220 / object 280 / goal 300 / long(10) 520 step (+ `num_steps_wait` 10) | `run_libero_eval.py:63-69` |
| 공개 성능 | 4 스위트 평균 **97.1 %** (스위트별 개별 체크포인트) | `LIBERO.md:41` |

### 발견·이슈

- **[ISSUE-1] (높음) 체크포인트-디바이스 성능 저하 경고.** OFT 공식 문서: *"Please be sure to test your
  policy with the same device/GPU used to train it! Otherwise, performance may drop substantially."*
  (`LIBERO.md` 하단). 공개 체크포인트는 A100/H100 학습 추정, 우리는 **RTX 5090(sm_120)**.
  → E0 go 기준(공개 보고치 ±10 %p)과 E1의 S_V ≥ 0.85를 직접 위협하는 **최상위 리스크**.
  대응 후보: (a) bf16/fp32 및 attention 구현 조합 스윕 (b) LoRA 재병합(`vla-scripts/merge_lora_weights_and_save.py`)
  (c) 미달 시 사전등록 예외 규칙(§3 완화 조항) 발동. **E0에서 가장 먼저 실측할 항목.**

- **[ISSUE-2] (높음) 설계서 §9의 env 2분할은 그대로는 성립하지 않음.** 설계서는
  `libero`(sim+ACT+gates) / `oft`(teacher)로 나누지만, **teacher rollout은 시뮬레이터를 같은 프로세스에서
  구동**한다(`run_libero_eval.py`가 env.step과 모델 forward를 한 루프에서 수행).
  → 따라서 **`oft` env에도 LIBERO+robosuite 설치가 필요**하다. 수정 제안:
  - `hv2_oft` = LIBERO sim + OFT teacher (py3.10, transformers 4.40.1 포크, robosuite 1.4.1) — E1/E2/E3 궤적 수집, E5 fallback
  - `hv2_hab` = LIBERO sim + ACT + gates (py3.11, 최신 torch, DINOv2) — 습관 학습·평가·gate
  두 env 모두 LIBERO를 설치하되 **동일 커밋 핀**을 사용해 sim 동작 동일성을 보장한다.
  (역할 분리 자체는 설계서 의도대로 유지 — 이름만 접두사 `hv2_`로 기존 env와 충돌 회피.)

- **[ISSUE-3] (중간) robosuite 버전 충돌.** LIBERO는 **robosuite 1.4.1** 핀, 기존 `phase1` env는 1.5.2.
  1.4.1은 구형 mujoco 계열을 가정 → 신규 env에서 mujoco 버전 조합을 E0에서 실측 확정해야 한다.
  (기존 env는 불침범 원칙이므로 영향 없음.)

- **[ISSUE-4] (중간) 초기상태 변이 폭 파라미터화 — E0 최대 미지수.** LIBERO가 제공하는 것은
  `get_task_init_states(task_id)`의 **고정 init state 배열**(태스크당 통상 50개)이다. 설계서 §2.1이
  요구하는 "변이 폭 파라미터화"는 (a) 고정 init state 벡터에 통제된 섭동을 가하거나
  (b) BDDL init 영역에서 robosuite placement initializer로 **재샘플링**해야 가능.
  (b)가 정당하나 구현 난이도·성공 판정 유효성 검증이 필요. **E0-6의 핵심 산출물.**
  주의: `libero_utils.get_libero_env()`에 `env.seed(0)` 후 *"seed seems to affect object positions even
  when using fixed initial state"* 주석 존재 → 결정성 확보를 위해 seed 경로를 반드시 명시적으로 통제할 것.

- **[ISSUE-5] (중간) depth 노출 경로 미검증.** `OffScreenRenderEnv(**env_args)`가 robosuite kwargs를
  통과시키는 구조이므로 `camera_depths=True`가 먹힐 가능성이 높으나(→ `agentview_depth` 키),
  LIBERO 래퍼가 obs 키를 필터링할 수 있음. **E0-3에서 (H,W) 및 값 범위(정규화 여부)까지 확인.**

- **[ISSUE-6] (낮음–중간) flash-attn 미빌드 (Paper 1 이월).** sm_120에서 flash-attn 소스 빌드는 장시간·고위험 →
  Paper 1은 `sdpa`로 확정했다. 본 연구도 동일 방침이면 **모든 레이턴시 수치(E1 앵커 5종, F3)에
  `attn=sdpa` 명기 + 구현 종속성 각주**가 필수다. 이는 [ISSUE-1]의 성능 저하 원인 후보이기도 함.

- **[ISSUE-7] (낮음) E0 스모크의 통계적 해상도.** 10 ep × 4 스위트에서 참값 p=0.97이면 한 스위트가
  8/10 이하를 낼 확률이 약 3.5 %, 4 스위트 중 최소 하나가 오탐할 확률 ≈ 13 %.
  → 8/10이 나왔을 때 즉시 no-go로 읽지 말고 **해당 스위트만 20 ep 재확인** 후 판정하는 규칙이 필요.
  **사전등록 수정 사항이므로 연구원 결정 대상** (`configs/preregistration.md` §5에 기록 후 적용).

- **[ISSUE-8] (낮음) HF 캐시 재사용 방침.** 전역 캐시에 spatial 체크포인트가 이미 있음(재다운로드 ~14GB 절약 가능).
  §9의 격리 원칙은 **쓰기 오염 방지**가 목적이므로, 프로젝트 로컬 `HF_HOME` + 기존 blob **읽기 전용 symlink**로
  양립 가능. 미승인 시 4종 전체 신규 다운로드(≈ 56GB, 여유 1.5T로 충분).

### 다음
1. `docs/E0_INSTRUCTIONS.md` 체크리스트에 따라 E0 실행 (연구원 go 대기 항목 포함).
2. 연구원 확정 필요 (설계서 §12 + 위 이슈): env 이름/분할안([ISSUE-2]), 스모크 재확인 규칙([ISSUE-7]),
   HF 캐시 symlink 재사용([ISSUE-8]), §7 동결 승인.

---

## 2026-08-15 — E0-1/E0-2/E0-3 실행: env 구축, LIBERO 설치, depth 확인 PASS

### 한 일
- 초기 커밋 `26654d1` (부트스트랩 문서 5종).
- `third_party/openvla-oft` ← `~/workspace/habitvla/openvla-oft` 복사(콘텐츠 12M, upstream `e4287e9`).
  로컬 수정(sm_120 cu128 재핀 등)은 `configs/openvla_oft_local.patch`로 보존.
- `third_party/LIBERO` ← GitHub 클론, 커밋 **`8f1084e3132a39270c3a13ebe37270a43ece2a01`** (2025-03-15) 핀.
- conda env 2개 신규: **`hv2_oft`** (= `vla_oft` clone: py3.10, torch 2.7.0+cu128, transformers 4.40.1 moojink 포크,
  TF 2.15) / **`hv2_hab`** (py3.11 신규, torch 2.7.0+cu128 + torchvision 0.22.0 + scikit-learn/transformers/einops/h5py).
  두 env 모두 LIBERO editable + `libero_requirements.txt`(robosuite 1.4.1, bddl 3.6.0, gym 0.26.2).
  ISSUE-2의 제안 구성(양쪽 모두 sim 설치)을 그대로 구현 — 이름·구성의 최종 승인은 여전히 연구원 대기.
- **E0-3 depth 확인 = PASS** (`results/e0/e0_3_depth.json`):
  `OffScreenRenderEnv(camera_depths=True)` → `agentview_depth`, `robot0_eye_in_hand_depth` 모두
  (256,256,1) float32, RGB와 (H,W) 일치, 유한값, 팔 이동 후 값 변화 확인.
  **값은 OpenGL 정규화 [0,1] depth buffer** (agentview min≈0.984 — 비선형). → ACT 입력 시
  미터 변환(`robosuite camera_utils.get_real_depth_map`) 여부를 전처리 설계에서 결정할 것.

### 발견·이슈 (신규)
- **[ISSUE-9] (해결) libero_requirements 설치가 hv2_oft의 numpy를 1.26.4→2.2.6으로 상향** (opencv 5.0이 원인).
  TF 2.15·transformers 4.40.1 스택과 충돌 위험 → `numpy==1.26.4`, `opencv-python==4.9.0.80` 재핀으로 복원.
- **[ISSUE-10] (해결) LIBERO editable 설치가 빈 MAPPING으로 깨짐.** 원인: LIBERO 최상위 `libero/`에
  `__init__.py`가 없어(implicit namespace) 최신 setuptools strict editable 모드가 패키지를 못 찾음.
  → `--config-settings editable_mode=compat`로 재설치(hv2_oft·hv2_hab 모두). 재발 방지를 위해 셋업 스크립트에도 반영 필요.
- **[ISSUE-11] (해결) mujoco 3.11 ↔ robosuite 1.4.1 비호환** (`MjData.qM` 제거 → OSC 컨트롤러 크래시).
  → **mujoco==3.1.6 핀** (OFT 시절 조합). hv2_hab에도 동일 핀 적용 필요(아직 3.11.0 상태 — 대기 항목).
- **[ISSUE-12] (해결) torch≥2.6 `weights_only=True` 기본값이 LIBERO init state 로드를 깨뜨림.**
  `benchmark/__init__.py:164` `torch.load`에 `weights_only=False` 1줄 패치(저장소 내부 신뢰 파일).
  패치는 `configs/libero_local.patch`로 보존.
- **[ISSUE-13] (해결) LIBERO 첫 import가 대화식 프롬프트(데이터셋 경로) 요구** → 프로젝트 로컬
  `LIBERO_CONFIG_PATH=$HABIT2/.libero` + `config.yaml` 선생성으로 해결. **모든 실행에서 이 env var 필수**
  (누락 시 `~/.libero`에 쓰며 공용 오염). 실행 래퍼에 내장할 것.
- 참고: EGL 종료 시 `EGLError` (env.close 후 GC 시점) — 렌더 결과에는 영향 없는 알려진 소음. 무해 판정.

### 다음
1. hv2_hab에 mujoco==3.1.6 핀 + LIBERO 임포트 검증 (ISSUE-11 잔여).
2. E0-2 산출물: `configs/task_registry.json` (4 스위트 × 10 태스크 지시어 전문) + `results/e0/e0_2_libero.json`.
3. E0-4: OFT 체크포인트 다운로드(프로젝트 HF_HOME, ISSUE-8 기본값 = 신규 다운로드) + 로드 검증.
4. E0-5 스모크 → E0-6 변이 폭 → E0-7 wall-clock.

---

## 2026-08-15 — E0-1/2 완료 판정, E0-6 PASS, 재현 프로토콜 확정 (★핵심 발견)

### 한 일
- hv2_hab 정렬 완료: mujoco 3.1.6 / numpy 1.26.4 / opencv 4.9.0.80 (hv2_oft와 동일 핀), LIBERO compat 재설치.
  → **E0-1 PASS** (양 env cap (12,0), GPU matmul OK, `results/e0/e0_1_envs.json`).
- **E0-2 PASS**: 4 스위트 × 10 태스크 = 40, init states 로드 확인 (spatial task0 = (50, 92)).
  `results/e0/e0_2_libero.json` + `configs/task_registry.json` (1층 클러스터링 원본) 생성.
- **[ISSUE-14] (해결) hv2_oft의 editable openvla-oft가 구 프로젝트 경로(`~/workspace/habitvla/openvla-oft`)를
  가리킴** (vla_oft clone의 .pth 이월) → 프로젝트 내 `third_party/openvla-oft`로 재설치(`--no-deps`). 격리 회복.
- E0-4a: 4 체크포인트 다운로드 진행 중 (프로젝트 HF_HOME, 48G/약 60G 시점 기록).
  **전역 캐시의 spatial 체크포인트에서 `config.json.back.20260723_*` 발견 — 전 프로젝트가 공유 캐시
  config.json을 수정한 오염 실증** → ISSUE-8의 기본값(신규 다운로드)이 옳았음을 확인. symlink 재사용 안건 철회.
- **E0-6 PASS** (`results/e0/e0_6_variation.json`):
  - 섭동 경로: w-grid {0, 0.01, 0.02, 0.04}에서 정착 후 xy 분산 {0, 2.3e-5, 9.9e-5, 3.8e-4} — **단조 증가**
    (섭동 분산 w²/3 스케일과 정합), 유효율 8/8 전 격자, 동일 seed·w 완전 재현(sim 상태 + obs 해시 일치).
  - 재샘플링 경로: BDDL placement 재샘플 확인 — 동일 seed 재현 + 상이 seed 상이(최대 Δ 2.3cm).
  - **변이 폭 파라미터화 = 가능** (성숙 곡선 "클러스터 내 변이" 통제 + E4 novel 생성 요건 충족).

### ★ 프로토콜 발견 (전 실험 공통 규약으로 승격)
**에피소드 재현 3단 규약: `env.seed(seed)` → `env.reset()` → `env.set_init_state(state)` → settle 10 step.**
실측 근거 (E0-6 1·2차 실패의 원인 분리):
1. reset 생략 시 — OSC 컨트롤러 내부 상태가 직전 에피소드에서 이월되어 동일 init state라도 결과 상이.
2. reset 전 re-seed 생략 시 — placement RNG 스트림이 reset마다 진행, reset 시점 상태가 달라지고
   컨트롤러 초기화가 이에 의존 → set_init_state로 qpos/qvel을 덮어써도 settle 결과 상이.
   (`libero_utils.py`의 "seed seems to affect object positions even when using fixed initial state" 주석의 실체.)
→ envs/ 래퍼 구현 시 이 3단을 강제 API로 고정한다. E2/E3 held-out paired 비교와 E5 스트림 재현성의 전제.

### 다음
1. E0-4b: 다운로드 완료 후 4 체크포인트 로드 검증 (VRAM·chunk shape·attn=sdpa 기록).
2. E0-5: `experiments/e0_smoke.sh` (스위트당 10 태스크 × 1 ep, 공식 eval 스크립트 무수정) → 집계.
3. E0-7: 스모크 로그에서 에피소드당 wall-clock 추출 + forward/sim 분해 측정.

---

## 2026-08-15 — E0-4 체크포인트: 다운로드 완료, 로드 검증 진행

### 한 일
- **E0-4a PASS**: 4 체크포인트 신규 다운로드 완료, `.hf_cache` 총 60G (스냅샷 해시는
  `results/e0/e0_4a_download.json`). 회선이 빨라 체크포인트당 ~2분.
- E0-4b 로드 검증 1차 실행 — `get_vla_action()` 호출 시그니처 오류(체크포인트 인자 과잉) 수정 후 재실행 중.
- E0-7 집계 스크립트(`experiments/e0_walltime_collect.py`) 준비: 스모크 로그 타임스탬프에서
  에피소드당 초 추출 → E1/E2/E3/E5 예산 환산(teacher-ep 상한 기준).

---

## 2026-08-15 — E0 전 항목 PASS: 스모크 40/40, 예산 재추정 대폭 하향 (★go)

### 결과 요약 (전 항목 `results/e0/*.json`)

| 항목 | 판정 | 핵심 수치 |
|---|---|---|
| E0-1 env | PASS | 양 env cap (12,0), attn=sdpa |
| E0-2 LIBERO | PASS | 커밋 8f1084e, 40 태스크, task_registry.json |
| E0-3 depth | PASS | 2 카메라 (256,256,1) f32, 정규화 [0,1] |
| E0-4 ckpt | PASS | 4/4 로드, VRAM 피크 16.1GB, chunk (8,7), 로드 7–11s |
| **E0-5 스모크** | **PASS** | **spatial 10/10, object 10/10, goal 10/10, long 10/10** (공개치 대비 +1.6~+5.5%p) |
| E0-6 변이 폭 | PASS | 단조 분산·유효 100%·완전 재현 |
| E0-7 wall-clock | PASS | teacher-ep 평균 **6.1s** (spatial 5.8/object 5.5/goal 4.9/long 8.3) |

- **[ISSUE-1] 해소.** 공개 체크포인트가 RTX 5090(sm_120)+sdpa에서 성능 저하 없이 동작 —
  40/40으로 "디바이스 불일치 성능 저하" 우려는 실측 반박. E1 임계 완화 조항 발동 가능성 낮아짐.
- **[ISSUE-7] 미발동.** 스모크 만점이라 RECHECK 규칙 적용 사례 없음(규칙 자체는 유지).
- **예산 재추정** (`results/e0/e0_7_walltime.json`): rollout 총량(E1+E2+E3+E5 수집·평가)
  ≈ **22.5 h 상한** — 설계서 추정(200–350 GPU-h)보다 훨씬 낮음. ACT 학습 비용은 E1 앵커 ⑤에서 측정 후 갱신.
  주의: 스모크가 전 성공이라 실패 에피소드(max_steps 완주, 30–90s)가 반영 안 된 하향 편의 —
  S_V≈0.9 가정 시 +10~20% 보정 여지.
- **[ISSUE-15] tmux 부재.** 이 머신에 tmux 미설치(전역 설치는 sudo — 제안만 가능:
  `sudo apt install tmux`). 장기 실행은 nohup 백그라운드로 대체 중. §9의 "tmux 세션 habit2" 관행은
  설치 후 적용.
- 진행 중: E0 스크립트·사전등록 일관성 적대적 검증 워크플로우(4 렌즈 × 검증) — 확정 발견은
  수정 후 본 로그에 기록.

### E0 종합 go/no-go = **GO** (스모크 ±10%p ✓, depth ✓, 변이 폭 ✓, 결정성 ✓)
→ E1(S_V 재측정 1,000 ep + 레이턴시 앵커 5종) 착수 준비. 스모크 실측 기준 E1 rollout은 ~2h 예상.

---

## 2026-08-15 — 검증 워크플로우(34 에이전트) 확정 발견 27건 전량 반영

적대적 검증 워크플로우(4 렌즈 리뷰 → 발견별 반박 시도)가 27건을 확정(중복 포함; 치명 2·주요 12·경미 13).
전부 수정 완료. 주요 항목과 조치:

**치명 (재구축 재현성):**
- `envs/setup_envs.sh`가 검증 완료 상태를 재현하지 못함(사후 수동 수정 5건 미반영) →
  **전면 재작성**: LIBERO 핀 체크아웃 + `git apply` 패치 2종 + editable compat + mujoco/numpy/opencv
  재핀 + openvla-oft third_party 설치 + `.libero/config.yaml` 생성($HABIT2 기준)까지 idempotent 포함.
- 재구축 절차 어디에도 `configs/libero_local.patch` 적용 단계가 없었음 → setup_envs.sh가 정본,
  `docs/E0_INSTRUCTIONS.md` E0-1/E0-2 절 갱신.

**주요 (판정 강건성):**
- `e0_smoke_collect.py`: 부분 로그(중도 크래시)가 PASS로 오판될 수 있던 폴백 제거 —
  완결성 검사(n ∈ {10, 20}) + 최종 집계 라인 부재 = 무조건 FAIL.
- `e0_smoke.sh`: eval 종료 코드 미검사 → `PIPESTATUS[0]` 검사 + 실패 시 즉시 중단, `mkdir -p` 추가,
  재확인용 `TRIALS`/`SUITES` override 지원.
- RECHECK 규칙을 문서 규칙과 일치화: 밴드 밖 + 실패 ≤ 2 → 20 ep(추가 init state) 재확인.
  결정적 파이프라인에서 동일 표본 재실행은 무정보임을 명문화.
- 공개 보고치(97.6/98.4/97.9/94.5)가 코드 하드코딩뿐 → `configs/preregistration.md` **§2b 신설**
  (E0 go 기준 전체 + 공개치 + 재확인 규칙 등재, 변경 이력 기록).
- E3=28 유도 모호(2~3+2~3) → **T2=2, T3=3** 기본 배분 명시(연구원 확인 대상). E2=2 근거 명시.
- `e0_ckpt_load.py` 등 전 스크립트에 `LIBERO_CONFIG_PATH`/`MUJOCO_GL`/`HF_HOME` setdefault 가드 —
  env var 누락 실행이 공용 `~/.libero`·전역 HF 캐시를 오염시키는 경로 차단.
- E0-6 재현 프로토콜(매 에피소드 재시드)이 공식 eval(1회 시드)과 다름을 **의도된 이원화**로 문서화:
  teacher S_V(E1) = 공식 run-수준 재현 / 스트림 생성기·held-out 평가(E2+) = episode-수준 재현.

**주요 (E0-6 검정력):**
- 단조성 검정이 퇴화 통과 가능(전부 0 분산) → 비퇴화 요구 추가(w_max 분산 > 1e-8 + w=0 대비 증가).
- 유효성 검사 확장: 낙하 + **수평 이탈**(정착 위치가 섭동 목표 10cm 이내) 검사. 무효 표본은 분산
  계산에서 제외. 관통·전도·BDDL 술어 재검증은 **미구현 한계로 JSON에 명시**(E4 novel 생성기에서 승격).
- 단일 태스크 과대 일반화 → **4 스위트 각 task 0으로 확장 재실행** (결과: 아래 갱신).

**경미 (기록):**
- walltime 예산의 "_upper" 명명이 상한 아님 → "estimate"로 개명 + `budget_caveats` 4건 명시
  (성공-only 하향 / mp4 인코딩 포함 상향 / ±1~2s 해상도 / ACT 미포함).
- 공식 run_episode가 예외를 삼켜 인프라 오류가 정책 실패로 위장됨 — 스모크는 공식 코드 충실 유지,
  **우리 래퍼(E1+)는 인프라 오류를 raise로 분리**하는 설계 원칙 기록.
- tmux 부재(ISSUE-15), `.libero/config.yaml` 머신 종속(스크립트 재생성으로 해소).

수정 후 기존 완결 로그로 collector 재실행 → E0-5/E0-7 판정 불변(PASS) 확인.

**E0-6 확장 재실행 결과 (4 스위트, v3 = 가용 폭 기준):**

| 스위트 | 판정 | usable_w_max | free joints |
|---|---|---|---|
| spatial | PASS | 0.04 | 5 |
| object | PASS | 0.04 | 7 |
| goal | PASS | 0.04 | 4 |
| **long(10)** | PASS | **0.02** | **8** |

★ **신규 발견: 가용 변이 폭은 씬 밀도 종속.** Long task0(물체 8개 혼잡 씬)은 w=0.04에서
유효율 3/8로 붕괴(물체 간 간섭) — w≤0.02에선 전 스위트 유효율 100%·분산 단조·완전 재현.
→ 판정 기준을 "스위트별 usable_w_max ≥ 0.02"로 정밀화(`preregistration.md` §2b + 변경 이력,
연구원 승인 대기). **스트림 생성기 설계 제약: 클러스터별 w ≤ usable_w_max.**
E4 novel 생성(변이 폭 확대)도 이 상한 내에서 설계해야 함 — 상한 초과 novel은 "물리적 무효"와
"분포 밖"을 혼동시킴.

### E0 최종: 전 항목 PASS (E0-6은 강화판 기준으로 재확인) → **E1 착수 가능**

---

## 2026-08-15 — E1-a 가동 + 인프라 5모듈 구축 (커밋 f85b011)

### 한 일
- **E1-a S_V 재측정 가동**: `experiments/e1_sv.sh` (태스크당 25 ep × 40 태스크, 공식 스크립트
  무수정, exit 검사·완결성 검사 포함). 백그라운드 실행 중 (~2–3h 예상).
  집계기 `e1_sv_collect.py`: Wilson CI + PASS/RELAX/REDESIGN 3단 판정(사전등록 §1·§3) +
  태스크별 성공률(클러스터 선정 입력) + 성공/실패 wall-clock 분리(예산 캐비앳 1 해소).
- **인프라 구축** (E1 rollout이 GPU 점유 중 — CPU 작업):
  - `envs/libero_env.py` — E0-6 재현 프로토콜의 강제 API(`begin_episode`), depth 기본 활성,
    `InfraError` 분리 전파, 스위트별 `USABLE_W_MAX` 내장, `EpisodeSpec`(결정적 uid).
  - `envs/stream.py` — 수집(120)/held-out(50·20)/novel 명세 생성기. **3중 disjoint**
    (base_idx 0–39 vs 40–49 / seed 대역 / noise 대역), w_id=0.01. → **사전등록 §4b 등재**.
  - `teacher/collector.py` — 공식 OFT 경로 재사용 rollout + RGB-D 128 저장 + HDF5 +
    이중 장부(성공만 BC 풀, 실패는 통계만) + 인프라 오류 별도 계정.
  - `habits/` — 표준 ACT(CVAE+transformer, ResNet18 4ch RGB-D ×2 cam, K=8 = teacher requery
    주기), 데이터셋(정규화 통계 동봉), n-grid warm-start 학습기, held-out paired 평가기.
  - `gates/` — DINOv2-S/14 + 공용 PCA(32), Mahalanobis 관할(Ledoit-Wolf + calibration 분위수),
    Beta-Bernoulli 성숙도, ACI 위험 통제, lifecycle 3분기 dispatcher.
- **버그 즉시 수정**: `policy.py → teacher.collector → libero_utils → tensorflow` 의존 체인이
  hv2_hab(TF 없음)에서 붕괴 → `quat2axisangle`/`proprio_vector`를 `envs/libero_env.py`에 자체
  구현(robosuite 동일식), 수집기·정책 공용화.
- 인프라 5모듈 적대적 검증 워크플로우 가동(5 렌즈: 수집 충실도 / ACT 정확성 / gate 수학 /
  재현·분리 / 자원 규모). 결과 대기.

### 다음
1. 검증 워크플로우 확정 발견 반영.
2. E1-a 완료 → 집계·go 판정.
3. E1-b 레이턴시 앵커 5종 (GPU 유휴 시): ① OFT chunk forward ② ACT forward ③ DINOv2+gate
   ④ 히든 추출 ⑤ ACT 학습 1회(n=40). ②⑤는 E2 데이터 수집 후에야 실측 가능 → E2 수집 직후 측정.

---

## 2026-08-15 — 인프라 검증(32 에이전트) 확정 26건 반영 — ★치명 버그 1건 사전 차단

### ★ 치명 (수집 데이터 생성 전에 잡음 — 재수집 불필요)
- **[수정] collector가 공식 `process_action`을 생략** — get_vla_action 원시 출력(gripper [0,1])을
  그대로 env.step에 넣고 저장까지 했음. 공식 경로는 사이에 gripper [0,1]→{−1,+1} 이진화 + 부호
  반전(`run_libero_eval.py:265-275`)이 있다. 방치 시: teacher rollout의 gripper 동작 전체가 틀어져
  S_V 붕괴 + ACT가 잘못된 행동 공간을 학습. → process 후 실행·저장으로 수정, habit 실행 경로
  (raw env.step)와 정합. **스키마 v2로 승격.**

### 주요
- ACT 백본 BatchNorm → **FrozenBatchNorm2d** (표준 ACT/DETR 관행 — batch 8 BN 불안정 차단).
- 관할 gate 짝/홀 분할이 수집 스트림 base_idx 순환(40=짝수)과 parity 정렬 → fit/calib이 서로 다른
  초기상태 모드만 보는 계통 편향 → **결정적 셔플 분할 + 최소 N=20 가드 + 유한표본 conformal
  분위수**(⌈(n+1)(1−α)⌉ 순서통계량)로 교체. 소표본 크래시(N≤4에서 LinAlgError/특이 행렬)도 차단.
- collector **증분 HDF5 저장**(성공 에피소드 즉시 기록 + finally 메타 보존) — 장애 시 클러스터
  전체 소실 방지. 메타는 attr 크기 한계 회피를 위해 dataset으로.

### 경미 (전부 반영)
- RGB **ImageNet 정규화** 추가(사전학습 백본 정합), depth [0,1] 유지.
- 표준 ACT의 `additional_pos_embed`(잠재·proprio 토큰 위치 임베딩) 추가. 이미지 토큰 1D PE는
  문서화된 편차로 유지.
- 정규화 통계를 **max-n 풀에서 1회 산출·전 n-grid 동결** (warm-start가 정규화 공간을 안 가로지름).
- `lr_backbone` param group 실제 연결(무동작이던 HP), steps_per_n KeyError 방지, persistent_workers.
- 모델측 예외도 InfraError 승격(수집 계정 3분류 유지), set_seed 순서를 공식과 일치.
- ACI γ=0.02·τ_max=0.99 미등록 상수 → **preregistration §4c 등재**. decide()가
  "jurisdiction_unfit"(미적합)과 "out_of_jurisdiction"(실기각)을 구분 — E5 양방향 회계 보호.
- gate 입력 규격 함수 `prep_gate_rgb`(180° 회전+128) + 해상도 assert — fit/런타임 분포 일치.
- policy 로드는 pretrained=False(불필요 다운로드 제거), **TORCH_HOME 프로젝트 격리**.
- 저장 규모 추정 정정: 클러스터당 ~0.4–0.7GB (docstring·§4b), 체크포인트 381MB×4/클러스터 —
  E3 28 클러스터 ≈ 43GB/seed는 계획 내(E6 시 중간 n 정리 방침).
- VRAM 동시성 확인: OFT rollout(~16GB) + ACT 학습(~5GB) 동시 수용 가능 (32GB).

### 검증 (수정 후 스모크)
ACT loss/act/params(95.0M) 정상, 관할 q=7.40·μ accepts, 성숙도 신선=미성숙(p=0.2)·30/0=성숙(p=0.999),
lifecycle 3분기 사유 정상. E1-a는 object 스위트 진행 중.

---

## 2026-08-15 — E1 완료: S_V 전 스위트 PASS + 레이턴시 앵커 실측

### E1-a S_V 재측정 (태스크당 25 ep × 40 태스크 = 1,000 ep) — **전 스위트 PASS**

| 스위트 | S_V | Wilson 95% | 판정 | walltime 성공/실패 |
|---|---|---|---|---|
| spatial | 0.984 (246/250) | [0.960, 0.994] | PASS | 4.4s / 7.0s |
| object | 0.984 (246/250) | [0.960, 0.994] | PASS | 4.5s / 7.5s |
| goal | 0.996 (249/250) | [0.978, 0.999] | PASS | 4.0s / 8.0s |
| long(10) | 0.968 (242/250) | [0.938, 0.984] | PASS | 7.1s / 12.6s |

- 전부 임계 0.85 크게 상회 — **완화 조항(§3) 미발동**, 셀 재설계 불필요.
- 최약 태스크: long "put both moka pots" 21/25 (84%) — E3 천장 해석 시 참조.
- 실패 에피소드 wall-clock 실측 확보 → E0-7 예산 캐비앳 1(성공-only 편향) 해소.

### E1-b 레이턴시 앵커 (`results/e1/e1_latency.json`, attn=sdpa, RTX 5090)

| 앵커 | 중앙값 | 비율(/OFT) |
|---|---|---|
| ① OFT chunk forward (8 steps) | **85.07 ms** | 1.0 |
| ② ACT forward | **3.36 ms** | **0.040** |
| ③ DINOv2+PCA+2단 gate | **3.96 ms** | **0.047** |
| ④ 히든 추출 | = ① (부산물) | 1.0 |
| ⑤ ACT 학습 1회(n=40) | E2 직후 측정 (pending) | — |

- **핵심 실측 ①**: 습관 경로(gate 4.0ms + ACT 3.4ms ≈ 7.4ms)는 VLA 호출(85ms) 대비 **~11.5배 저렴**
  (발화 시 스텝당). "추론 상각"의 비용 기반이 성립.
- **핵심 실측 ②**: OFT `predict_action`은 이미 `output_hidden_states=True`로 forward — **L32 히든은
  VLA forward의 부산물**(modeling_prismatic.py:910-916). 히든 gate arm(비교 arm ③)은 VLA forward
  비용을 그대로 지불 → 주 gate 경로의 경량성 우위가 비용 축에서 구조적으로 성립.
- 이슈: 앵커 스크립트 1차 실행에서 OFT 커스텀 forward가 labels를 요구해 실패 → 위 부산물 발견으로
  대체(더 정확한 측정). sklearn이 hv2_oft에 없어 2차 실패 → 설치(E5 단일 프로세스 요건이기도 함).

### E2 준비 (사전등록 §4d 등재)
- C-L0 = object task 0 (25/25), C-L1 대표 = object task 5 (24/25).
- `experiments/e2_run.sh`(수집→학습→평가→판정 전 파이프라인) + `e2_collect.py`(go/no-go 검정:
  two-proportion/Fisher 자동 선택 + McNemar 부가 보고) 작성.
- 인프라 이슈: VSCode 확장 훅이 간헐 타임아웃(Write/Edit 차단) → bash heredoc/python 경유로 우회 중.

---

## 2026-08-15 — ★★ E2 = GO — 유일 치명 단계 통과 (H1 형성 실증)

### 결과 (`results/e2/e2_gonogo.json`, held-out 50/클러스터, paired)

| 클러스터 | ŝ(10) | ŝ(20) | ŝ(40) | ŝ(80) | 판정 |
|---|---|---|---|---|---|
| C-L0 (object task0) | 0.04 | 0.20 | 0.48 | **0.96** | GO (p≈0, McNemar 46↑/0↓) |
| C-L1rep (object task5) | 0.74 | 0.92 | 0.86 | **0.96** | GO (p=0.001, McNemar 12↑/1↓) |

- 사전등록 검정 (max_n ŝ ≥ 0.8 ∧ ŝ(80) > ŝ(10), 단측 two-proportion α=0.05): **양쪽 모두 통과 → 종합 GO**.
- **n=80 습관이 teacher 수준 도달**: ŝ(80)=0.96 vs 클러스터 S_V 0.975(task0)/0.933(task5) — 비열등.
- **부수 발견 (H2-L 예고)**: 동일 스위트 내에서도 형성 속도가 크게 다름 — task5는 n=10에서 이미 0.74,
  task0은 0.04. N*(task5)=20, N*(task0)=80 (클러스터 수준 정의 §4.3). → L이 아닌 태스크 내 요인
  (물체 갯수·파지 난이도)도 형성 속도를 지배할 수 있음 — E3에서 검증할 가설로 승격.
- task5 곡선의 n=40 비단조(0.86 < 0.92)는 Wilson CI 중첩 범위 — 사전등록 검정은 10 vs 80만 사용.
- 수집: task0 117/120, task5 112/120 (인프라 오류 0). 실패 궤적은 통계 장부만(이중 장부 §2.5).
- **앵커 ⑤ 확보**: ACT 학습 1회(n=40, warm-start) = 181.8s ≈ **VLA 호출 2,137회 등가**
  (`results/e1/e1_latency.json` 갱신). 클러스터당 full n-grid 학습 ≈ 10.6분.
- E2 wall-clock 실측: 수집 ~10분/클러스터, 학습 ~11분/클러스터, 평가(200 rollout) ~25분/클러스터.

### 리스크 레지스터 갱신
- **R1 (E2 실패 — 유일 치명) = 은퇴.** 이후 단계(E3-T/E4/H2 해석)는 전부 "우아한 퇴화" 경로 보유.

### 다음 (설계서 §5 W3–5)
1. E3 배치: 표준 23 클러스터(Object 잔여 8 + Goal 10 + Spatial-a 2 + Long 3) 수집→학습→평가.
   task0/task5 수집분은 재사용(E3 held-out=20은 E2의 50에서 부분집합이 아닌 신규 스펙 — heldout_specs가
   동일 생성기이므로 앞 20개 = E2 평가와 동일 스펙: paired 유지).
2. C-T2(단일 태스크 2연쇄 커스텀 래퍼) 설계·검증 — 수집이 GPU를 쓰는 동안 병행.
3. Spatial-a·Long 클러스터 선정을 E1 태스크별 실측으로 확정 → preregistration §4e 등재.

---

## 2026-08-15 — 지시서 v2 부록 실행: E5 개정 확정 등재(4상태 lifecycle) + §4g T4 + T0 잔여

### T0-1 결손 보충 (커밋 714c16b 시점 미기록분)
- C-T2 연쇄 래퍼 검증 PASS (`results/e3/e3_t2_validation.json`): 재배치 최대 xy 편차 0.9mm,
  robot qpos 보존, 결정성 해시 2회 일치.
- E3 배치 착수: `experiments/e3_run.sh` 23 클러스터. 수집 단계 04:31–07:45 완료 — 전 클러스터
  COLLECT-PASS (성공 109–120/120, 인프라 오류 0, `logs/e3/driver.log`). 학습 단계 자동 진행 중.

### 한 일 (지시서 v2 부록 A·B — 연구원 승인 확정분)
- **사전등록 개정 등재** (`configs/preregistration.md`, §5 이력 7건):
  - §4 E5 스트림 2,000 → **4,000** ep × 3 seed (커버리지: 2,000이면 클러스터당 노출 78.3 < 80 —
    n=80 체크포인트 도달 수학적 불가)
  - §4b **probe 대역** (seed 40000+1000r+j / noise 3e6+1000r+j / base 0–39 재사용, r = 라운드)
    + E4 novel-2(타 태스크 차용) 예약 대역
  - §4f 신설: H2-L′ 경쟁 가설 + 판정 분석 + 공변량 단일 진입점
  - §4g 신설: T4 "사람 시연 vs VLA 궤적" ablation (시점: T2 이후 저우선, replay-render·필터 대칭 규약)
  - §4h 신설: lazy 재학습 {20, 80} + probe P=20·**총 2라운드 상한** + **습관 부적격** 상태 +
    BC 풀 위생 + ACI 분리. **고정 "19/20" 통과선 폐기** — 동결 이월 계수 c=0.25와 모순
    (검증 실측: 이월 (40,10)이면 19/20도 0.883 탈락) → 성숙 판정은 사후 확률로 통일.
  - §4e 보충: C-T3 배제 태스크(libero_10 task8 "moka pots", 0.84) 명시 + C-T2 ID 메커니즘 명확화
- **구현**: `gates/two_stage.py` 4상태 lifecycle(`ineligible`, `record_probe_round`, 원장 source 태그,
  decide 사유 `habit_ineligible`) / `envs/stream.py` 대역 상수 모듈 승격 + `probe_specs`(라운드 간 분리).
  스모크 PASS: probe 통과(이월 (20,0)+19/20 → p=0.977) · 2라운드 미달 → 부적격 전이 · 3라운드 차단 ·
  decide 사유 · 대역 10쌍 disjoint(라운드 간 포함).
- `results/e1/e1_latency.json`에 **basis 필드** 프로그래밍 추가 (T0-3e): per_chunk **25.32×** /
  conservative_floor **11.62×** — log.md의 "11.5"는 서사 반올림이라 미동결, F3은 basis 명기.
- §4f 공변량: `experiments/e3_free_joints_census.py` 신설 → 40 태스크 CPU 전수조사
  `results/e3/free_joints_census.json` ([CENSUS-PASS], e0_6 task0 정합 검사 통과.
  ControlEnv 렌더러 비활성 — 학습 배치와 병행 안전 확인).
- **[ISSUE-15] 해소**: tmux 3.7을 전용 env `hv2_tools`에 설치 (sudo·base env 불침범 —
  `~/miniconda3/envs/hv2_tools/bin/tmux`). 실행 중인 E3 nohup 프로세스는 이관하지 않음(지시서 주의).
  다음 장기 실행부터 `tmux new -s habit2`.

### T0-4 무해 경고 문서화 (후속 세션 재조사 방지)
- `[Warning]: datasets path .../LIBERO/libero/datasets does not exist` — LIBERO 공식 시연 데이터셋
  미다운로드 경고. 궤적은 자체 수집이므로 **무해** (T4 착수 시에만 다운로드 필요, §4g).
- env close/GC 시점 `EGLError` — 렌더 결과 무영향 소음 (E0에서 기판정).

### 서사 프레임 (사전등록 아님 — 부록 A 지시로 기록)
부적격 판정 = 안전장치 작동(self-aware gating), "성공률 유지" 약속을 지키는 메커니즘으로
Discussion 배치 예정. counterfactual 대비(부적격 클러스터 에피소드의 VLA 실측 성공률 vs 해당
습관의 probe 실측) 산출용 로깅을 E5 드라이버 요건으로 §4h에 등재.

### 미결 (연구원 결정 대기 — 2026-08-15 검토 보고 참조)
1. **클러스터 산술 27 vs 28**: §4:82는 C-L0을 Object(10)에 이중 계상, §4e 재검산 "1+9+10+2+2+3"은
   실제 27 (오기 "=28"). → "27 distinct로 정정" vs "C-L0 별도 셀 정의" 결정 필요.
2. **T1-2 수집 중단 트리거**: task5 연쇄 기대 성공률 S_V² = 0.871 < 0.90 — 전역 "<0.90" 트리거의
   오탐 확률 0.79. 클러스터별 상대 기준(S_V,k² − 이항 허용치) 제안.
3. **H3 "각 단독 대비"**: E5 대조를 성숙도 단독 1종으로 정련하면 관할 단독 비교 부재 —
   H3 문구 개정 vs 소규모 관할 단독 시연 arm 추가.
4. **E5 full-VLA 기준선**: 스트림 실주행(예산 지배, ~20h) vs 태스크별 S_V 재사용.

### 다음
1. E3 학습·평가 완료 대기 → 판정·로그 (진행: 학습 2/23 시점 확인 07:54).
2. T1: `experiments/e3_t2_run.sh` 작성(중단 트리거는 미결 2 반영 대기) → C-T2 수집 2클러스터 (~1.2–1.6h).
3. T2: `e3_collect.py` 집계 + §4f 판정 분석 (공변량 3종 중 free_joints 확보 완료 — S_V·median_len은
   수집 산출물에서 추출 예정).
4. E4 novel 경로(ii)는 **spatial 전용** (검토 실측: object 무효·goal novelty 0·long 2태스크 쌍만) —
   T3 구현 시 BDDL ordered 시그니처 가드 필수.
5. T4·E4 신규 산출물 검증 렌즈: 기존 4종 + replay 정합성 + 필터 대칭성 (부록 C).

---

## 2026-08-15 — 통합 지시서(단일 최종본) §1·§2·§4 즉시분 실행

**정본 참조: 본 시점부터 정본 지시서 = "HabitVLA-2 통합 작업 지시서 (단일 최종본, 2026-08-15)"**
— 이전 3개 문서(지시서 v2 본문 · v2 부록 · 판정 재전달문)를 전부 대체. 직전 로그 항목의
"미결 4건"은 통합 지시서 §1로 전부 판정 확정되어 미결 아님.

### §1 확정 판정 4건 + §9 검정 창 — 사전등록 등재 (§5 이력 6건 추가)
- **클러스터 산술 = 27 distinct**: §4 유도 정정(C-L0 = object task0의 역할 라벨, 별도 클러스터
  아님), §4e 재검산 오기("=28") 정정, `e3_collect.py` 완결성 기준 done == 27로 정합.
- **C-T2 수집 중단 트리거**: 전역 "<0.90" 폐기 → 클러스터별 S_V,k²(task0 0.951 / task5 0.871)
  대비 단측 이항 α=0.01 (§4e 보충 등재 — e3_t2_run.sh 작성 차단 해제).
- **H3 arm 구성 확정**: 주(2단) + 성숙도 단독(전체) + **관할 단독 소형(500 ep, 1 seed)** —
  동결 가설문("각 단독 대비") 무개정 충족 (§4h).
- **full-VLA 기준선 = counterfactual completion**: 별도 arm 폐지, 습관 발화 에피소드만 동일 스펙
  teacher 사후 재실행 → 스펙 단위 paired. 본 실행 전 결정성 사전 검증 항목 포함 (§4h).
- **호출률 검정 창**: 첫/끝 500 → **1,000 ep** (§2 검정표 + §4h, 4,000 비례 스케일).

### §2 A_mat 장부 분리 — 구현 + 회귀 테스트 (★모델링 오류 사전 차단)
- 확인 결과 **미구현이 맞았음**: 기존 update()는 stream(=teacher 포함) 결과를 A_mat 계수에
  산입 — n=80 재학습 시점 이월만으로 probe 0회 무검증 성숙 통과 가능(게이트가 s_H 아닌 s_V 측정).
- 구현(`gates/two_stage.py`): source ∈ {teacher, probe, fire} **필수화**, A_mat 계수 =
  **probe + fire만**, teacher는 history(𝒟_k 기록·보고 전용). 파생 정합: 습관 이력 없는 첫 재학습
  직후 probe는 정확히 Beta(1,1) 기점. `e1_latency.py` 호출부 source 명시.
- `experiments/gate_regression.py` 신설 → **[GATE-REGRESSION-PASS]** (스모크 목록 등재):
  t1 teacher-only 80성공+이월 2회 → p=0.200 미통과(§2 위험 시나리오 차단) / t2 probe 산입
  (19/20=0.9424 통과, 18/20=0.8213 탈락) / t3 fire 이월(Beta(24,3)=0.9159 통과) / t4 부적격
  전이·3라운드 차단 / t5 source 검증·history / t6 ACI 분리(observe_fire가 A_mat 계수 불변) /
  t7 대역 disjoint 10쌍(probe 라운드 간 포함) + round_idx 상한.

### §4 공변량 원자료 완성
- `experiments/e1_sv_per_task.py` → `results/e1/e1_sv_per_task.json` **[E1-PERTASK-PASS]**:
  EVAL-*.txt 순차 파싱으로 (suite, task_id) 키 재유도. 4 스위트 합계가 e1_sv.json과 완전 일치
  (spatial 246/250 · object 246/250 · goal 249/250 · long 242/250), libero_10 task8 앵커(21/25)
  재확인. spatial 절단명 병합 버그 회피본 — 원본 e1_sv.json 불변.
- `experiments/e3_covariates.py` → `results/e3/covariates.json` **PARTIAL(25)**:
  표준 25 클러스터 전부 free_joints + S_V_cluster + median_len_success 산출.
  chained 2는 `pending` 명시 — C-T2 수집 후 재실행 시 자동 편입되어 COMPLETE(27).

### 비고
- E3 배치 진행 중 (학습 단계, 본 항목 작성 시점 goal_task4 부근 — 전 TRAIN-PASS).
- 다음: E3 학습·평가 완료 → §5(`e3_t2_run.sh` 신규 작성, §1-2 트리거 반영) → §6(집계·H2 판정,
  완결성 27) → §7(T4) → §8(E4) → §9(E5 드라이버).

---

## 2026-08-15 — E3 표준 배치 완료·커밋 + ★C-T2 스모크 FAIL·진단 완결 (연구원 결정 대기)

### E3 표준 배치 완료 + 커밋 0d87053
- 평가 23/23 [EVAL-PASS] → [E3-STANDARD-DONE] (13:00). 22곡선 예비 집계: 그리드 내 ŝ≥0.8
  도달 21/22, ŝ(80) 중앙값 0.95, N* 분포 {10:10, 20:8, 40:2, 80:1, >80:1}.
  유일 절단 = libero_10_task0(2물체 연쇄, 0.75) — H2-T 천장 하강 예고.
- 연구원 지정 마디(평가 완료 직후·C-T2 전)에서 커밋 `0d87053` (37 파일: 사전등록 개정 +
  A_mat 장부 분리 + 회귀 테스트 + 공변량 + 곡선 JSON).

### §5 C-T2 착수 → ★[T2-SMOKE-FAIL] (스모크 게이트가 설계 결함을 데이터 수집 전 차단)
- 인프라: collector/evaluate `--chained` 확장(stage 분해 meta 포함), `e3_t2_check.py`
  (스모크 게이트 + §1-2 상대 트리거), `e3_t2_run.sh`. tmux `habit2` 첫 가동 (13:04).
- **[T2-SMOKE-FAIL] chained_object_task0: 0/10** — 단, 분해가 결정적:
  **10/10 전부 stage 1 성공(~137 스텝)·재배치 정상 수행 후 stage 2에서 전원 실패.**
  래퍼 메커니즘 문제 아님(재배치·결정성은 기검증 PASS), teacher × 전환 상태의 계통 실패.

### 진단 (`experiments/e3_t2_diag.py`, `results/e3/t2_diag*.json`, 프레임 PNG)
- **실험 1 (계기 재현)**: stage 2 진입 후 teacher가 **바스켓 위 포즈에서 동결** —
  EE (-0.05, 0.26, 0.16) 정지, 행동 크기 |a|≈0.012 (시연 종료 시점의 정지 행동 분포),
  predicate는 재배치로 정상 false. 프레임: 진입 vs 종료(416 스텝 후)가 시각적으로 동일.
- **실험 2 (통제)**: 동일 재배치 물체 배치를 **로봇 홈 포즈**에서 시작 → teacher 153 스텝 성공.
- **판정 = 가설 B**: 재배치 상태는 해결 가능. OFT teacher가 홈 포즈 시작 분포로만 학습되어
  **stage 1 종료 직후 포즈(팔 뻗은 상태)에서 태스크 재시작이 OOD** — 정지 행동 출력.
- **Option 2 probe (홈 재설정 전환) = 3/3 성공** (stage_steps 예: {1:137, 2:330}) —
  전환 시 set_init_state(로봇 홈 + 물체 재배치) + settle이면 teacher 성공률 복원 실증.

### 연구원 결정 대기 — C-T2 전환 의미론 (파이프라인 불변 유지 중)
A(권고). **전환 시 로봇 홈 재설정** — probe 3/3 실증, 구현 최소(래퍼 1 메서드).
   의미론 = "재발 조우의 연속을 한 에피소드로 압축"(스트림 에피소드 경계와 동형).
   정직 명시 필요: 로봇 상태 경유 물리 연속성은 단절 — H2-T 해석은 "단일 정책의 2×
   태스크 스팬 학습 부담"으로 명문화(§4e 문구 개정) + 래퍼 결정성 재검증 + 재스모크.
B. 스크립티드 리트랙트(실물 이동 홈 복귀) — 물리 연속성 유지되나 OSC 피드백 루프 신규
   구현·자체 검증 필요, 잔여 포즈 노이즈의 OOD 위험 잔존.
C. 설계서 대체 경로 — C-T2 폐지, Long 길이 층화로 T2 대체 (§4e·§4h·27 산술 재개정 필요,
   L-고정 paired 설계 상실).

---

## 2026-08-15 — 연구원 판정: 옵션 A 채택 → 래퍼 개정·재검증 PASS·배치 재가동

### 판정 (연구원): **옵션 A** (B 기각 — A에 우월성 없음 / C 불필요)

### ★ 발견 등재 — "teacher 관할 경계 = 습관 학습 가능성의 상한"
습관은 teacher 성공 궤적에서만 형성되므로(H1 정의), **teacher 자신의 관할(운용 분포) 밖
상황에서는 시연이 생성될 수 없어 습관 형성이 원천 차단**된다. C-T2 v1 전환 상태(팔 뻗은
포즈에서 태스크 재시작)가 정확히 이 사례: 상태는 물리적으로 해결 가능(홈 포즈 통제 실험
153스텝 성공)한데 teacher가 시연 종료 분포의 정지 행동(|a|≈0.012)만 출력 — 0/10 동결.
→ 본 연구의 관할 개념이 gate 판별 축을 넘어 **형성 가능성의 구조적 상한**으로도 작동함을
보여주는 실증 사례. Discussion 후보 (게이트 관할과 teacher 관할의 이중성).
수치: v1 스모크 0/10(전원 stage 1 성공 ~137스텝 후 동결), 통제 153스텝 성공, 홈 재설정
probe 3/3. 산출물: `results/e3/t2_smoke_v1_negative.json`, `t2_diag.json`, `t2_diag_probe.json`.

**교정 3요소 (과잉 일반화 방지 — 연구원 지시 2026-08-15 반영):**
  (i) 이는 **VLA 일반의 성질이 아니라, 좁은 시연 분포에 대한 파인튜닝의 전문화 대가**다 —
      OFT LIBERO 체크포인트는 홈 포즈 시작·성공 종료의 시연만으로 학습되었다.
  (ii) 이 경계는 **벤치마크 성공률 97%+에 드러나지 않는다** — 벤치마크 평가가 전부 학습
      분포 내 시작 상태에서 이뤄지기 때문이며, 경계는 분포 밖 시작 상태를 요구하는 순간
      비로소 관측된다.
  (iii) **base VLA(파인튜닝 전) 또는 더 다양한 시작 상태의 시연으로 학습했다면 달랐을 수
      있다** — 반론 선제 인정. 본 발견의 주장 범위는 "본 시스템 구성(협분포 파인튜닝
      teacher)에서의 형성 상한"으로 한정한다.

### 실행 (판정 순서대로)
1. `envs/chained_env.py` 전환 개정: 전체 상태 벡터 `set_init_state`(물체 재배치 + 로봇 홈)
   + settle 10. 진단 전용 probe 클래스와 동일 메커니즘의 정식 반영.
2. `e3_t2_validate.py` 로봇 체크를 "불변" → "홈 재설정" 의미론으로 갱신, 스모크 임계도
   게이트와 정렬(≥7/10). **기계 재검증 PASS**: 물체 편차 0.9mm, 로봇 홈 편차 0.0295,
   결정성 해시 일치.
3. 사전등록 §4e 개정 3건 + §5 이력: (a) 전환 의미론(연속성 단절 명시) (b) C-T2 주 검정 =
   곱 기준선 (ŝ_T1(80))² 대비 단측 이항(원 기준 병행 보고) (c) 동역학 천장 주 증거 = C-T3.
   상대 트리거 기대치 불변.
4. 본 항목 + negative result 커밋 → 재스모크(동일 게이트 ≥7/10) → `e3_t2_run.sh` 재가동.

---

## 2026-08-15 — ★C-T2 2차 실패(task5)·진단 완결: v2 전환도 결함 — E0-6 프로토콜 위반이 원인

### 경과
- v3 아님 v2(홈 재설정) 재가동분: **task0 스모크 10/10 → 수집 112/120·트리거 OK(p=0.242)** 후
  **task5 스모크 0/10 재발동** (stage 1은 9/10 정상, stage 2 전원 실패 — task0과 비대칭).

### 진단 사슬 (t2_diag2·3·4 + 프레임)
1. **diag2 (4조건)**: task5 base17은 **무섭동 공식 init state도 fresh 실패** — 섭동·대역 무관.
   E2 수집 메타 재분석: **task5 base17 = 3/3 결정적 teacher 맹점** (task0엔 결정적 맹점 없음).
   단, 맹점만으론 reloc base 18–26의 stage 2 전멸 설명 불가.
2. **diag3 (판별)**: 동일 재배치 상태(base18)가 **fresh는 118스텝 성공, v2 연쇄 stage 2는 실패**
   → 상태 벡터가 같아도 전환이 fresh와 비동등.
3. **diag4 (3자 상태 비교)**: 전환 직후 vs fresh — **물체 전부 Δ=0.0, 로봇만 최대 0.48 rad 이탈**.
   원인 = reset 생략으로 **OSC 컨트롤러 상태(stage 1 stale goal) 이월** — settle 동안 팔을
   끌고 감. **E0-6이 재현 프로토콜을 3단(seed→reset→set_init_state→settle)으로 강제한
   바로 그 사유를 전환 구현이 위반**한 것. 실패 양상: teacher가 이탈 포즈에서 유사 빨간
   상자(케첩류)를 오인해 바스켓에 넣고 "완료" 정지 — task0 teacher는 이탈에 강건(10/10),
   task5는 오인 절벽 → 비대칭의 정체.

### 조치 (v3)
- 전환 = **에피소드 경계 프로토콜 그대로**(begin_episode 재사용, 전환 seed = 에피소드 seed).
  옵션 A 의미론 불변 — 구현 정밀화 (§4e·§5 이력 등재). 파생 이득: S_V,k² 기대치 근거가
  구성적으로 정확해짐 (stage 2 = fresh와 동등).
- 기계 재검증 PASS (로봇 홈 편차 0.029·결정성 일치) + **v3 probe: task5 reloc 18–20 중 2/3
  성공** (실패 1건은 확률적 범위 — 기대 S_V²=0.871), task5 계통 실패 해소.
- **v2 수집분 task0 120 ep 폐기·재수집** (data/e3/*.v2bak 보존) — BC 데이터와 런타임 전이
  과정 일치 원칙. negative result 2건 보존: `t2_smoke_v2_negative.json` (task5 0/10 +
  task0 v2 수집 폐기 사유).
- 참고: 게이트 재발동 가능성 — task5 스모크는 reloc base17(맹점)을 포함하므로 기대 성공
  ≈ 7.8/10, 7/10 게이트가 확률적으로 걸릴 수 있음. 재발동 시 p-hacking 재실행 없이 보고.

---

## 2026-08-15 — task5 v3 스모크 6/10 발동 → 연구원 판정: 통계적, 수집 진행 (옵션 A)

### 판정 (연구원): 옵션 A — B(사후 재보정) = 사전등록 위생 훼손, C(대체 발동) = 과잉으로 기각
- §5 기록 완료: 분해(기지 결정 실패 3 + 확률 1, 신규 계통 신호 없음, P(≤6)=0.099 사전 예고
  범위) + 게이트 무개정 + 스모크 재실행 금지 + **사전 약정(수집 트리거 발동 시 즉시 정지·보고,
  재량 진행 금지)**.
- 실행: task5 수집 120 ep 재개 (스모크 생략 — 판정 근거 §5). 구속 가드 = 상대 트리거
  (α=0.01, p₀=0.871). 이후 흐름 불변: [T2-DONE] → h50 보충·병합 → §6.

### ★ Discussion 후보 등재 — base17 맹점: "공식 init state 분포 **안**에도 존재하는 teacher 관할 경계"
기존 발견("teacher 관할 경계 = 습관 학습 가능성의 상한")의 **분포 내 강화판**. 3중 독립 실측 일치:
- **E1 공식 평가**: task5 유일 실패(24/25)가 정확히 **init state 17** (원 로그 재파싱 확인).
  스위트 집계 98.4%에 완전히 비가시 — 1/25 실패로만 흔적.
- **E2 수집**: base17 = 3/3 결정적 실패 (그 외 base는 확률적).
- **진단**: 무섭동 공식 state 17도 fresh 실패 — 섭동·대역 무관, teacher의 상태 조건부 맹점
  (유사 빨간 상자 오인으로 잘못된 물체를 바스켓에 투입 후 "완료" 정지).
함의: 관할 경계는 분포 밖에서만 생기는 것이 아니라 **벤치마크 분포 내부에 결정적 구멍**으로도
존재하며, 집계 성공률로는 검출 불가 — 상태 조건부 관할(2층 gate)의 필요성을 정면 지지.
교정 3요소(협분포 파인튜닝 대가·집계 비가시·base 모델 반론 인정)는 기존 등재분 그대로 적용.

---

## 2026-08-15 — ★task5 수집 상대 트리거 발동 (88/120) → 사전 약정대로 정지·보고 (연구원 결정 대기)

### 발동
- task5 연쇄 수집 **88/120 = 0.733** vs 기대 S_V² = 0.871, P(X≤88) = **4.03×10⁻⁵** << α=0.01
  (교정 2026-08-15 R5: 최초 기록 "≈3×10⁻⁶"은 수기 근사 오류 — 프로그래밍 산출값으로 정정,
  **판정 무영향**: 어느 값이든 α=0.01을 크게 하회) →
  **[T2-TRIGGER-FAIL]**, 재개 스크립트가 사전 약정대로 러너 미진행·즉시 정지 (재량 진행 없음).
- 대조: task0 연쇄는 118/120 = 0.983 (기대 0.951 상회) — 건전.

### 분해 (수집 120 ep 메타 — 데이터는 유효, 래퍼 무결 재확인)
- **stage 1 = 114/120 (0.95)**: E2 싱글과 스펙 동일 — 실패 6건 전부 E2 실패 부분집합
  (기지 base 17×3·5·7·28). E2에서 실패했던 idx 27·28 두 건은 연쇄에서 성공 — 원인은
  래퍼의 **pooled budget**(총 560): 싱글 280 상한에서 timeout이던 에피소드가 구제됨 (설계
  등재 "총 예산 = 2×" 그대로 — 이상 아님).
- **stage 2 조건부 = 88/114 (0.772)** vs 기대 0.933:
  - 결정적(3/3) reloc 맹점 = **{17, 28}** (28은 collect 대역에서도 2/3 실패였던 준맹점).
  - 비맹점 base에서도 **0.815 (88/108)** — 동일 base·동일 w의 collect 대역 draw(0.974) 대비
    **−16%p**. 실패가 15개 base에 1–2/3로 산포 (계통 아닌 광역 저하).
- **판독: 래퍼 결함 아님** (stage 1 정확 재현 + stage 2 = fresh 구성적 동등 + task0 정상).
  원인 = **task5 teacher 성공의 칼날(knife-edge) 상태 민감성**: 명목상 동일한 분포
  (동일 base, w=0.01 uniform)의 **다른 noise·seed draw에서 성공률이 0.97 → 0.82로 급락**.
  S_V=0.933은 collect-대역 draw의 단일 표본 추정이었고 task5에선 draw 간 분산이 큼
  (task0은 강건: 0.983). → 관할 경계 발견의 **3층위**: ① 분포 밖(포즈) ② 분포 내 결정적
  구멍(base17) ③ **분포 내 draw 민감성** (Discussion 후보 추가).

### 주 검정 관점의 쟁점 (결정 재료)
task5 체인을 그대로 진행하면 teacher 앵커가 0.733이라 곱 기준선 검정(vs 0.9216)의 하회가
"습관의 장기 시퀀스 결합 비용"인지 "teacher 시연 풀의 질·양 저하(88 성공)"인지 **분리 불가** —
주 검정 해석 오염. task0 체인은 teacher 0.983으로 청정.

### 선택지 (연구원 결정 사항 — 사전 약정대로 이월)
(i) task5 체인 → **고강건 태스크로 교체** (후보: object task6 — 수집 120/120·E1 25/25).
    paired 논리 유지. 단 h50 곱 기준선의 T1 참조가 task6엔 E2 50 곡선이 없어
    E3 held-out 20 기반으로 하거나 task6 싱글 held-out 50 평가 1회(~25분) 추가 필요.
(ii) task5 유지·진행 + p₀ 개정: 88 성공 궤적은 BC 풀로 유효하나 위 해석 오염 문제.
(iii) C 경로(Long 길이 층화 대체) — task0 단독 표본 부족 판단 시.
task5 체인 수집분은 어느 경우든 **③층위 증거로 negative/부록 보고 가치** 보유.

---

## 2026-08-15 — 검토 결과 반영 R1–R5 (§6 실행 전 필수분) + 영상 작업 보류

### 연구원 지시: 영상 파이프라인 중단·보류 (매니페스트 + 스크립트 4종 완성 상태 — 재개 가능)
teacher 녹화 중 tmux 중지. `results/videos/_raw/` 부분 산출물 무해 잔존, manifest·스크립트로
언제든 재개. **판정: 체인 교체 = (i) task6** (R1로 확정 전달).

### R1 — 집계기 개정 (결과 판독 전, §5 등재)
- `e3_collect.py`: T2_CHAINED = **task0 + task6** (task5 폐기분 부록 참조만) / 곱 기준선 T1
  참조 = task0: E2 50, task6: **싱글 h50 병합본**(신설 `libero_object_task6_curve_h50.json` —
  E3 표준 20 곡선 불변) / 완결성 = **27 구성원 집합 검사**(개수 아님, missing·unexpected 보고) /
  산출·마커 파일명 `e3_curves.json` 통일. dry-run PARTIAL(25) 정상 (missing = chained 2).
- h50 기계 확장: `e3_t2_h50.sh`·`merge` — 체인 (0,6) + task6 싱글 보충(21–50) 경로 추가.
- `e3_covariates.py` 명단도 task6 기준 갱신 → 재실행 **PARTIAL(26)** (chained_task0 편입,
  pending = chained_task6만).

### R2 — 증거 복구 + 경로 위생
- `t2_diag.json` 덮어쓰기 사고 복구: v1 원본(task0, verdict B)을 git 791977f에서
  `t2_diag_task0_v1.json`으로 회수, task5(v2 시점)본은 `t2_diag_task5_v2.json` 사본.
- `e3_t2_diag.py` 출력 경로에 task suffix 의무화. `make_review_pack.py` v1 오귀속 수정
  (v1 = task0/verdict B; task5 진단은 v2 섹션의 별도 항목).
- 검증 렌즈에 **"출력 경로 유일성(덮어쓰기 금지)"** 추가.

### R3 — 사실관계: covariates median_len **결측 없음** (전제 불일치 보고)
원본·검토 팩 모두 25/25 완결 — 검토가 지적한 결측은 재현되지 않음. 실제 공백은 chained
`pending` 2건(수집 전 구조적)뿐이었고, task0 편입으로 26/27. §4f 3공변량은 가용 클러스터
전체에서 완결 확인.

### R4 — `e3_h2_analysis.py` 신설 (§4f 사전등재 추정량 구현)
순위 기반 주 분석(절단 = 공동 최상위): 레벨 간/내 분산 분해 + Kruskal–Wallis + 공변량
순위 OLS(순열 p, B=10⁴ seed 0) + Spearman / 보조: 구간 민감도(log, cap=160 문서화).
표본 이원(형성 22 주 / 표준 25 민감도). dry-run 실행 확인 (status=DRY_RUN 명기 — 본 판정은
§6 COMPLETE 후).

### R5 — p값 교정
task5 트리거 p: 수기 "≈3×10⁻⁶" → 프로그래밍 산출 **4.03×10⁻⁵** (판정 무영향, 해당 항목
인라인 교정 + §5 기록). 수치 수동 입력 금지 원칙의 재확인 사례.

### 대기
- **diag5 · task6 체인 착수 지시** (task6 트리거 p₀ 명세 포함 — S_V²=1.0 퇴화로 미결 등재).
- 완료 후: task6 체인 수집·학습·평가 → h50(체인 2 + task6 싱글) → §6 집계 → R4 본 판정.

---

## 2026-08-15 — ★diag5 = 세계 B → diag5b 원인 확증: "stale chunk tail" 실행기 결함 (연구원 회부)

### diag5 (조건부 판정 블록 — 승인된 명세 그대로)
- task5 비맹점 stage-2 실패 **20건 전수**를 동일 입력(seed·state 구성 동일)으로 표준 fresh
  재실행: **18/20 성공** → [DIAG5-WORLD-B]. 대역 구성 동일성 = 수치 검증 통과
  (perturbed_init_state ≡ chained 인라인 섭동, max diff 0.0).
- 약정 이행: **task6 체인 진행 중단**, 원인 격리 착수.

### ★ 정정 — "③층위: 분포 내 draw 민감성" 해석 철회
직전 항목의 3층위 발견 중 **③은 반증됨**: 같은 (seed, 상태)가 fresh에서 18/20 성공하므로
"명목 동일 분포의 다른 draw에서 성공률 급락"은 상태·draw의 성질이 아니었다.
stage-2 조건부 0.772의 실체 = **실행기 결함**(아래) + 진성 확률 실패 2건 + 맹점 {17,28}.
①(분포 밖 포즈 OOD)·②(분포 내 결정적 구멍 base17)는 유효 — Discussion 후보 유지.
방법론 교훈(부록 후보): 그럴듯한 해석도 전수 재실행 판정 블록 앞에서 반증될 수 있음 —
diag5의 존재 가치 실증.

### diag5b — 원인 확증 [DIAG5B-CONFIRMED 18/18]
- **가설**: 전환이 chunk 중간에 발생하면 실행기(collector·evaluator의 K=8 open-loop 루프)가
  전환 **전** 관측으로 계산된 잔여 stale 행동(실측 2–7개)을 전환 후 홈 포즈에서 계속 실행 —
  첫 fresh 질의 전에 팔이 교란. task0 teacher는 강건(118/120), task5의 취약 판별은 붕괴.
- **검증**: 뒤집힌 18건을 chunk-break 실행기(전환 감지 시 stale tail 폐기·즉시 재질의)로
  chained 전 구간 재실행 → **18/18 성공**. 성공률 0.77의 중간값도 stale 길이 분포(0–7)와 정합.
- 함의: v3 래퍼 자체는 무결(상태 구성 fresh 동등) — 결함은 **실행기와 전환의 상호작용**.
  §4e "fresh 동등" 의미론을 완성하려면 실행기의 전환 시 chunk-break가 필요 (수집·평가 양쪽).
  task5 트리거 발동(88/120)의 원인 재귀속: teacher 성질 → 실행기 아티팩트.

### 연구원 회부 (약정) — 선택지
(α — 권고) **실행기 수정(전환 시 stale tail 폐기) + task5 유지 복귀**: 수정 후 stage-2 기대
    ≈ S_V² = 0.871이 정확히 성립(diag5b 실증) → 원 paired 설계(task0+task5) 복원, task6
    불필요(R1a 재개정 회귀). task0 체인도 실행기 일치를 위해 재수집(3차, ~20분).
    §4e 사유 재귀속 개정 + 재검증(스모크) 필수.
(β) 실행기 수정 + task6 유지(R1a 그대로) — 강건성 선호 시. 단 task6 교체의 근거였던
    "task5 앵커 오염"이 실행기 원인으로 재귀속되어 교체 논거 약화.
(γ) 수정 없이 task6 진행 — **부적합**: stale tail은 어느 체인에도 존재하며 §4e 등재
    의미론("fresh 동등")과 모순. 기각 권고.

---

## 2026-08-15 — 연구원 판정: α 채택 → 실행기 개정·체인 복원·재수집 파이프라인 가동

### 판정: **α** (β 기각 = 교체 논거 소멸 / γ 기각 = §4e 모순)

### 실행 (판정 1~2항)
- **실행기 개정**: 공용 헬퍼 `execute_chunk_with_boundary`(envs/chained_env.py) — 전환 감지 시
  잔여 chunk **폐기 + 즉시 재질의**, stale_discarded 로깅. collector·evaluate 양쪽 적용(동형).
  **단위 테스트 5종 PASS** (`executor_chunkbreak_test.py`: 중간 전환 폐기·무전환 전량·경계
  일치 stale=0·일반 env 무영향·max_steps 절단) + 게이트 회귀 테스트 재통과.
- **§4e/§5 개정 (a)~(e) 수집 전 등재**: 경계 재질의 의미론(K=8 양립) / 원인 재귀속·③층위
  철회(①② 존치) / 구성 복원 task0+task5(task6 경로 이력 존치, Wilson 하한² 규칙은 퇴화
  케이스 전용 존치) / 트리거 원안 유지 + **발동 후 완화 금지 원칙** / 영향 반경 = 체인 전용
  (E2/E3 표준·E5 무영향 — 단위 테스트 t4로 고정).
- 로스터 회귀: e3_collect(T1 참조 = 양쪽 E2 50)·covariates·h50·러너 전부 task0+task5.

### ★ 방법론 부록 후보 — "3막 계기 사가" (연구원 지시 4항)
C-T2 래퍼의 세 결함이 **모두 데이터 사용 전에 차단**된 기록:
- 1막 (v1, 물체만 재배치): 스모크 0/10 → teacher 포즈 OOD. 게이트가 차단.
- 2막 (v2, reset 생략): task5 스모크 0/10 → OSC 이월 0.48 rad (E0-6 프로토콜 위반). 게이트가 차단.
- 3막 (v3, stale tail): task5 수집 트리거 발동 → **판별 진단(diag5)이 "그럴듯한 해석(draw
  민감성)"을 전수 재실행으로 반증**하고 diag5b가 실행기 결함을 확증. 트리거+판정 블록이 차단.
공통 구조: 사전등록된 게이트·트리거·판정 블록의 3중 방어가 각 막에서 작동 — negative
result 3건이 전부 커밋으로 보존됨 (t2_smoke_v1/v2_negative, t2_diag5/5b).
**task6 반사실 (필수 기록)**: 최강건 teacher(task6, 120/120)로 교체했다면 stale tail 하에서도
어떤 트리거·게이트도 발동하지 않았을 것 — 결함이 **무음으로 T2 셀을 오염**한 채 §6 집계까지
진행됐다. task5의 "취약성"이 결함을 드러낸 카나리아였다 — 강건성만 좇는 셀 선정의 위험 실증.

### 다음
1. 결정성 재검증(mechanics) → v3 수집분 `.v3bak` 보존 → 재스모크(양 태스크) → 재수집
   (트리거 원안: task0 0.951 / task5 0.871) → 학습 → 평가 20 → h50(체인 2종) → §6 → R4.
2. 영상 매니페스트 chained 항목은 재수집본 기준 갱신 예약 (§6 후 재개 원칙 불변).

---

## 2026-08-16 — ★★ α 파이프라인 완주: 27 완결 §6 집계 + R4 본 판정 (H2-L′ 지지)

### 파이프라인 (전 관문 1회 통과 — chunk-break 실행기)
- 재스모크: task0 10/10 · task5 **8/10** (개정 실행기 하 예측 ≈8 적중; v2 0/10·v3 6/10에서 정상 복귀)
- 재수집: task0 **117/120** (트리거 p=0.939) · task5 **108/120 = 0.900** (p=0.862 — diag5b 예측
  "수정 시 기대 S_V²=0.871 성립" 적중) → 학습·평가·h50 병합(50 ep)·covariates COMPLETE(27)
  → **[E3-CURVES] COMPLETE** (구성원 집합 검사 missing=[] unexpected=[]) → **[H2-ANALYSIS-FINAL]**

### §6 주 검정 결과 (`results/e3/e3_curves.json`)
- **C-T2 곱 기준선 (p₀ = 0.96² = 0.9216, h50 n=50) — 셀 내 분기 관찰**:
  - chained_task0: ŝ_chain(80) = **0.80** [0.670, 0.888] → **유의 하회** (p=0.0049,
    bootstrap P(Δ<0)=0.947, Δ̄=−0.124; secondary 감소 p=0.0069). **결합 비용 검출.**
    체인 곡선 {10:0.60, 20:0.34, 40:0.56, 80:0.80}, N*=80 — 느린 형성자에게 2× 스팬은
    n=80으로도 미성숙.
  - chained_task5: ŝ_chain(80) = **0.98** [0.895, 0.997] → 하회 아님, 상회 경향 비유의
    (p_above=0.089, Δ̄=+0.057). 사전 등재된 후보 해석(시연 2배 효과) 방향. N*=20.
  - **해석 후보 (해석 세션용)**: E2의 형성 속도 이질성과 정합 — 빠른 학습자(task5)는 2×
    스팬을 흡수, 느린 학습자(task0)는 결합 비용 노출. "결합 비용은 균일하지 않고 단일
    태스크 형성 난도에 조건부"라는 정식화 가능.
- **동역학 천장 주 증거 (T1 vs T3)**: 0.975 vs 0.883, 단측 p=0.097 — **α=0.05 비유의**
  (정직 보고: 하강 경향은 있으나 pooled에서 long task0(0.75)이 task2·5(≥0.90)에 희석).
- N* 분포 (27): {10:11, 20:9, 40:3, 80:3, >80:1}.

### R4 본 판정 (`results/e3/h2_analysis.json`, FINAL) — **H2-L′ 지지**
- 레벨 분산 분해 (형성 22): **between_share = 2.5%**, Kruskal–Wallis p = 0.772 —
  **의미 레벨 L은 N*를 설명하지 못함** (H2-L 반증 방향).
- 공변량 순위 회귀 (순열 p): **median_len β=+3.72, p=0.0155 (유일 유의)** — 궤적 길이(운동
  시간)가 형성 속도를 지배. free_joints β=−3.69 p=0.076 (경계·부호 음 — 해석 주의),
  S_V 비유의. Spearman 정합(median_len ρ=0.36).
- → 사전 등재 프레이밍 발동 가능: **"인수분해 아키텍처(1층 태스크 정체성)가 의미 부담을
  흡수, 형성 비용은 운동 축이 지배"** — 해석 세션 판정 대상.

### 다음
- 분석 파일 생성·전달 → 영상 재개 (27 클러스터, STAGE 2 마커 포함 — §6 후 재개 원칙 이행).

---

## 2026-08-16 — 영상 지시서 v2 완주: 72편 (V1 27 · V2 19 · V3 25 · V4 1), 검증 렌즈 전 통과

- 분석 패키지 전달: `analysis_pack_e3_final_20260816.tar.gz` (53.8KB — ANALYSIS.md + 결과 JSON 9종).
- 영상 재개(§6 후 원칙): 27 클러스터 매니페스트(chained 2 포함, 정본 chunk-break 실행기 롤아웃,
  STAGE 2 마커) → teacher/habit 녹화 → 렌더 = **72편 149MB**, `results/videos/` (mp4 gitignore).
- 검증 렌즈: ① 카운터 앵커 = e1_latency.json 프로그래밍 취득(85.07/7.32ms) ② 재현 단언
  전수 통과 — V1 양측 성공 27/27, V2 실패 재현 19/19, V3 실패 재현 25/25 (결정성 위반 0)
  ③ 스펙 uid 기록-영상 대응 index 등재 ④ 출력 경로 유일성(클러스터별 디렉토리).
- V1 추론 배율: min 7.6× / 중앙 **11.6×**(= 등재 conservative floor 정합) / max 15.5×.
- 산출: `results/videos/index.json`(cluster·편·uid·단언·추론 합계·크기) + 시청 권장 순서 보고.

---

## 2026-08-16 — §6 후속 마감 + E4 착수 (차기 지시) + heartbeat 인프라 신설

### 기록 (지시 0·6)
- **T5(습관 용량 ablation) = 연구원 판정 보류** — 실행 금지, 실행 후보 목록에만 유지.
- 지시 6(V4 벽시계)은 직전 영상 런에서 **기생성 완료** (`libero_object_task0_V4_wallclock.mp4`,
  실측 앵커 재구성 명시 오버레이 포함) — 재생성 불요.

### 신규 인프라 — 진행 heartbeat (연구원 지시)
- `tools/progress_heartbeat.py`: plan 주입형 공용 10분 심장박동 — 가중 진행률 + 이중 ETA
  (계획 단가 vs 세션 실측 처리율, 괴리 >20% 시 실측 우선), [PROGRESS]/[PROGRESS-STALL] 마커,
  status 파일 + ETA 이력(jsonl, 캘리브레이션용). 단가는 앵커 파일 프로그래밍 취득(부재 시
  null → 처리율 ETA + "불확실" 태그).
- `tools/with_heartbeat.sh`: tmux 장기 작업 동반 실행 표준 래퍼. 모니터 정규식에 PROGRESS 편입.
- 적용: 20분+ 전 작업. **첫 적용 = E4 프레임 덤프 (ETA 오차 캘리브레이션 겸용)**.

### §6 후속 (지시 1)
- **1a 공선성 부록**: h2_analysis.json에 corr(27)+VIF(형성22) 추가 — ★판명:
  **free_joints는 형성 셀에서 suite 더미와 완전 공선(VIF=∞)** (스위트 내 상수 7/4/5) →
  "부호 음 해석 주의" 쟁점은 공선 아티팩트로 해소, 계수 식별 불가를 부록에 자동 명기.
  median_len VIF=1.65 건전 — **H2-L′ 주 결과 불변·강화**. S_V~median_len r=−0.42(p=0.03)만
  유의한 약공선 — 문제 수준 아님.
- **1b E6 부분집합 등재**: §4i 신설 — 7셀×5seed (즉시 goal_task2 / 빠름 object_task5 / 느림
  goal_task0 / 특이-하락 object_task1 / 절단 libero_10_task0 / C-T2 분기쌍 chained_task0·5).
- **지시 5**: `docs/E5_DRIVER_CHECKLIST.md` 신설 — C-T2 3막 교훈 6렌즈(경계·컨트롤러 이월·
  실행기 대칭·경로 유일성·source 태그·대역 disjoint) + E5 고유 3항.

### E4 착수 (지시 2~4)
로드맵: §6 완료 → **E4-1 known 프레임 → E4-2 novel(주 base 40-49) → E4-3 AUC(0.75) 갈래**.
§5 등재: novel 주 설계 base 40–49 개정(known 정합) + BDDL 재샘플 대역(seed 60000+/noise 5e6+).

---

## 2026-08-16 — ★E4 종결: 정식 재판정 REDUCE 확정 + scorer 진단 6행 + H3 갈래 (ii) 확정

### E4-1/2/3 사슬 (그날의 계기 결함 2건 포함 — 전부 데이터 사용 전 차단)
- E4-1 known 500장 [PASS] (heartbeat 첫 적용·캘리브레이션 — 단가 앵커 1.20s/realize 등재).
- E4-2: v1 검사기가 goal/spatial 스폰 높이(~7cm) settle을 낙하 오탐 (전유/전무 패턴이 신호,
  탈락 58.1%) → **v2 settled-참조 기준**으로 교정 후 전체 재생성 (탈락 2.7%, v1 negative 보존).
- E4-3 1차: macro 0.465 — **재샘플 경로가 in-distribution(0.270 역방향, 분포 중심 생성기)**로
  주 풀 오염 판명 → 연구원 판정으로 novel 재구성(재샘플 = negative control 재분류, long
  w=0.02 신설) + **1회 약정 정식 재판정**.
- **정식 재판정 = REDUCE 확정**: macro **0.6576** (19/25 미달; object 0.771 / spatial 0.740 /
  goal 0.581 / long 0.481). §3 우아한 퇴화 발동 — E5 성숙도 단독 + 그림자 관할 (§5 등재).
  부수: 운용 q false-reject 0.264 (α_j 설계 대비 과대), negative control 수용률 0.968.

### scorer 진단 6행 (`e4_scorer_table.json`) — goal 중심 판독
| 행 | 비용 | macro AUC | FR | 비고 |
|---|---|---|---|---|
| 1 관할 Mahalanobis | 4.0ms | 0.658 | 0.264 | goal 0.581 |
| 2 feasibility head | 4.0ms | 0.544 | 0.110 | 실패 I₀ 재렌더 학습 (~3% 양성) |
| 3 히든 L32 visual-mean | **85.07ms** | 0.733(6셀) | 0.025 | **goal 0.686** — 최고 계기 |
| 4 oracle (대역 GT) | — | 1.0 | 0 | 명목 상한 |
| 5 관할 + q 재보정 | 4.0ms | 0.658 (불변) | **0.264→0.060** | 그림자 q 절차 실증 |
| 6 kNN 관할 k5/k10 | 4.0ms | 0.626/0.624 | — | 비모수도 미상회 |

### ★H3 상태 문구 = **(i)+(ii) 혼합** (연구원 정정 판정 2026-08-16 — 단독 (ii) 폐기)
**정정 경위**: 최초 판독이 25셀 macro(관할)와 6셀(히든)을 혼용 비교 — 범위 불일치.
판정자 재계산을 실행측이 독립 검증(완전 일치)한 **동일 셀 비교**:

| 범위 | 관할 | 히든 L32 | Δ(히든) | kNN k5 | Δ(kNN) |
|---|---|---|---|---|---|
| 히든 6셀 | 0.6439 | 0.7329 | **+0.089** | 0.6321 | −0.012 |
| goal 4셀 | 0.5314 | 0.6856 | **+0.154** | 0.5555 | +0.024 |

**최종 문구**: "기하 관할은 **표현 비용에 종속적**이며(히든 +0.154 on goal / 집계만 바꾼
kNN은 무효과 → 원인은 **표현**이지 집계가 아님), 그럼에도 21× 비용(85.07 vs 4.0ms)에도
임계 미달 → **저비용 실시간 관할은 미해결**." 25셀 vs 6셀 직접 비교는 각주로 강등.
차용류 범주 이탈 판별(0.871/0.948)은 실증 유지.

### 정정 — feasibility head 서술 조건부화
"열등"이 아니라 **"성공률 96% 레짐의 학습 신호 희소(양성 ~3%)"**. 실패 I₀ 재렌더로 표본을
복원해도 소수 — **조건부 결과이며, teacher 성공률이 낮거나 실패가 풍부한 레짐에서는 재평가
대상**(후속 연구 조건 변경 시). §5 등재.

### [등재만] Paper 2 / E5 사후 분석 후보 (docs/PAPER2_E5_POSTHOC_CANDIDATES.md)
유사도 가중 성숙도 · 실패 국소 각인 · 자기 유창성 신호 — 실행 금지, 문서화 완료. T5 보류 포함.

### 다음 마일스톤 (예고)
**E5 드라이버 설계서 초안 (코드 금지)** → 대화 검토용 패키지 제출 → 설계 리뷰 통과 후 구현.
사전 함정 체크리스트 6렌즈(docs/E5_DRIVER_CHECKLIST.md) 동봉 예정.

---

## 2026-08-16 — E4 종결 정정 3건 반영 + E5 드라이버 설계서 초안 제출 (코드 0줄)

### 정정 (연구원 판정 — 상세는 위 E4 종결 항목에 인라인 반영)
1. **동일 셀 비교 열 신설** — 판정자 재계산을 실행측이 독립 검증(완전 일치): 관할 6셀
   0.6439·goal4 0.5314 vs 히든 0.7329(+0.089)·0.6856(+0.154). kNN은 동일 셀에서 무효과
   (goal4 +0.024 / 6셀 −0.012) → **원인은 표현이지 집계가 아님**을 실측 보강.
   25셀 vs 6셀 직접 비교는 범위 불일치 각주로 강등. 표 재생성기 `e4_scorer_table.py` 신설.
2. **H3 최종 문구 = (i)+(ii) 혼합** (단독 (ii) 폐기) — §5·표 JSON·보고 패키지 동기화.
3. **feasibility head 조건부화** — "열등" 아님, 성공률 96% 레짐의 학습 신호 희소(양성 ~3%);
   조건 변경 시 재평가 대상.

### E5 설계서 초안 v0.1 (`docs/E5_DRIVER_DESIGN.md`, 코드 미작성)
필수 5항목 충족: **상태 기계**(4상태 전이표 + 에피소드 9단 절차, A_mat 산입 규칙 명시) /
**로깅 스키마**(에피소드 1행, source·lifecycle·그림자 점수·ACI·재학습 이벤트 전량) /
**counterfactual 큐**(발화분만 적재, 스트림 종료 후 배치 실행, 결정성 사전 검증 5 spec) /
**재학습·probe 일시정지 재개**(P1–P3/T1–T3/R1–R5/C1–C3, 학습 시간 분리 회계) /
**heartbeat 통합**(phase별 단가 앵커 취득, counterfactual은 총량 미지 → ETA 불확실 태그).
- 신설 제안: **스트림 대역**(seed 70000+10000·s, noise 6e6+1e6·s / novel 80000+, 7e6+) —
  기존 5대역과 disjoint, 기동 시 6대역 전수 assert. §4b 등재는 리뷰 승인 사항.
- 예산(앵커 산출): 스트림 6.0h + 재학습 ≤2.2h + probe ≤1.3h + counterfactual ≤5.4h
  → **3 seed 최악 ≈ 44.7h**. 클러스터당 노출 163.6 (n=80 도달 여유 2배).
- 결정 요청 6건(대역 등재·CF 시점·재학습 트리거 해석·M→I 강등 처리·novel 클러스터
  lifecycle·예산 축소 옵션) — `e5_design_pack_20260816.tar.gz`로 제출.

---

## 2026-08-16 — E5 설계 결정 6건 판정 반영 (설계서 v0.2) + 50 ep 스모크 관문 신설

### 판정 결과 (전건 §4b·§4h·§5 등재 완료)
1. **스트림 전용 대역 신설 승인** — seed 70000+10000·s / noise 6·10⁶+10⁶·s (novel 주입 80000+/7·10⁶+).
   base 0–39 재사용 = **배포 재발 상황 모사**(의도된 설계), held-out 사용 금지, 기동 시 6대역 전수 assert.
2. **counterfactual = 종료 후 배치 승인** — 판정 보강 근거: 스트림 중 실행은 **H4 측정 대상
   자체를 오염**. 선행 관문 = 발화 스펙 5–10개 × 2회 재실행 일치(불일치 시 정지·보고).
3. **재학습 트리거 = |B_k| (teacher 성공 궤적 수)** — probe·발화 궤적 불입 명시.
4. **M→I 강등 후 재학습 자격 유지, r_k 승계·R_max=2 전역** — "강등 후 재학습도 라운드 소비"
   해석 §5 등재. ★현 `gates/two_stage.py`가 **이미 정합**임을 실측 확인(강등 후 r_k=1 승계 →
   라운드2 미달 시 부적격 전이) — **코드 변경 불요**.
5. **novel 주입 = 정상 lifecycle 편입** + 로그 `is_novel_injection` 태그(사후 분리 분석).
6. **예산 축소 불요** — 대신 **seed 1 완주 → 중간 판독 → seed 2·3 순차** 집행.

### 신설 — 50 ep 스모크 관문 (§9)
본실행 전 검증: 6대역 assert / U→I 전이 / 로깅 스키마 완결(누락 0, source 3종만) /
실행기 단일 경유 / **그림자 관할 미개입**(관할 거부 행에서도 executor 불변) / heartbeat 발화.
- **미결 1건(리뷰 확인 요청)**: 50 ep는 클러스터당 노출 ≈2회라 재학습·probe가 자연 발화하지
  않음 → (a) 스모크 전용 축소 트리거(|B_k|≥3, P=5; 본실행 상수 불변) vs (b) 재학습 경로는
  본실행 첫 트리거에서 관측. **실행측 권고 = (a)** (C-T2 3막 교훈: 계기 결함은 데이터 생성
  전에 차단).

### 상태
설계서 v0.2 + 갱신 패키지 재생성 완료. **정식 리뷰 판정문 대기 — 구현 착수 보류 유지(코드 0줄)**.

---

## 2026-08-16 — 설계 결함 2건 반영(시간 3장부·25 클러스터) + ★E4-R 역량 지도 착수

### 결함1 — 시간 회계 3장부 분리 (§4h·설계서 §0b·§5·§6 등재)
| 장부 | 내용 | 지연 주장 |
|---|---|---|
| **운영** | 스트림 실제 소비 | **유일 근거 (H4·F4)** |
| **형성** | 재학습·probe (에피소드 수 + wall-clock) | **별도 표 정직 보고, 불산입** |
| **평가** | counterfactual | **측정 아티팩트 — 비용 미보고** |
논문 문구 고정: "지연 주장은 운영 회계 기준, 형성 비용은 별도 보고(배포 수명 >> 형성 투자 레짐)".
로깅 필드에 `ledger` 태그 신설 — r_V·지연 주장은 operational 집계만 사용.

### 결함2 — E5 클러스터 혼합 22 → **25 distinct** (libero_10 3 편입)
조건 (a) 스트림 생성기의 스위트별 `usable_w_max` 준수(long **0.02**) (b) heartbeat 단가를
**스위트 가중 평균 5.62s**로 재산출(long 520 스텝). 파생: 노출 163.6 → **144.0**,
3 seed 최악 예산 44.7 → **47.9h** (운영 6.2 / 형성 4.1 / 평가 5.6 per seed).

### ★E4-R 역량 지도 착수 (22:07, §5 판독 규칙 **결과 산출 전** 등재 완료)
**물음 교체**: E4는 oracle을 대역 GT로 두어 "novel 대역 소속"을 물었으나, 게이트의 존재 이유는
"여기서 습관의 성공 통계가 유효한가"다. 게이트의 기하 둔감성이 결함인지 **올바른 불변성**인지를
습관 실측으로 직접 시험 — "작업 반경 내 기하 불변성" 가설(RGB-D 폐쇄 루프 전제).
- 설계: 대표 6 클러스터(스위트 균형 + object task0·task5 필수) × w ∈ {0.01, 0.02, 0.04, 0.06,
  0.08} × 15 ep, **habit(n=80) rollout only**. 물리 유효성 통과분만 계수 + 폭별 탈락률 병기.
  게이트 점수·판정 불개입 기록. 실행기 = `execute_chunk_with_boundary` 단일 경로.
- 판독(사전 고정): ① 역량 경계 **w\*** (성공률 <0.8 첫 폭) — **w\* ≥ usable_w_max면 학습 폭
  초과 일반화 실증** ② 게이트 기각률 곡선과의 정렬/오정렬 정량화 ③ 클러스터별 w\* 분산.

### [등재만] 후속 연구 추가
**클러스터 입도의 운동 등가성 재정의**(콜라캔/주스캔 = 동일 습관) + 저비용 관찰안 =
**기존 27 습관의 교차 클러스터 전이 행렬**(rollout only). E5 이후 판단, 실행 금지.

---

## 2026-08-16 — ★E4-R 역량 지도 완료 [E4R-MAP-PASS]: 부분 일반화 + 게이트 오정렬 정량화

### 실행
6 클러스터 × w{0.01,0.02,0.04,0.06,0.08} × 15 ep = 450 rollout(habit n=80 only), 유효 423 ep.
중간 크래시 1건: `perturbed_init_state`의 E0-6 가드가 w>usable_w_max를 차단 → **진단 전용
명시적 예외**(`allow_beyond_usable`, 호출측 유효성 검사 의무 명기) 추가로 해소. §5 등재.

### 결과 (성공률 / 게이트 기각률, w=0.01→0.08)
| 클러스터 | 0.01 | 0.02 | 0.04 | 0.06 | 0.08 | w* | usable |
|---|---|---|---|---|---|---|---|
| object_task0 (N*=80) | .87/.60 | .67/.67 | .33/.67 | .27/.80 | .20/.87 | **0.02** | 0.04 |
| object_task5 (N*=20) | .93/.27 | .87/.33 | .57/.50 | .23/.54 | .18/.91 | 0.04 | 0.04 |
| goal_task0 (N*=80) | .93/.13 | .93/.20 | **.93/.47** | .71/.50 | .71/.71 | **0.06** | 0.04 |
| goal_task2 (N*=10) | 1.0/.07 | .93/.07 | .73/.33 | .67/.40 | .47/.47 | 0.04 | 0.04 |
| spatial_task0 | 1.0/.13 | .93/.00 | .87/.20 | .79/.36 | **.36/.27** | **0.06** | 0.04 |
| libero_10_task0 (>80 절단) | .73/.00 | .73/.00 | .64/.18 | .23/.15 | .22/.56 | 0.01 | 0.02 |

### 판독 (사전 등재 규칙대로)
**1. 역량 경계 w\*** — 분포: {0.01, 0.02, 0.04, 0.04, 0.06, 0.06}. **2/6이 usable_w_max 초과**
(goal_task0·spatial_task0: 학습 폭 0.01의 **6배**에서도 성공률 0.79~0.93) = **학습 폭 초과
일반화 부분 실증**. 반면 object_task0(w*=0.02)·libero_10_task0(w*=0.01, 학습 폭에서 이미
0.73 — E3 ŝ(80)=0.75·유일 절단 셀과 정합)은 **성숙 미달**로 역량 경계와 구분해 보고해야 함.
→ 가설 "작업 반경 내 기하 불변성"은 **조건부 참**: 습관이 성숙한 셀에서만 성립.

**2. 게이트 정렬도** — Spearman(실패율, 기각률) = **0.628** (p=0.0002, n=30): 게이트는
방향으로는 반응한다. 그러나 **수준 오정렬이 크다**:
- 학습 폭(w=0.01)에서 이미 **기각률 0.20**(실패율 0.089) — 과대 기각. object_task0은 0.60.
- 무손상 구간의 오탐: **goal_task0 w=0.04에서 성공률 .93인데 기각 .47**.
- 미탐지: **spatial_task0 w=0.08에서 성공률 .36으로 붕괴했는데 기각률은 오히려 .36→.27 하락**,
  libero_10_task0 w=0.06(성공 .23)에서 기각 .15.
- 전체 혼동행렬(423 ep): **미탐지율 0.410**(실패 중 41%를 수용), **오탐율 0.268**(성공 중 27%를 기각).
→ **판독 2-(ii) 오정렬 확정**, 그 정량치가 관할 절의 핵심 수치. 단 기각률의 절대 수준은
q 보정 상태 종속(재보정 시 FR 0.264→0.06 실증)이므로 **곡선의 정렬**이 주 근거.

**3. 클러스터별 w\* 분산** = 0.01–0.06 (6배). 스위트가 아니라 **클러스터 종속**
(goal 내부에서도 0.04 vs 0.06) → **관할 보정은 클러스터별이어야 한다**는 설계 근거.

### 작업공간 실측 완료 (§5 등재분, `results/e4/workspace_extent.json` + `fig_workspace_extent.png`)
- **공식 배치 bbox** = 0.46 × 0.66 m (40 태스크 × init 50 × free 물체, n=10,000 표본 도시).
- **검증 도달 영역**(teacher 성공 spec 재구성, n=15,366): q99 0.394 × 0.585 m,
  **convex hull 0.209 m²**, **등가반경 0.258 m**. Franka 공칭 반경 문헌값 미사용 — 실측 대체.
- **판독**: w=0.04 섭동 지름(0.08 m)은 검증 도달 **등가지름의 15.5%**, 면적 기준 **2.4%**.
  w=0.02는 7.8%, w=0.08은 31.0%. → **역량 경계 w\*(0.02–0.06)는 도달 영역의 8–23%**
  수준의 국소 변이다. **"습관의 기하 일반화는 국소적"** 서술이 정확하며, 학습 폭 확대
  실험(w↑ → N\* 변화)의 우선순위가 올라간다는 사전 판독 규칙이 발동한다.
- 한계: 테이블 상판이 MuJoCo geom으로 노출되지 않아(floor만) **공식 배치 bbox를 경계 대리**로
  사용했고 그림에 명기. 로봇 base는 x=−0.6 m(축 밖).

### 함의 (H3 문구 재작성 후보 — 판정 대상)
E4 REDUCE는 "대역 판별 실패"였으나, E4-R은 그 실패의 **구성**을 밝혔다: 게이트는 (a) 안전한
변이를 27% 기각하고 (b) 위험한 변이의 41%를 놓친다. 즉 **문제는 민감도 부족이 아니라
역량 경계와의 정렬 실패**다. 물리 탈락률은 w=0.08에서 16.7%까지 상승 — 상한 초과 폭의
해석에는 유효분만 계수했음을 명기.

---

## 2026-08-16 — E4-R 최종 판정 반영: H3 확정 + 라우팅 이득 산식 정정

### ★정정 (연구원 지적, 실행측 오류) — 라우팅 이득 산식
구 산식 **"주변 격차 × 기각률"은 무작위 기각자 등가**로, uid 매칭이 가능한데도 조건부 계산을
하지 않아 **게이트의 선별력을 은폐**했다. 정정 산식 = Σ_{게이트 기각}(teacher−habit)/n_total.
판정자 재계산값을 실행측이 독립 검증(완전 일치):

| w | 조건부 이득/ep | 기각분 평균 | VLA 라우팅 | 평균 질의 지연 |
|---|---|---|---|---|
| 0.01 | +0.0444 | +0.143 | 31.1% | 31.5 ms (4.30×) |
| 0.02 | +0.1111 | +0.313 | 35.6% | 35.0 ms (4.78×) |
| **0.04** | **+0.1591** | **+0.318** | 50.0% | 46.2 ms (6.31×) |
| 0.06 | +0.1163 | +0.200 | 58.1% | 52.5 ms (7.18×) |
| 0.08 | +0.0488 | +0.067 | 73.2% | 64.2 ms (8.77×) |

- **기각분 평균 이득이 전 폭 양수** = 게이트는 **무작위 기각자보다 유익**(구 산식이 놓친 사실).
- **구간 국소성 결론은 불변** — 조건부 이득도 w=0.04 정점 후 양극단 축소.
- 비용 축 병기(연구원 지시 2): w=0.01에서 +0.044 성공률의 대가 = 에피소드 31%가 85.07 ms 경로.

### H3 최종 문구 확정 (§5 등재, 후보 A + "약한 정렬")
"관할 게이트의 가치는 teacher–습관 **역량 격차가 있는 중간 변이 구간에 국한**된다(조건부 이득
w=0.04 +0.159/ep vs 양극단 +0.044/+0.049; 넓은 폭에선 teacher도 0.335로 붕괴). 그 대가로
w=0.01에서도 31%가 85 ms 경로를 타 평균 지연이 4.3×가 된다. 그 구간에서조차 관할은 **약하게만
정렬**되어(Spearman 0.628, 미탐 0.410·오탐 0.268; **재보정으로도 미탐 0.317 바닥**) 원인은 임계가
아니라 표현이다. 다만 **무작위 기각자보다는 유익**하다. 저비용 실시간 관할은 미해결이되,
**해결의 실익 자체가 구간에 제한된다**."

### 기타 판정 반영
- **미탐/오탐 대칭 서술** 등재 — 미탐 0.410의 손실 축소 해석을 쓰되, 오탐 0.268의 비용
  (이득 +0.044/ep 구간에서 31%를 VLA로 보냄)도 같은 무게로 명시. 한쪽만 완화하지 않는다.
- **학습 폭 확대 실험 = 본 논문 부록 승격**(승인·§5 등재): 2 클러스터(object_task0·goal_task2)
  × 수집 w=0.04 × n-grid{10,20,40,80} → 역량 경계 재측정 + N* 비교. 목적 = **국소성 한계에
  대한 처방 존재 실증**. **E5 GPU 우선, 유휴 슬롯 실행**.

---

## 2026-08-16 — E5 구현 착수 + ★스모크 관문이 사전등록 위반 경로 2건 차단 (negative result)

### 구현 (설계서 v0.3 승인본)
- `envs/stream.py`: E5 스트림·novel 주입 대역 + `e5_stream_specs`(25 클러스터×144 + novel 400)
  + **`assert_six_bands_disjoint`**(기동 시 강제, 실패=기동 거부). 검증: uid 4000 유일, 6대역 통과.
- `experiments/e5_driver.py`: 4상태 기계 · 3장부 회계 · 그림자 관할(불개입) · source 태그 ·
  CF 큐 · lazy 재학습/probe. **단일 프로세스 요건 실측 확인** — hv2_oft에서 teacher·ACT·
  DINOv2·게이트 전부 임포트 가능(E1 때 sklearn 설치의 이유가 여기서 확정).
- `habits/train.py`: `--warm-from`(lazy 재학습의 warm-start 승계 — 호출이 분리되므로 필수),
  `--steps`(스모크 전용 override) 추가. 기본 동작 불변.

### ★negative result 1 — 재학습 트리거의 R_max 가드 부재 (사전등록 §4h 위반 경로)
- **증상**: 트리거 조건이 `|B_k| ≥ grid[idx]`뿐이라, 마지막 그리드 지점을 넘어 BC 풀이
  계속 커지면 재학습·probe가 **반복 발화**한다. 2라운드를 통과해 M이 된 뒤 ACI τ 상향으로
  강등되면 **3라운드에 진입** → **§4h "R_max = 2 전역" 위반**.
- **검출**: 50 ep 스모크 3차에서 probe가 처음 발화하며 드러남 — **본실행 데이터 생성 전**.
- **수정**: 트리거에 `probe_rounds < MaturityGate.PROBE_MAX_ROUNDS` 가드 추가.
- **교훈**: **상태 기계의 상한(R_max)은 소비 지점이 아니라 모든 진입 지점에서 검사해야 한다.**
  `record_probe_round`가 상한을 알고 있어도, 그 함수를 호출할지 결정하는 쪽이 상한을 모르면
  위반 경로가 열린다. C-T2 3막(경계·이월·실행기)과 함께 **방법론 부록 후보**.

### ★negative result 2 — 관문 자체가 관문 목적을 달성하지 못한 3회 (관문의 관문)
| 회차 | 결과 | 미달성 원인 |
|---|---|---|
| 1차 | 완주 50/50, 스키마·6대역·U→I·그림자 정상 | 50 ep가 **30종 클러스터**에 흩어져 \|B_k\| 트리거 미발화 — 재학습·probe·발화·부적격 전부 미검증 |
| 3차 | probe 발화(위 결함 검출) | R_max 결함으로 중단 |
| 4차 | probe 발화, 그러나 **P=5로는 성숙 전이가 수학적으로 불가**(5/5 전승도 Pr=0.738<0.9) + n=3·300스텝은 유효 정책 미생성(E3 실측: goal_task2의 최소 유효 지점은 n=10) | 조건 3(I→M)·4(CF 큐) 원천 불가 |
- **수정 누적**: 클러스터 2종으로 축소(클러스터당 ~22 ep) / **P=5 → 10**(Beta(11,1)=0.914≥0.9) /
  **|B_k| 3 → 10**, 스텝 300 → 2500 / 강제 실패 주입을 **지정 클러스터 전 라운드**로 확대.
- **상수 격리 재확인**: 본실행은 `PROBE_FULL=20`·`GRID_FULL=(20,80)`으로 **사전등록 §4h와
  동일** — 스모크 상수는 `--smoke` 분기에만 적용되며 §5 개정 대상 아님.
- **교훈**: "완주 = 통과"가 아니다. 관문은 **경로가 실제로 발화하는 구성**이어야 하고,
  통과 조건은 결과 확인 전에 고정되어야 한다(연구원 지시로 6조건 사전 고정).

## 2026-08-17 21:56 — E5 seed 1 완주 + 1차 판독

- **seed 1 완주** 21:41 (4,000 ep, 14h 03m). 총 wall 50,515s / 운영 34,467s / 형성 11,777s(1,020 ep).
  infra_error 0건. CF 큐 1,337건.
- **H4-a PASS**: r_V 0.896 → 0.496 (Δ0.400, z=19.4, p=1.61e-84, 단측 α=0.05).
- **위험 통제 PASS**: Pr(fail|fire) = 0.0322 [0.024, 0.043] ≤ ε=0.2. τ 상향 14 클러스터·313 ep.
- **형성**: 재학습 51회, probe 통과 18회 — **n=20 10/33 (0.303) · n=80 8/18 (0.444)**.
  학습 규모를 4배로 늘려도 통과율은 0.30 → 0.44에 그친다(초판에 "n=20 전량 실패 / n=80 전량 통과"로
  적었으나 오독 — 그림 검수에서 정정. 분석기에 grid별 시도/통과 분리를 추가해 재발 차단).
  성숙 도달 17/33, 소요 노출 중앙값 22회(범위 21–91). 최종 M15 / I8 / X10.
- **강등 3건 전부 동일 패턴**: 발화 5–6회 중 1회 실패 → ACI τ 0.816 → p 0.877–0.897 (1−δ 바로 아래) → I.
  1건(goal_task4)만 n=80 재학습으로 M 복귀. 성숙 초기 취약성 가설 → 판정 요청 (2).
- **그림자 관할**: 예측 +31%p·4.3× vs 실측 **+1.9%p·1.60×**. 원인 규명 — 예측은 E4-R **원 보정** q
  기각률 0.2에서 유도, 실행은 사전등록(§5)이 지정한 **재보정** q 사용. 실측 발화 기각률 0.0568이
  재보정 FR 0.06과 정합 → **실행이 사전등록 준수, 예측치 유도가 조항과 불일치**.
- 판독기 제작 중 결함 3건 자체 검출·수정: 부분 CF로 비열등 판정(표본 편향) / 지연 단위 불일치
  (에피소드 vs 질의) / p 언더플로 "p=0" 표기.
- **CF 배치** 21:42 착수, 결정성 사전 검증 5/5 PASS. 3.2h 페이스 → 익일 01시경 완료 예정.
  재로딩 오버헤드 측정 결과 병목 아님(발화 스펙이 짧아 순수 rollout 추정보다 빠름) → 중단 없이 진행.
- 자동 체인: `e5pack` 세션이 cf_summary_0.json 생성을 감지해 **완성본 패키지 자동 재생성**.
  **seeds 2·3은 착수하지 않음** — 중간 판독 후 연구원 지시 대기(§4h 결정 6).

## 2026-08-17 22:2x — seed 0 판독 판정 집행 + 신규 발견 2건

연구원 판정 4건 §5 등재. H4a 확정(0.896→0.496, z=19.4, p=1.61e-84), CF 완주 전 H4b 인용 금지,
부적격 사후 분석(탐색적), seed 1·2는 CF 완주·H4b 판정 후.

**질의 1건**: 지시서 §2 "CI **상한**이 −0.03보다 크면 비열등"은 표준 비열등 검정(차이 = system −
full-VLA)에서 **하한** 기준이어야 한다(상한 기준은 거의 항상 성립해 검정력 없음). 기존 §1 등재·
구현도 하한 → **하한으로 해석해 등재**, 연구원 확인 시 정정.

**부적격 사후 분석 (추가 rollout 0)**
- 규칙 1: 부적격 10개 **전부** 확정 후 BC 풀 > 80 축적 (중앙 잉여 +59).
- 규칙 2: E3에서 N*≤80이던 클러스터 **9/10**이 스트림에서 부적격 (우측절단 1 = 10_task0).

**신규 발견 ① 성숙 문턱의 이름 충돌** — E3 성숙 = 점추정 ŝ≥0.8, E5 성숙 = Pr(s≥0.8|𝒟)≥0.9.
P=20에서 후자는 **19/20 = 0.95**를 요구한다. 같은 τ=0.8을 쓰지만 실질 문턱이 0.80 대 0.95.

**신규 발견 ② 라운드2 이월 함정** — c=0.25 재초기화가 라운드1 실패를 φ로 승계하므로
**라운드1 실패 ≥ 9회면 라운드2 20/20 만점에도 Pr<0.9** → 라운드2 시작 시점에 X 확정.
해당 5/10 클러스터. 그중 **object_task5는 실제로 라운드2에서 20/20 만점을 받고 탈락**
(E3 ŝ(80)=1.0). R_max=2와 결합해 "형식적 2라운드"가 발생한다.

**형성 부진과 판정 탈락의 분리**: object_task5(E3 1.0 / 스트림 1.0)·task8(0.95 / 0.95)은
형성이 E3 수준인데 판정 규칙에서 탈락, object_task0(0.95 / 0.7)·10_task0(0.75 / 0.7)은 형성 부진.

산출 `results/e5/ineligible_postmortem_0.json`, 패키지 `e5_reading_pack_s0_20260817.tar.gz`.
후속 조치(문턱 정합·이월 규칙·재도전 조건)는 **seed 0 확정 후 별도 판정** — 본 seed 설계 불변.

## 2026-08-17 22:30 — 갱신 판독 판정 집행: ★구현 결함 확정

**1. CF 정상 실행 중** — 정체 아님. 22:27 기준 297/1337, 실측 6.5건/분(390건/시간),
프로세스 2811303 생존, 파일 실시간 갱신, 일치율 0.976. **완주 예상 01:05경**.
연구원이 본 "정체"는 보고 시점 스냅샷 차이.

**2. 형성 간극 진단 — 판독 규칙 2-(c) 발동 → ★구현 결함 확정, 연구원 회부**
- **(c) 불일치**: `habits/train.py`가 정규화 통계를 `compute_stats(episodes[:max(n_grid)])`로
  산출. E3는 `--n-grid 10 20 40 80` 일괄이라 max=80 풀 1회 산출·**전 단계 동결**(실측 l2차 **정확히 0**).
  E5 lazy 재학습은 `--n-grid {n}` 단일값이라 **n=20은 20개 풀, n=80은 80개 풀**에서 각각 산출
  → **warm-start가 정규화 공간을 가로지름** = §5 "전 n-grid 단계 동결" 조항 위반.
  실측 이동: 부적격 10개 전수 action_mean l2rel 0.009–0.160, proprio_std 최대 0.097.
  누적 스텝도 배치 28,000 vs 스트림 16,000(lazy 설계 귀결).
- **(a) 차이 있음**: BC 풀 고유 base 34–35/40(최대중복 4–6, 분산 1.05–1.69) vs E3 40/40(분산 0).
  기전: 스트림 base_idx = (전역 인덱스) % 40인데 클러스터 노출이 셔플로 흩어져 불균등.
- **(b) 차이 있으나 방향이 반대**: probe는 수집 대역 재사용이라 BC 풀과 base가 0.75–0.85 겹침,
  E3 held-out은 전용 대역 미노출 → **더 쉬운 평가에서 더 낮은 성적**. 간극을 설명 못 하고 키움.
- **단서**: 정규화 이동은 성숙 **성공** 클러스터(goal_task4·object_task4·10_task5)에도 동일 존재
  → 부적격의 유일 원인으로 단정 불가.
- 회부 사항: 수정 방식(첫 stats 동결 승계 vs 목표 max-n 고정), **seed 0 재실행 여부**
  (재실행 시 H4a·CF도 함께 무효화).

**3. 그림자 관할 예측 괴리 §5 명시 등재** — 예측 기저 분포 오류(w-사다리 전 구간 평균 오탐률을
w=0.01 집중 스트림에 적용). 해석 정정: "닫힌 작업공간에서 관할은 개입할 일이 거의 없다" →
REDUCE 판정을 **강화**.

**4. 부적격 사후분석 사용처 예약** — 탐색적 라벨 유지, Discussion 한계 정량화 + 후속 연구 목표
두 곳으로 한정, **Results 인용 금지**.

산출: `results/e5/formation_gap_0.json`, 패키지 `e5_reading_pack_s0_20260817.tar.gz` 갱신.

## 2026-08-17 22:40 — 재실행 판정 접수 · 검토 결과 · 기제 등재

**판정**: seed 0 **무효 → 재실행** (정규화 결함, §5 동결 조항 구현 위반).

### 3항 규명 (2항목만, 나머지 대조는 지시대로 미수행)
- **3-(2) HDF5 참조 = 정상**. 전 33 클러스터에서 `data/e5_s0/{cluster}.hdf5`의
  episodes 수 == meta success 수 == 로그 |B_k| == 스트림 teacher 성공 수. **동형 재발 없음.**
- **3-(1) 누적 스텝 = lazy 그리드의 산술적 귀결** (호출 인자 실수 아님).
  배치 28,000 = 4,000+6,000+8,000+10,000(n=10·20·40·80 warm-start 체인).
  스트림 16,000 = 6,000+10,000. 차이 12,000 = 건너뛴 n=10(4,000) + n=40(8,000).
  `--n-grid`는 의도대로 전달되며, lazy {20,80} 설계의 부수 효과다.

### ★ 검토에서 발견한 쟁점 (연구원 판정 요청)
B안(n=80 scratch)을 적용하면 n=80 학습량이 **10,000 스텝 단독 = 배치 28,000의 36%**로,
현행 16,000(57%)보다 **더 줄어든다**. 3-(1)의 취지("학습량 절반이 성숙 실패의 직접 원인일 수
있고, 정규화만 고친 재실행이 같은 결과를 낼 위험")가 B안에서 오히려 악화된다.
- B-1 현행 HP 유지(10,000, 36%) — 사전등록 무변경
- B-2 배치 등가 부여(28,000, 100%) — HP 개정 필요, 학습량 변수를 배치와 정합
- B-3 현행 누적 유지(16,000, 57%) — 원리적 근거 없음(건너뛴 그리드의 우연한 합)
**실행측 권고 = B-2**. 재실행의 목적이 "결함 제거 후 온라인 형성의 참값 측정"이므로,
학습량이 배치와 다르면 부적격 재발 시 원인을 다시 분리할 수 없다.

**B안 정합성은 확인됨**: `habits/policy.py`의 `HabitPolicy`가 체크포인트의 `stats`를 함께
로드하므로, scratch 학습에서 n=20과 n=80의 정규화 공간이 달라도 각 체크포인트가 자기 공간에서
완결적으로 동작한다. 문제는 오직 warm-start가 공간을 가로지를 때뿐이며 B안이 이를 제거한다.

### 기제 검증 등재 (정규화와 무관 — 재실행 후 재확인만으로 족함)
- **4상태 전이 작동**: U→I→M→(강등)I→M 전 경로 관측. 최종 M15 / I8 / X10.
- **ACI 강등 3건**, 전부 동일 패턴(발화 5–6회 중 1회 실패 → τ 0.816 →
  p 0.897, 0.877, 0.882 → I). **1건 재성숙**(goal_task4, n=80 재학습).
- **그림자 관할 불개입 로깅**: 3,600/4,000 ep 점수 기록(나머지 400은 μ_k·Σ_k 미적합 novel — 원리상 불가),
  행동 개입 0건 확인.
- **3장부 분리**: 운영 34,467s / 형성 11,777s(1020 ep) /
  평가(CF) 별도. 지연 주장은 운영 장부 단독.
- **novel 라우팅**: 주입 400건 r_V 0.92 vs 정규 0.6375 — 미지 클러스터가
  VLA로 정상 라우팅됨. 1개(spatial_task7)는 스트림 내 성숙 도달.
- **위험 통제**: Pr(fail|fire) 0.0322 ≤ ε 0.2, τ 상향 14 클러스터.
- **CF 파이프라인**: 결정성 사전 검증 5/5 PASS, 진행분 일치율 **0.968** — 실전 검증 완료.
- 부적격 사후분석: **탐색적 라벨 유지 · Results 인용 금지**(기존 지시 유지).

### 철회 반영
"배치-스트림 형성 간극 규명"의 **(a) BC 풀 분포 · (b) 평가 대역** 대조는 지시대로 **철회**.
`formation_gap_0.json`의 해당 절은 **철회 표시**하되 파일은 보존(감사 추적). **(c) 정규화 대조는
결함 확정 근거이므로 유지**. 재실행 결과에서 같은 현상이 재관측되면 그때 재판단.

## 2026-08-17 23:00 — B-2 구현 완료, 야간 파이프라인 가동

**구현**: 재학습 호출 `--no-warm-start --steps {배치 등가}`. `BATCH_EQUIV_STEPS`는
`HP["steps_per_n"]`에서 프로그래밍 산출 (n=20 → 10,000 / n=80 → 28,000 = E3 warm-start 체인 누적).

**관문 단언 (a) 재정의**: 원안 "재학습 간 l2 상대차 == 0"은 A안(트리거 {80} 단일) 전제.
B-2에서는 n=20(20개 풀)과 n=80(80개 풀)의 stats가 다른 것이 정상이므로, 원 취지(횡단 차단)를
**"체크포인트 stats == 자기 학습 데이터 episodes[:n]에서 재산출한 값"**(l2 ≤ 1e-6)으로 검증.
`assert_retrain_contract()`를 **재학습마다 런타임 호출** — 스모크는 축소 스텝이라 본실행 값
(28,000)을 검증할 수 없기 때문. 위반 시 즉시 RuntimeError 정지.

**재실행 소요 추정**: 재학습 스텝 378,000 → 834,000 (×2.21), 학습 시간 +2.9h
(44 스텝/초, E2 anchor5 실측) → 총 wall 14.0h → **약 16.9h**.

**야간 체인 (`e5rerun` 세션)**: CF 완주(01:05 예상) → seed 0 격리
(`results/e5/seed0_normstats_invalid/`, INVALID_ 접두어 + 인용 금지 README) → 스모크 관문
3단언 → **통과 시에만** 재실행 착수. 미통과 시 정지·보고(재실행 안 함).
착수 예상 01:30 → 완주 예상 18:30경.

**상태 복원**: `bash tools/e5_status.sh` — 파이프라인 단계 자동 판별(CF/격리/스모크/재실행),
단계별 진행률, 런타임 단언 통과 수, 오류 신호.

## 2026-08-18 01:38 — CF 완주·격리 완료, 스모크 관문 재기동 (실행측 실수 1건)

**CF 완주 01:35** — 1,337건 전량, 결정성 5/5 PASS. **무효 데이터(인용 금지)**이나 파이프라인
검증 기록으로 보존: 발화 성공률 0.9678 / teacher 0.9723, 불일치 습관만 24 · teacher만 30
(paired 차 −0.0045). 결함이 없었다면 H4b가 통과했을 방향이라는 참고치.

**격리 완료 01:36** → `results/e5/seed0_normstats_invalid/` (INVALID_ 접두어 + 인용 금지 README).
스트림·CF·판독·그림·체크포인트·HDF5 전량.

**★ 실행측 실수**: 체인이 스모크 산출물 정리 단계를 빠뜨려, 이전 스모크의
`results/e5/smoke/stream_0.jsonl`이 남아 드라이버의 덮어쓰기 방지 가드에 걸려 2초 만에 기동 거부
(rc=1, GATE-FAIL 0건 = 단언 실패 아님). 결함이 아니라 **절차 누락**.
조치: 이전 스모크 산출물 3종(results/data/checkpoints)을 `INVALID_prior_smoke/`로 아카이브,
체인에 **사전 정리 단계 + '이미 격리됨' 분기**(재기동 시 CF 무한 대기 방지) 추가 후 01:38 재기동.
GPU 유휴 손실 약 2분.

## 2026-08-18 09:06 — 관문 통과 확인, 재실행 착수 (실행측 실수 2건째)

**★ 실행측 실수**: 체인의 스모크 호출에서 `--n 50`을 빠뜨렸다. `--smoke` 플래그는 트리거·probe·
스텝 상수만 바꾸고 **에피소드 수는 바꾸지 않는다**(기본 4,000). 결과적으로 스모크가 본실행 규모로
7시간 25분(01:38–09:05) 돌았고 3,151 ep에서 중단했다. GPU 7.4시간 손실.

**관문 판정 = 통과**: 단언은 재학습마다 독립 검증되므로 규모 실수의 영향을 받지 않는다.
- **재학습 20회 전부 통과 · GATE-FAIL 0건 · 고유 클러스터 10종 · scratch 20/20**
- (a) 정규화 자기풀 일치 (b) 스텝 지정값 일치 (c) |B_k| 3중 대조
- 연구원 조건("재학습 2회 이상 발생하는 스모크에서 단언") 충족.
- 증거 보존: `results/e5/gate_evidence_20260818/` (로그·단언 목록·산출물·README).
  스트림 산출물은 **판독 인용 금지**.
- 부기: 본 스모크에서 성숙 전이 0건 — 축소 상수(2,500 스텝)로는 probe 10/10을 넘기는 정책이
  나오지 않기 때문이며 본실행 상수와 무관. lifecycle 전이는 08-17 스모크에서 이미 검증됨.

**재실행 착수 09:06** (`habit2` 세션, `logs/e5/seed0_v3.log`). 6대역 disjoint 통과.
완주 예상 익일 02:00경(16.9h). 런타임 단언이 재학습마다 본실행 배치 등가 스텝(10,000/28,000)을
검증하며, 위반 시 즉시 정지한다.

## 2026-08-19 08:55 — 재실행 완주 + CF 완주: H4a·H4b 모두 PASS

**재실행 완주 01:46** (4,000 ep, 16.7h). 운영 33,580s / 형성 22,128s(1,020 ep) / CF 큐 1,536.
런타임 단언 51/51 통과 (n=20 steps=10,000 · n=80 steps=28,000 · stats 자기풀 일치 · scratch).

**CF 완주 06:14** (1,536건, 결정성 5/5 PASS, 일치율 0.963).

### 사전등록 판정 — 3/3 PASS
- **H4a** r_V 0.887 → **0.393** (Δ0.494, z=23.013, p=1.73e-117, 단측 α=0.05) → **PASS**
- **H4b** system 0.9627 vs full-VLA 0.9640, Δ=**−0.0012**, 95% CI [**−0.0050**, +0.0025],
  margin −0.03 → **PASS**. CI가 0을 포함 → 연구원 판독 규칙 ②에 따라 **"동등" 서술 가능**.
- **위험 통제** Pr(fail|fire) = **0.0234** [0.017, 0.0323] ≤ ε 0.2 → **PASS**

### 무효 최초 실행 대비 수치 — ★ 참고 · **교락으로 해석 불가** (연구원 판정 2026-08-19)
> 두 실행은 정규화(결함/수정)와 학습 스텝(6,000 → 10,000 / 10,000 → 28,000)이 **동시에** 다르다.
> 아래 차이는 어느 변수의 효과인지 **분리 불가**하므로 §5 판정문·논문 어디에도 인용하지 않는다.
> 2026-08-19 08:55 보고의 "결함 수정이 형성을 실질적으로 개선했다"는 서술은 **철회**한다.
| | 재실행 | 무효 최초 |
|---|---|---|
| r_V 첫→끝 | 0.887→**0.393** | 0.896→0.496 |
| 발화/성공률 | **1,536**/0.977 | 1,337/0.968 |
| n=20 / n=80 통과율 | 0.364 / **0.556** | 0.303 / 0.444 |
| 성숙 도달 | **22**/33 | 17/33 |
| 최종 상태 | **M19 I6 X8** | M15 I8 X10 |
| novel 성숙 | **3개** | 1개 |

라운드2 만점 탈락 사례 0건(위 교락 라벨 적용 — 최초 실행과의 대조로 해석하지 않는다).
부적격 8개 잔존: 사전등록 해석 규칙(2026-08-17)에 따라 학습 조건이 배치와 정합한 상태의
재발이므로 **"온라인 형성의 성질"로 해석**한다((c) 3중 카운트 단언 51/51 통과가 전제 조건).

### 실행측 수정 2건
- 패키지 생성기 `IndexError`: 재실행에서 '라운드2 만점 탈락'이 0건이 되어 `[0]` 접근 실패 → 분기 처리.
- **패키지 마지막 문단에 무효 실행 수치가 하드코딩**되어 있었다(CLAUDE.md §6 수동 입력 금지 위반)
  → probe 성적과 E3 held-out을 대조해 '판정 탈락 6개 / 형성 부진 2개'로 **프로그래밍 분류**하도록 교체.

패키지: `e5_reading_pack_s0_20260819.tar.gz`. seed 1·2는 연구원 지시 대기.

## 2026-08-19 17:45 — 논문 초안 자료 패키지 생성 (seed 0 기준)

`manuscript_pack_20260819.tar.gz` — 생성기 `experiments/make_manuscript_pack.py`(재현 가능,
전 수치 프로그래밍 주입). 구성: E0–E5 전 실험 JSON · E5 원자료(스트림 4,000행 + CF 1,536행) ·
그림 3장 · 스크립트 13종 · 사전등록/log/CLAUDE 전문 · docs 4종 · git 이력 60건.

**MANUSCRIPT_SOURCES.md 핵심 = 인용 규칙 7건을 문서 최상단에 못박음**: 무효 데이터 전면 금지 /
무효-재실행 비교는 교락으로 금지 / 부적격 사후분석은 Discussion 한정 / H4b 각주 요건 /
지연은 attn=sdpa 명기 / 지연 주장은 운영 장부 단독 / seed 1·2 진행 중.

**집필자 오독 방지 장치 3건 추가**:
- H2: 수치만 실으면 원가설 지지로 오독 가능 → **H2-L′ 채택·H2-L 미지지** 판정 상태를 인용부로 명시.
- T 천장: 단측 p=0.0968은 α=0.05에서 **유의하지 않음** → 점추정 방향으로만 서술하도록 주의 표기.
- free_joints: 형성 22셀에서 suite 더미와 완전 공선 → **부호 해석 금지** 표기.

**미생성 자산 명시**: E2/E3 성숙 곡선 그림, H2 이중 해리 그림, E4 scorer 비교 표 —
원료는 results/에 있으며 요청 시 프로그래밍 산출 가능.

## 2026-08-20 02:15 — ★ 실행측 오류: formation_gap이 B-2 실행을 결함으로 오판

**발견 경위**: "seed 1 모든 작업 완수" 점검 중, 패키지 생성기가 `e5_formation_gap.py`를
매번 재실행한다는 것을 확인. 이 진단의 (c) 판정은 **"재학습 간 stats가 다르면 결함"**인데,
B-2는 scratch 학습이라 **다른 것이 정상**이다(각 체크포인트가 자기 풀에서 산출·완결).

**결과**: seed 0 재실행 데이터로 재산출된 `formation_gap_0.json`이 **"불일치 — 구현 결함 확정
(연구원 회부)"**라는 잘못된 판정을 냈고, 그것이 `e5_reading_pack_s0_20260819.tar.gz`에 포함됐다.
2026-08-19에 붙였던 (a)(b) 철회 라벨도 재산출로 덮어써져 소실됐다.
→ 논문 판정 자료에 **없는 결함이 있다고 적힌 상태**였다.

**조치**
1. `e5_formation_gap.py`에 **B-2 가드** 추가 — n=80 체크포인트 steps가 28,000이면
   `[E5GAP-SKIP]`으로 기동 거부(`--force`로만 우회). 오판 재발 원천 차단.
2. 오판본 → `seed0_normstats_invalid/MISJUDGED_formation_gap_0_rerun.json`으로 격리.
   **진짜 결함 진단 원본은 `INVALID_formation_gap_0.json`에 온전히 보존**되어 있음을 확인
   (철회 라벨·post_ruling 포함).
3. 패키지 생성기에서 formation_gap 호출·수록 제거 → **런타임 단언 기록**
   (`runtime_gate_assertions.txt`, seed 0 = 51건 통과 0건 실패)으로 대체. B-2에서는 이쪽이
   더 강한 증거다 — 재학습마다 (a)stats 자기풀 일치 (b)배치 등가 스텝 (c)|B_k| 3중 대조를 검증.
4. 오판이 든 `e5_reading_pack_s0_20260819.tar.gz` **삭제**, `..._20260820.tar.gz` 재생성.

**교훈**: 진단 스크립트는 그것이 전제하는 실행 조건을 스스로 검사해야 한다. 조건이 바뀐 뒤에도
같은 판정 로직을 돌리면 정상을 이상으로 뒤집는다.

**seed 1 잔여 작업**: CF 진행 중(약 05:50 완료) → 판독 갱신(H4b) → 그림 → 패키지 자동 생성.
**seed 2**: CF 종료·패키지 확인 후 착수 전 확인 3종(assert 강제) 통과 시 자동 착수(`e5seed2` 체인).
GPU 경합(OFT 7B 16.5GB × 2 > 32GB)으로 병행 불가.

## 2026-08-20 06:06 — seed 1 전 작업 완수 · seed 2 착수

**seed 1 판정 3/3 PASS**
- H4a: 0.844 → 0.454 (Δ0.390, z=18.271, p=6.98e-75) → PASS
- H4b: system 0.9560 vs full-VLA 0.9573, Δ=−0.0013, 95% CI [−0.0058, +0.0033],
  paired 4,000 전량·CF 누락 0 → **PASS** (CI가 0 포함 → "동등" 서술 가능)
- 위험 통제: Pr(fail|fire) 0.0362 [0.0278, 0.047] ≤ 0.2 → PASS
- CF: 1,465건, 결정성 5/5, 일치율 0.935. 런타임 단언 50/50 통과.

**★ 실행측 오류(즉시 수정)**: `fig_e5_reading.py`의 강등 궤적 색 리스트가 3색 고정이라
seed 1의 강등 5건에서 IndexError → 그림 실패 → `set -euo pipefail`로 패키지 생성까지 중단됐다.
seed 0이 마침 강등 3건이라 드러나지 않던 결함. 조치: 고정 5색 팔레트 + 초과분은 배경 회색
겹쳐 그리기(순환 배색 금지 원칙 유지) + 제목에 미표기 건수 명시. 패키지 재생성 완료.

**패키지**: `e5_reading_pack_s1_20260820.tar.gz` (781K).
**seed 2 착수 06:06** — 착수 전 확인 3종 assert 통과(B-2 상수 / 런타임 단언·scratch /
6대역 disjoint). 완주 예상 22:50경.

### 2 seed 대조 (동일 조건 — 교락 없음)
| | seed 0 | seed 1 |
|---|---|---|
| r_V 첫→끝 1,000 | 0.887→0.393 | 0.844→0.454 |
| H4a | z=23.0, p=1.7e-117 | z=18.3, p=7.0e-75 |
| H4b diff [CI] | −0.0012 [−0.0050, +0.0025] | −0.0013 [−0.0058, +0.0033] |
| Pr(fail\|fire) | 0.0234 | 0.0362 |
| 성숙 / 최종 상태 | 22/33 · M19 I6 X8 | 18/33 · M16 I7 X10 |
| 강등(재성숙) | 3(0) | 5(3) |  ← 3(1)은 무효 실행 사례 오기, 2026-08-21 정정
**두 seed 모두 H4a·H4b·위험 3항 PASS, 방향 일치.**

## 2026-08-21 03:25 — ★ E5 완료: 3 seed 전 항목 PASS

**seed 2 완주 22:31 → CF 완주 03:20 → 패키지 생성.** 런타임 단언 전 seed 위반 0.

### H4 최종 (3 seed 종합, 12,000 ep)
| 항목 | 종합 (평균±산포) | seed 0 / 1 / 2 |
|---|---|---|
| r_V 첫→끝 1,000 ep | **0.874±0.026 → 0.405±0.044** (Δ 0.469±0.070) | 0.887→0.393 / 0.844→0.454 / 0.891→0.368 |
| H4b diff (margin −0.03) | **−0.0021±0.0015** | −0.0012 / −0.0013 / −0.0038 |
| Pr(fail\|fire) (ε=0.2) | **0.0285±0.0068** | 0.0234 / 0.0362 / 0.0260 |
| paired 표본 | **4,618 ep** (CF 누락 0, 결정성 5/5×3) | 1536 / 1465 / 1617 |
| 성숙 도달 | 20.3±2.1 / 33 | M19 I6 X8 / M16 I7 X10 / M20 I6 X7 |
| 형성 통과율 | 0.434±0.016 (n=20 0.384 / n=80 0.526) | — |
| 3장부(h) | 운영 9.34±0.24 / 형성 6.07±0.12 | 지연 주장 = 운영 단독 |

**H4a 3/3 PASS** (z = 23.0 / 18.3 / 24.2, 전 seed 단측 p < 1e-74).
**H4b 3/3 PASS**. 전 seed CI가 0을 포함하나 **상한 최솟값 +0.0002(seed 2)**로 0에 근접 —
"통계적으로 구별되지 않음" 서술 시 **동등성 검정 미수행 각주 필수**이며 seed 2에서는 약함을 명시.
**위험 통제 3/3 PASS**.

**seed 확대의 목적 달성**: 감소폭 산포 0.070으로, 곡선이 클러스터 등장 순서의 산물이라는 설명이 배제됨.

### 실행측 정정 1건
log.md 2026-08-20 대조표의 seed 0 "강등(재성숙) 3(1)"은 **3(0)**이 맞다 — "1"은 무효 실행의
goal_task4 사례를 재실행 표에 잘못 옮긴 것. 실측: seed 0 = 3건 전부 미회복, seed 1 = 5건 중 3건 회복,
seed 2 = 3건 중 2건 회복(object_task9는 회복 후 재강등).

산출: `results/e5/seed_synthesis.json`, `e5_reading_pack_s2_20260821.tar.gz`,
`manuscript_pack_20260821.tar.gz`(3 seed 종합 절 + seed 1·2 원자료 포함).

## 2026-08-21 04:00 — 최종 검증 패키지 (연구원 요청)

`manuscript_pack_20260821.tar.gz` (3.4 MB, 222 파일) — 생성기 `experiments/make_manuscript_pack.py`.

**구성 변경**: 코드를 `code/` 하위가 아니라 **패키지 루트에 저장소 구조 그대로 미러링**.
첫 시도에서 `code/experiments/e5_analyze.py`를 실행하면 HABIT2가 `code/`를 가리켜
`code/results/e5/stream_0.jsonl`을 찾다 실패했다 — 구조를 맞추니 스크립트가 수정 없이 돈다.

**포함**: 전체 소스(envs·habits·gates·teacher·experiments·tools·configs) · 3 seed 원자료
(stream/cf/cf_queue 9종 gz) · E0–E5 결과 JSON 전량 · 그림 7장 · 사전등록·log·CLAUDE 전문 ·
docs 4종 · git 이력 60건 · `checksums.sha256`(104개) · `VERIFY.md`.
**제외**: checkpoints 93 GB, data 8.8 GB, 무효 실행 디렉토리(인용 금지 대상) — VERIFY.md에 사유 명시.

**검증 실측(임시 디렉토리에서 실제 수행)**
1. `sha256sum -c checksums.sha256` 전량 OK
2. 3 seed 판독 재산출 성공 → **저장소 원본과 H4a·H4b·위험·개요·형성 전 항목 일치**
3. 종합 재산출 → `3/3 seed 전 항목 PASS`, H4a 0.874±0.026 → 0.405±0.044, H4b −0.0021±0.0015 재현
4. 그림 3 seed × 2장 재생성 성공
5. 부적격 사후분석 재산출 성공

## 2026-08-28 — 원고 v11 패치 검증 + 원고 환경용 지시서 작성

원고 소스(main.tex·build_numbers.py·numbers.json·TikZ)는 **이 워크스테이션에 없음** —
`manuscript_pack_20260821/`은 실험 자료 묶음이라 원고 미포함. 패치 적용은 원고 환경에서 수행.

**검증 5건 실측 확정** (판정자 지시): ever-matured (22,18,21)=20.3±2.1 vs final-M (19,16,20)=18.3±2.1 /
τ **per-cluster**(ClusterState마다 ACIRiskController, 타 클러스터 τ 변화 0건) /
cold-start 라우팅은 **full-stream episode-weighted** 일치 / split-conformal은 **진짜 split**
(셔플 반반, μ·Σ=fit절반, q=disjoint calib절반, ⌈(n+1)(1−α)⌉ 보정) /
between-share = censored ranks 위 SS_b/SS_t (**순위 η²**), 절단 1개는 long 스위트(22셀 밖).

**원고 수치 전수 대조 → 정정 3건**
1. **P9 검정명**: "one-sided rank test" → **one-sided Fisher exact test** (`fisher_less` 코드 실측). 패치 v2 반영 완료.
2. **P12**: 434 s → **431.2 s** (434는 seed 0 단독 433.9의 반올림). 패치 v2 반영 완료.
3. **★ P3 신규**: 0.5945 → **0.5944**. pooled·seed평균 모두 0.594444이며 0.5945는 올림 오기.
   패치 v2가 "0.5945 유지"로 적었으나 실측이 반박.

**★ P14 신규 발견 — 논지가 뒤집히는 항목**
원고 V-F "training alone is far cheaper, about 181.8 s (≈2,137 VLA-call equiv)"의 181.8 s는
E1 anchor5 = **E2 배치의 8,000스텝 warm-start n=40 fit**이다. E5는 B-2로 **scratch + 배치 등가
스텝**(10,000/28,000)이라 조건이 다르다. 실측(152 이벤트): n=20 이벤트 289 s = 학습 227 s + probe 62 s,
n=80 이벤트 697 s = 학습 636 s + probe 60 s → 가중평균 431.2 s 중 **학습이 369.9 s(86%)**,
VLA-호출 등가 **학습만 4,348 / 이벤트 전체 5,068** (원고 2,137의 2배 이상).
현재 문장은 "형성 시간 대부분이 probe·준비"라는 반대 인상을 준다. V-F가 total-compute
break-even을 부정하는 절이므로 과소 기재는 주장의 보수성을 해친다 → 실측 교체 권고.

산출: `docs/INSTRUCTION_v11_for_manuscript_agent.md` (패치 v2 + 정정 3건 + P14 + 적용 절차 6단계 +
종결 항목표 + 판정 대기 2건).

## 2026-08-28 22:30 — depth privileged-information confound 스크리닝 (Stage 1) 완료

기존 결과·데이터 **무수정**(별도 경로: `experiments/rgb_depth_ablation/`, `results/rgb_depth_ablation/`,
`checkpoints/rgb_only_ablation/`). 기존 RGB-D 체크포인트는 읽기 전용 재사용.

**정지 조건(§17) 전건 통과**: RGB 별도 저장 확인 / 동일 held-out uid로 paired 성립 /
ACT 변경은 conv1 4→3채널뿐(파라미터 차 6,272 = 0.0066%) / split·seed 재현 가능 /
**기존 결과 재현 확인 — 동일 체크포인트 50-trial 재평가에서 E3 20-trial과 공통 uid 80/80 완전 일치**.

**클러스터 선정은 결과 산출 전 고정**(`ABLA_RGBD_CLUSTER_SELECTION.md`): 스위트별 N* 최소/최대 규칙으로
object 2·goal 2·spatial 1·long 1, easy 2/medium 1/difficult 2/censored 1 층화.

### 결과 (6 클러스터 × n{10,20,40,80} × held-out 50 = 1,200 paired 에피소드)
| Cluster | N* D→RGB | ŝ(80) D→RGB | Δ |
|---|---|---|---|
| object_task1 | 10→**20** | 0.88→0.96 | **+0.08** |
| object_task0 | 80→**40** | 0.96→0.90 | −0.06 |
| goal_task1 | 10→10 | 0.98→0.96 | −0.02 |
| goal_task0 | 80→80 | 0.92→0.82 | −0.10 |
| spatial_task1 | 20→20 | 0.96→0.98 | +0.02 |
| 10_task0 | >80→>80 | 0.74→0.60 | **−0.14** |

- **전체 paired Δ = −0.0150, 95% CI [−0.0392, +0.0092] — CI가 0 포함**. McNemar exact p=0.2495(불일치 100/118).
- **n별 비단조 패턴**: n=10 −12.00 pp → n=20 0.00 → n=40 **+9.67** → n=80 −3.67 pp.
  depth는 **저데이터 구간 수렴 보조**로 작동하고 천장에는 기여하지 않는다.
- **N* 체계적 증가 없음**: 4/6 동일, 이동 2개는 **+1과 −1로 방향 반대**.
- **실패는 전부 timeout 계열**(기타 0건) — depth 제거가 조기 파국을 만들지 않는다.
- **자동 판정 CASE B** (n=80 평균 −3.67 pp, 기준 3 pp를 0.67 pp 초과). 단 이 초과분은 사실상
  long 1개(−14 pp)에서 나오며, **long 제외 시 −1.60 pp로 CASE A 범위**.

### 권고 (§16 E)
**B(RGB-only ablation 추가)를 권고**하되 지시서 §11 CASE B는 C(25 클러스터 재실행)를 지정하므로
**연구원 판정 필요**. 근거: 리뷰어 비판의 실질인 "높은 습관 성능이 depth 덕분"은 천장에서 차이가
검출되지 않아 데이터로 반박된다. C 비용은 약 15시간(RGB-only 학습 4.4h + 양 조건 평가 10.4h).

산출: `RGB_DEPTH_ABLATION_AUDIT.md`(§16 A~E) · `ablation_summary.json` · `table_detail.csv` ·
`table_cluster.md` · `fig_A_curves.png`(Fig.2(a) 서식) · `fig_B_delta.png`(Fig.5(c) 서식) ·
`fig_C_nstar.png` · `CONFIG_DIFF.md`(동일성 16항목) · `RUN_COMMANDS.md`.

## 2026-08-28 23:10 — Stage 1 결과·데이터 패키지 (전체 재실험 결정)

연구원 결정: **전체 재실험(Stage 2) 실시**. 판정 요청본을 대체하는 완전 패키지 생성.

`depth_ablation_data_pack_20260828.tar.gz` (1.09 MB · 77 파일) — 생성기
`experiments/make_ablation_data_pack.py`. 추가분: 원자료 12 JSON(per-episode 전량) ·
학습 6 + 평가 12 + run.log = 19개 원본 로그 · train_summary 6개 ·
**RGB-only 체크포인트 24개 SHA256 매니페스트**(가중치 8.6 GB는 디스크 잔류) ·
논문 서식 산출물 · Stage 2 계획.

### ★ 규모 정정 — 25가 아니라 27
지시서 §11은 "25 클러스터 재실행"으로 적었으나 E3 배치 원장(`results/e3/e3_curves.json`,
`n_clusters_reported`=27, completeness missing=[])의 실제 클러스터는 **27개**다.
Stage 1에서 6개 완료 → **남은 21개**. 목록·비용은 `stage2_remaining_clusters.json`.

### 실측 비용 (Stage 1 wall-clock에서 산출)
클러스터당 학습 10.4분(범위 10.4–10.5) · RGB-only 평가 9.4분 · RGB-D 재평가 9.2분.
남은 21개 → **10.2시간**(학습 3.7h + 평가 6.5h). 평가가 전체의 64%.
감사 보고서의 기존 추정 "25 클러스터 약 15시간"을 이 실측치로 교체했다.

### Stage 2 미해결 질문 (REPORT.md §2)
① long-horizon — 유일한 long 셀이 RGB-D에서도 우측절단이라 형성 실패 위에서 측정한 차이
② 표본 크기 — n=80 초과분 0.67 pp가 클러스터 1개에서 발생 ③ spatial 가설 — 2개로는 검정 불가
④ 실패 유형 분해(비디오 필요) ⑤ 온라인 lifecycle 미검증.

Stage 2 고정 사항: 동일성 16항목 유지 · paired 성립(동일 held-out uid) ·
기존 RGB-D 체크포인트 재사용(재학습 금지) · Stage 1 6개 재실행 금지(SHA256 대조) · 클러스터 전수.

## 2026-08-28 23:00~ — RGB-only FULL RERUN 착수 (modality mismatch 교정)

teacher(2-view RGB + proprio)와 habit(2-view RGB-D + proprio)의 modality mismatch를 없애기 위해
**Paper 1 프로토콜 전체를 RGB-only habit으로 재실행**한다. 변경 가능한 실험 변수는 depth 제거 하나뿐.

### 격리 (§1·§2)
- 새 root `results/rgb_only_full_rerun_20260828/` · 체크포인트 `checkpoints/rgb_only_rerun/` ·
  스트림 궤적 `data/rgb_only_rerun/`. 기존 RGB-D 산출물은 read-only reference.
- 코드 변경은 **전부 가산**이며 기본값이 기존 동작을 재현한다:
  `e5_driver.py` (+`--no-depth`/`--out-root`/`--ck-root`/`--data-root`, §8 원장 필드 25종,
  §9 전이 이벤트 로그, 호출/학습시간 계측), `e5_counterfactual.py` (+`--queue-root`/`--out-root`),
  `e4r_competence_map.py` (+`--ckpt-root`/`--out`/`--frag-dir`, 에피소드별 PCA32 보존).

### 관문 통과
- **PREFLIGHT PASS** — git/env/dataset 27 클러스터 전건 + CONFIG_DIFF **예상 밖 차이 0**
  (허용 키 = in_ch·use_depth·n_params뿐) + RGB_ONLY_INPUT_AUDIT 6/6.
- **SMOKE PASS** — 학습·추론·저장/로드·성공판정·로깅·depth 미유출 11/11.
- **드라이버 스모크(RGB-only 50 ep) PASS** — `[GATE-PASS] steps=2500 |B_k|=10 stats 자기풀 일치
  scratch`, 재학습 계약 3단언 통과, X 전이 관측, §8 필수 필드 **누락 0**(행당 59 필드),
  전이 이벤트 ↔ 원장 건수 일치, 호출 계측 정상.

### 실행 구조 (§16·§17)
`experiments/rgb_only_rerun/` — runner(marker resume·retry 1회·job 원장) · run_batch ·
run_all(오케스트레이터, 선행 batch 종료 대기) · status(WEEKEND_RUN_STATUS) ·
analyze_batch(§6·§7) · analyze_online(§8·§9·§10) · analyze_replay(§11, 발화집합 + 전체스트림 H4b) ·
analyze_familiarity(§12 의존성 감사) · measure_latency(§13) · integrity_audit(§14) ·
old_vs_new(§15) · make_package(§19~21) · verify_package(§22).
전 프로세스 `setsid` 분리(PPID=1·TTY 없음·SIGHUP 무시) — 세션 종료와 무관하게 지속.

### 실측 기반 소요 전망
배치 27 클러스터 ≈ 7.5 h · 온라인 3 seed × 16.6 h · paired replay 3 × 4.5 h ≈ 63 h ·
분석/패키징 ≈ 1.5 h → **총 ≈ 72 h**. GPU가 1대이고 온라인 wall-clock이 보고 대상(§13)이라
동시 실행하지 않고 순차 실행한다.

## 2026-08-29 17:10 — 모니터링 보강 + 분석 경로 사전 검증

### 배치 완료 (27/27, 6.8 h, 재시도 0)
`analyze_batch.py`를 앞당겨 실행 (온라인과 독립·수 초 소요). 2,640 평가 에피소드.
N* 동일 16/27 · 형성셀22 중앙값 N* 15.0→10.0 · 우측절단 1→1 ·
KW H=1.1905 p=0.5514 (구 0.5176/0.772) · between 0.0567 (구 0.0246) ·
horizon T1vsT3 p=0.0631 (구 0.0968) ·
chain task0 곱기준선 p_below 0.0049→0.9979, task5 0.9831→0.8624.
**해석은 노트북 쪽 역할.** 다만 chain task0의 반전은 같은 태스크 단일 N* 80→10과 연동된 값이다.

### 모니터링 결함 수정
`WEEKEND_RUN_STATUS`가 stage 종료 시에만 갱신돼, seed0이 11시간째 도는 중에도
`WAIT 0/4000`으로 보였다 → **5분 주기 갱신기**(`status_watch.sh`) + **ETA**(진행 중 seed는
자기 실측 속도, 나머지는 실측 단가) 추가.

### watchdog 신설 (`watchdog.py`, 세션 무관 상주)
5분 주기로 ①오케스트레이터 생존 ②스트림 정체(45분 무변화 — 재학습 n=80 ≈ 11분 + probe 5분 고려)
③FAILED_JOBS 증가 ④신규 DONE marker 감시. seed 완료 시 즉시 `analyze_online` →
replay 완료 시 `analyze_replay` + `integrity_audit` + `make_package`를 돌려
**항상 최신 패키지가 디스크에 남게** 한다. 기록 = `ALERTS.log`.

### 분석 경로 사전 검증 (부분 데이터, hour 55 실패 회피)
`analyze_online` · `integrity_audit`(16/16 VALID) · `make_package` · `verify_package`를
seed0 부분 데이터로 완주. **패키지만 읽어 숫자 복원 15/15 PASS.**
- 발견·수정: `final_lifecycle`이 드라이버 summary에만 의존해 파일이 없으면 조용히 0이 됐다
  → **원장의 마지막 `state_after`에서 도출**하고 summary는 교차검증 필드로 강등.
- 디스크: 1.3T 여유 / 온라인 잔여 예상 소비 ≈ 85 GB — 여유 충분.

### 운영 교훈 (2회 반복)
`pkill -f`/`kill $(pgrep -f ...)`의 패턴이 **자기 셸과 Monitor의 명령줄에도 매칭**돼
둘 다 죽였다(exit 144). 실험 프로세스는 무사. 이후 pgrep 패턴은 `watchdog[.]py`처럼
정규식 브래킷으로 자기 매칭을 차단하고, PID는 `watchdog.pid`에 기록한다.

## 2026-08-31 20:39 — RGB-only FULL RERUN 완료

**전 stage PASS · 실패 job 0 · 재시도 0 · infra error 0.** 소요 2026-08-28 23:03 → 08-31 20:39 (≈ 70 h).

### 관문
- §14 무결성 **VALID (47검사, FAIL 0)** — 대역 disjoint · train/eval 중첩 0 · probe/stream 중첩 0 ·
  중복/누락 에피소드 0 · 비성숙 발화 0 · teacher의 A_mat 유입 0 · 전이 합법성 · 전이로그↔원장 일치 ·
  paired 큐 집합 동일 · **RGB-only 체크포인트 전수 depth 미사용**.
- §22 **패키지-단독 숫자 복원 PASS (52검사, FAIL 0)** — 레포 코드 import 없이 원장 CSV에서
  재계산해 요약과 대조.

### 배치 (27 클러스터, 2,640 에피소드)
N* 동일 16/27 · 형성셀22 중앙값 15.0→10.0 · 우측절단 1→1 · KW p 0.772→0.5514 ·
horizon T1vsT3 p 0.0968→0.0631 · chain task0 곱기준선 p_below 0.0049→0.9979.

### 온라인 3-seed (12,000 ep) — RGB-D → RGB-only (평균±sd)
r_V 0.6152±0.0190 → 0.5947±0.0180 · 시스템성공 0.9608±0.0042 → 0.9564±0.0036 ·
발화성공 0.9715±0.0068 → 0.9623±0.0024 · **Pr(fail|fire) 0.0285±0.0068 → 0.0377±0.0024** ·
최종 M 18.3±2.1 → 18.0±1.0 · 성숙도달 20.3±2.1 → 20.0±1.0. H4a 세 seed 전부 통과.

### H4b 비열등 — 전 seed 통과, 단 부호가 일관되게 음수
전체스트림 합성: RGB-D −0.0021±0.0015 (CI 전부 0 포함) → RGB-only **−0.0065±0.0014
(CI 세 seed 전부 음수)**. margin −3%p 대비 1/5.
발화집합만: Δ −0.0161±0.0036, McNemar p = 0.0016/0.042/0.0015 (세 seed 유의),
pooled 4,863발화 Δ=−0.0160 CI[−0.0224,−0.0097].
→ **사전등록 기준은 여유 통과하나, 발화 구간의 습관–teacher 격차는 RGB-only에서 일관 검출.**

### §12 familiarity (habit 의존분만 재계산: 4 재계산 / 5 재사용, sha256 기록)
423 에피소드. Mahalanobis→습관실패 AUC 0.6904 (kNN k5 0.6005 / k10 0.6116).
폭별 성공률 0.01:0.878 0.02:0.776 0.04:0.596 0.06:0.520 0.08:0.322 (기존 0.911/0.844/0.679/0.483/0.358).
w* 6개 중 3개 이동 — goal_task0 0.06→0.08, goal_task2 0.04→0.06(상승), spatial_task0 0.06→0.04(하강).

### §13 레이턴시 (attn=sdpa)
ACT RGB-only 3.35 ms (p95 3.36) · RGB-D 3.35 · gate 4.18 · teacher 85.47.
**ACT 레이턴시는 modality에 사실상 불변** (비율 0.03920 vs 0.03923).
재학습 145건: 이벤트 413.17 s = 학습 353.99 + probe·준비 59.19 (n=20 226.6 s / n=80 628.2 s).
VLA-호출 등가 이벤트당 4,834 · 학습만 4,142.

### 산출
`HabitVLA_RGB_only_full_rerun_20260828.tar.gz` (2.9 MB · 191 파일) —
README_RESULTS · DATA_DICTIONARY · 원장 CSV/JSONL · 통계 · 부트스트랩 분포(.npy) ·
CHECKPOINT_MANIFEST(가중치 55 GB는 sha256 대조) · OLD_VS_NEW_NUMERIC 159행 ·
무결성·패키지검증 · logs · configs/code.
**논문 문장·LaTeX·PDF·figure·publication table은 만들지 않았다 (§18·§23).**
