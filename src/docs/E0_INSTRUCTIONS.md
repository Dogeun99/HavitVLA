# E0 작업 지시서 — 환경 구축·검증 (Claude Code 실행용)

- 근거: 설계서 §5 E0 행 + §9 인프라 원칙. Go 기준의 원본은 `configs/preregistration.md`.
- 원칙: **문제 축소 금지.** 항목이 막히면 우회·placeholder 대신 **왜 막혔는지**를 `log.md`에 기록하고 정지.
- 모든 항목은 (a) 실행 명령 (b) 합격 기준 (c) 산출 파일 이 세 가지를 갖는다.
- 산출물은 `results/e0/<항목>.json`. **stdout 마지막 줄에 반드시 PASS 마커**를 찍는다(§9 readiness 규칙).

```
[E0-PASS] item=<E0-x> status=<PASS|FAIL> json=results/e0/<file>.json
```

`readiness = 프로세스 생존 + 위 마커 존재`. **로그 "error" grep 금지**(TF 배너 오탐).

---

## 공통 환경 변수 (모든 세션에서 먼저 export)

```bash
export HABIT2=~/workspace/habitvla2
export HF_HOME=$HABIT2/.hf_cache          # ★ 전역 캐시 오염 방지 (§9)
export MUJOCO_GL=egl                      # headless 렌더
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
```

tmux 세션명 `habit2`, 장기 실행은 `python -u`.

---

## E0-1 — conda env 2개 신규 구축

**[연구원 확정 대기: `log.md` ISSUE-2]** teacher rollout이 시뮬레이터를 동일 프로세스에서 구동하므로
**두 env 모두 LIBERO가 필요**하다. 제안 구성:

| env | 역할 | py | 핵심 핀 |
|---|---|---|---|
| `hv2_oft` | LIBERO sim + OFT teacher (E1 S_V, E2/E3 궤적 수집, E5 fallback) | 3.10 | torch 2.7.0+cu128, transformers = moojink 포크(4.40.1), timm 0.9.10, tokenizers 0.19.1, peft 0.11.1, robosuite 1.4.1, numpy 1.26.4 |
| `hv2_hab` | LIBERO sim + ACT 학습/평가 + gates (DINOv2, conformal) | 3.11 | torch 2.7.0+cu128, transformers 최신, scikit-learn, robosuite 1.4.1, numpy 1.26.4 |

**기존 env(`habitvla`, `phase1`, `vla`, `vla_oft`) 불침범.** `base`에 설치 금지.

```bash
conda create -n hv2_oft python=3.10 -y
conda create -n hv2_hab python=3.11 -y
# torch는 두 env 모두 cu128 인덱스에서 (sm_120)
conda run -n hv2_oft pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

`hv2_oft`의 OFT 설치는 **프로젝트 사본 `$HABIT2/third_party/openvla-oft`에서** 수행한다
(구 프로젝트 경로 참조 금지 — ISSUE-14). sm_120 cu128 재핀은 `configs/openvla_oft_local.patch`로
보존되며, 사본이 없는 머신에서는 `envs/setup_envs.sh`가 upstream `e4287e9` 체크아웃 + 패치 적용으로
재구축한다.

- **합격**: 두 env에서 `torch.cuda.is_available() == True`, `torch.cuda.get_device_capability() == (12, 0)`,
  간단한 matmul이 GPU에서 성공.
- **산출**: `results/e0/e0_1_envs.json` (env별 python/torch/transformers/robosuite/mujoco/numpy 버전 + capability)
- **주의**: flash-attn은 **빌드하지 않는다** → `attn_implementation="sdpa"` 고정, JSON에 `attn=sdpa` 기록
  (`log.md` ISSUE-6).

## E0-2 — LIBERO 설치 (두 env 동일 커밋)

**재구축의 정본은 `envs/setup_envs.sh`** — 핀 체크아웃(`8f1084e3`), 로컬 패치 자동 적용
(`configs/libero_local.patch`, `configs/openvla_oft_local.patch`), editable **compat 모드**,
mujoco==3.1.6 / numpy==1.26.4 / opencv==4.9.0.80 재핀, `.libero/config.yaml` 생성(절대경로
머신 종속 — 스크립트가 `$HABIT2` 기준으로 재생성)까지 전부 포함한다. 아래는 개요만:

```bash
bash $HABIT2/envs/setup_envs.sh   # idempotent — 이미 구축된 env는 건너뜀
```

- **합격**: 두 env에서 `from libero.libero import benchmark; benchmark.get_benchmark_dict()`가
  4 스위트(`libero_spatial/object/goal/10`)를 반환하고, 각 스위트 태스크 수 = 10.
- **산출**: `results/e0/e0_2_libero.json` (LIBERO 커밋 해시, 스위트별 태스크 수·task id·지시어 전문,
  env별 robosuite/mujoco/bddl 버전)
- **부가 산출(중요)**: 태스크 지시어 전문은 §2.4 1층(지시어 정확 일치 클러스터링)의 원본이므로
  `configs/task_registry.json`으로도 저장한다.
- **리스크**: robosuite 1.4.1 ↔ mujoco 버전 조합 (`log.md` ISSUE-3). 실패 시 조합을 실측 탐색하고
  **성공 조합을 JSON에 핀으로 기록**.

## E0-3 — depth 노출 확인 ★ go 기준

`OffScreenRenderEnv`에 `camera_depths=True`를 추가로 전달해 obs 키를 확인한다
(참고: `openvla-oft/experiments/robot/libero/libero_utils.py:18-26`).

- **합격 기준(설계서 §5)**: depth 관측의 `(H, W)`가 RGB와 동일하고(기본 256×256),
  dtype·값 범위가 유효하며(전부 0/NaN 아님), **에피소드 진행에 따라 값이 변한다**.
- 추가 확인: 값이 robosuite의 **정규화된 depth**인지 미터 단위인지 판별해 기록
  (ACT 입력 전처리에 직결).
- **산출**: `results/e0/e0_3_depth.json` (키 목록, shape, dtype, min/max/mean, 정규화 여부 판정 근거)
- **막히면**: LIBERO 래퍼가 키를 필터링하는 경우 → 래퍼를 우회하지 말고 `envs/`에
  **명시적 서브클래스**를 만들어 노출한다(문제 축소 금지: depth 포기 금지, 왜 필요한지 §2.1 근거).

## E0-4 — OFT 4 체크포인트 로드

```bash
# HF_HOME 은 반드시 프로젝트 로컬
moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}
```

- **[연구원 확정 대기: ISSUE-8]** 전역 캐시의 spatial 체크포인트를 **읽기 전용 symlink**로 재사용할지,
  4종 전체(≈ 56GB)를 신규 다운로드할지. 미승인 시 기본값 = **신규 다운로드**(격리 우선).
- **합격**: 4종 모두 로드 성공 + 더미 관측 1스텝 forward가 유한한 action chunk 반환
  (2-image + proprio, L1 regression head, greedy).
- **산출**: `results/e0/e0_4_ckpt.json` (체크포인트별 로드 시간, VRAM 피크, dtype, action chunk shape,
  `attn=sdpa`)
- **주의**: VRAM 32GB에 7B bf16 1개는 여유. **동시 로드는 하지 않는다**(스위트별 순차 실행).

## E0-5 — 스위트별 10-ep 스모크 ★ go 기준

스위트당 태스크 1개 이상을 포함해 **10 에피소드**를 결정적으로 rollout(greedy, seed 고정).

- **합격 기준(설계서 §5)**: 성공률이 **공개 보고치 ±10 %p 이내**
  (4 스위트 평균 공개치 97.1 %, 출처 `openvla-oft/LIBERO.md:41` — 스위트별 개별 수치는 E0에서
  논문 표를 확인해 JSON에 함께 기록).
- **[연구원 확정 대기: ISSUE-7]** 10 ep는 해상도가 낮다(참값 0.97에서도 한 스위트가 8/10을 낼 확률 ≈ 3.5 %,
  4 스위트 중 최소 1회 오탐 ≈ 13 %). **8/10 발생 시 해당 스위트만 20 ep 재확인 후 판정**하는 규칙을
  제안하며, 채택 시 `configs/preregistration.md` §5 변경 이력에 기록 후 적용한다.
- **최상위 리스크(ISSUE-1)**: OFT 공식 문서가 *학습 GPU와 다른 디바이스에서 성능이 크게 떨어질 수 있음*을
  경고한다. 스모크 미달 시 원인 분리 순서 = ① dtype(bf16/fp32) ② attention 구현 ③ 이미지 전처리
  (180° 회전 등 `get_libero_image`) ④ LoRA 재병합 ⑤ 디바이스 자체 효과.
  **각 단계의 결과를 JSON과 `log.md`에 남긴다.**
- **산출**: `results/e0/e0_5_smoke.json` (스위트별 성공/실패 에피소드 리스트, task id, seed,
  init state index, 스텝 수, 공개치와의 차이)

## E0-6 — 초기상태 변이 폭 제어 확인 ★ go 기준

설계서 §2.1이 요구하는 **"변이 폭의 파라미터화 가능 여부"** 확인. LIBERO가 기본 제공하는 것은
`task_suite.get_task_init_states(task_id)`의 **고정 배열**뿐이다(`run_libero_eval.py:230`).

검증할 두 경로:
1. **섭동 경로** — 고정 init state 벡터의 물체 pose 성분에 통제된 노이즈(스케일 파라미터 `w`)를 가하고
   `env.set_init_state()` 후 시뮬레이터가 안정적으로 정착하는지(물체 관통·낙하 없음) 확인.
2. **재샘플링 경로** — BDDL init 영역에서 robosuite placement initializer로 재샘플링(정당한 경로).

- **합격 기준**: 변이 폭 파라미터 `w`를 바꿨을 때 (a) 초기 물체 위치 분산이 **단조 증가**하고
  (b) 물리적으로 유효한 초기상태 비율이 임계 이상이며 (c) **동일 seed·동일 w에서 완전 재현**된다.
- **산출**: `results/e0/e0_6_variation.json` (경로별 성공 여부, w 격자별 위치 분산·유효율·재현성 해시)
- **왜 중요한가**: 성숙 곡선의 "클러스터 내 변이" 통제와 E4 관할 파일럿의 novel 생성
  (변이 폭 확대)이 모두 이 기능에 의존한다. **여기서 막히면 E4 설계가 바뀌므로 즉시 보고**.
- **주의**: `libero_utils.get_libero_env()`의 `env.seed(0)` 주석 — *"seed seems to affect object positions
  even when using fixed initial state"*. 결정성 확보를 위해 seed 주입 경로를 명시적으로 통제하고,
  동일 조건 2회 실행의 관측 해시 일치를 확인한다.

## E0-7 — wall-clock 실측 → 예산 정밀화

- 측정: 스위트별 **에피소드당 초**(teacher rollout, 성공/실패 구분), sim step 시간과 모델 forward 시간 분해,
  env reset 시간, 프로세스 기동 오버헤드.
- 산출: `results/e0/e0_7_walltime.json` + **예산 재추정표**
  (E1 1,000 ep / E2 클러스터당 120+50 / E3 28 클러스터 / E5 2,000 ep × 3 seed을 실측 초로 환산).
- 설계서의 200–350 GPU-h 추정을 실측으로 대체하고, 초과 시 §5 R4 원칙(**셀을 줄이되 셀당 통계 유지**)
  으로 조정안을 제시한다. 조정 실행은 연구원 결정.

---

## E0 종합 go/no-go

| 기준 | 값 | 근거 |
|---|---|---|
| 스모크 성공률 | 공개 보고치 **±10 %p 이내** (ISSUE-7 재확인 규칙 적용 시 그 규칙 포함) | 설계서 §5 |
| depth | `(H, W)` 정상 + 유효 값 범위 | 설계서 §5 |
| 초기상태 변이 폭 | 파라미터화 **가능** | 설계서 §5 |
| (추가) 결정성 | 동일 seed 2회 실행에서 궤적 해시 일치 | §2.2 결정적 rollout 전제 |

**모두 통과 시** → E1(S_V 재측정 1,000 ep + 레이턴시 앵커 5종) 착수.
**미달 시** → 해당 항목의 원인 분리 결과를 `log.md`에 남기고 **연구원 판정 대기**. 임의 우회 금지.
