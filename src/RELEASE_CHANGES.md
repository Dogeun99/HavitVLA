# RELEASE_CHANGES — 원본 저장소 대비 변경 사항

- 원본: `/home/asmr/workspace/habitvla2` (git `master`, 최종 커밋 `5c12f9b` "Finish the RGB-only full rerun with all stages verified", 2026-08-31)
- 릴리스 생성: 2026-09-03
- 원칙: **실험 코드 경로는 손대지 않는다.** 변경은 (a) 워크스테이션 절대 경로의 이식성, (b) 빌드/검증 보조 파일 추가뿐이다.
  `habits/`, `envs/*.py`, `gates/`, `teacher/`, `configs/`, `docs/`는 원본과 바이트 단위로 동일하다
  (`diff -rq`로 확인, 2026-09-03).

## 1. 경로 이식성 (동작 동일, 위치만 스크립트 기준으로)

| 대상 | 변경 전 | 변경 후 |
|---|---|---|
| `experiments/{e0_ckpt_load,e1_sv_collect,e3_collect,e1_latency,e3_t2_validate,e0_walltime_collect,e0_smoke_collect,e2_collect}.py` | `HABIT2 = "/home/asmr/workspace/habitvla2"` | `HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))` |
| `experiments/e0_download_ckpts.py` | HF_HOME 접두사 단언·출력 경로가 절대 경로 | `HABIT2` 기준 |
| `experiments/rgb_only_rerun/runner.py` | `PY_HAB/PY_OFT`가 conda env 절대 경로 | 환경변수 `HV2_HAB_PY` / `HV2_OFT_PY` 우선, 기본값 `~/miniconda3/envs/hv2_{hab,oft}/bin/python` |
| `experiments/e5_driver.py` | 재학습 서브프로세스가 hv2_hab python 절대 경로 호출 | 모듈 상수 `PY_HAB` (위와 같은 규칙) |
| 셸 스크립트 17개 (`experiments/*.sh`, `experiments/*/*.sh`, `tools/*.sh`) | `HABIT2=/home/asmr/...`, `cd /home/asmr/...`, python/tmux 절대 경로 | `HABIT2=${HABIT2:-<스크립트 기준 루트>}`, `${HV2_OFT_PY:-…}`, `${HV2_HAB_PY:-…}`, `${HV2_TMUX:-…}` |
| `.py` docstring의 실행 예시 | `/home/asmr/miniconda3/envs/hv2_hab/bin/python …` | `$HV2_HAB_PY …` (주석 전용, 코드 영향 없음) |

## 2. `envs/setup_envs.sh` (재작성)

- env 이름·복제 원본을 환경변수로: `OFT_ENV`, `HAB_ENV`, `CLONE_OFT_FROM`(기본 `vla_oft`), `CLONE_HAB_FROM`, `SKIP_TORCH_INSTALL`.
- third_party가 git submodule로 비어 있으면 `submodule update --init`, 없으면 clone. LIBERO·openvla-oft 둘 다 핀 커밋 확인 + 패치 **멱등** 적용(이미 적용돼 있으면 skip).
- 검증 단계에서 LIBERO가 namespace 패키지(`__init__.py` 없음, 원본 ISSUE-10)라 `libero.__file__`이 `None`인 점을 반영해 `libero.__path__[0]`으로 설치 위치를 확인한다.
- `scipy`, `joblib`를 hab env 설치 목록에 명시(원본은 scikit-learn 의존으로 딸려 왔음).

## 3. 추가된 파일

| 경로 | 내용 |
|---|---|
| `build.sh` | 원클릭 빌드 (setup_envs.sh + 선택적 자산 링크) |
| `setup/hv2_hab.requirements.lock`, `setup/hv2_oft.requirements.lock` | 검증된 워크스테이션 env의 `pip freeze` 전체 |
| `setup/hv2_hab.environment.yml`, `setup/hv2_oft.environment.yml` | `conda env export --no-builds` |
| `setup/verified_hardware.txt` | GPU/드라이버 |
| `setup/link_local_assets.sh` | 체크포인트·HDF5·모델 캐시를 원본 디렉터리에서 하위 디렉터리 단위로 심볼릭 링크 |
| `verify/verify_release.sh` | 릴리스 검증 오케스트레이터 (scratch 복제 → 재산출 → 대조 → GPU 롤아웃) |
| `verify/compare_outputs.py` | JSON/CSV/NPY 대조 + evaluate.py 에피소드별 롤아웃 대조 |
| `results` → `../results` | 심볼릭 링크. 모든 스크립트가 `<HABIT2>/results/…`를 읽고 쓰므로 결과 폴더와 연결 |
| `README.md`, `RELEASE_CHANGES.md` | 본 문서들 |

## 4. 제거·제외

- `MUJOCO_LOG.TXT` (추적돼 있던 임시 로그) 제거.
- 원본에서 gitignore였던 것들은 그대로 제외: `checkpoints/`(192 GB), `data/`(15 GB), `.hf_cache/`(60 GB), `logs/`, 판정용 `*_pack` 디렉터리·tar.gz.
  판정 패키지의 **보고서 문서**는 `../results/reports/`에 모았다.
- `third_party/`는 git submodule(LIBERO `8f1084e`, openvla-oft `e4287e9`)로 등록하고, 로컬 패치는 `configs/*.patch`를 빌드 시 적용한다.
