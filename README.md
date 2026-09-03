# HabitVLA-2 — Amortized Inference via Habit Formation (release)

LIBERO 시뮬레이션에서 VLA(OpenVLA-OFT) 교사의 반복 경험으로 클러스터별 경량 습관 정책(ACT)이 형성되고,
관할·성숙도 2단 gate가 위험 보장 하에 VLA 호출을 선택적으로 생략함을 보인 연구의 **실제 동작 소스와 결과**를
두 폴더로 나눠 정리한 것이다 (원본 작업 저장소 커밋 `5c12f9b`, 2026-08-31 → 릴리스 2026-09-03).

| 폴더 | 내용 | 시작점 |
|---|---|---|
| [`src/`](src/) | 빌드해서 바로 쓰는 메인 소스 (envs/teacher/habits/gates/experiments/tools/configs/docs + third_party 서브모듈 + 빌드·검증 스크립트) | [`src/README.md`](src/README.md) · [`src/RELEASE_CHANGES.md`](src/RELEASE_CHANGES.md) |
| [`results/`](results/) | 본 실험 E0–E5 결과 JSON/CSV/그림, depth ablation, RGB-only 전체 재실행 데이터 패키지, 판정 보고서, git 이력 | [`results/README.md`](results/README.md) · [`results/VERIFICATION_REPORT.md`](results/VERIFICATION_REPORT.md) |

```bash
git clone --recurse-submodules <repo> && cd habitvla2_release/src
bash build.sh                      # conda env 2개(hv2_oft, hv2_hab) + LIBERO/OFT 핀 체크아웃·패치 → [BUILD-DONE]
bash verify/verify_release.sh --no-gpu   # 저장 결과를 원자료에서 재산출해 대조 (체크포인트 없이 21단계)
```
**clone 직후 위 두 줄만으로 빌드된다.** 저장소에는 절대 경로 심볼릭 링크가 없고(`src/results → ../results`는
저장소 내부를 가리키는 상대 링크), third_party 로컬 패치는 빌드 스크립트가 적용한다. 2026-09-03에 별도
디렉터리로 clone → 서브모듈 init → `build.sh` → 검증 21/21 PASS를 실측했다.

대용량 자산(체크포인트 192 GB, 궤적 HDF5 15 GB, 모델 캐시 60 GB)은 저장소에 **없다**. 없어도 빌드와
결과 재산출·검증은 되고, GPU 롤아웃 재평가·무결성 감사만 자산이 필요하다 — `src/README.md` §3·§4.

## 검증 요약 (2026-09-03)

상세는 [`results/VERIFICATION_REPORT.md`](results/VERIFICATION_REPORT.md).

**(a) 자산이 있는 환경** — 이 폴더만으로 빌드한 별도 env `hv2r_oft`/`hv2r_hab`, **27단계 전부 PASS**
- 저장된 원자료에서 요약·통계 36개 파일 재산출: 33개 바이트 동일, 3개는 출처 절대경로 등 휘발 필드만 상이(수치 동일).
- GPU: 학습·추론 스모크 11/11 · 체크포인트 held-out 재평가(RGB-only goal_task1 n=10/80, RGB-D object_task1 n=80)
  60 에피소드 전부 성공/실패·스텝 수까지 저장본과 일치 · 무결성 감사 VALID 47/47(체크포인트 253개) ·
  레이턴시 재측정 동일 수준(ACT 3.4 ms, teacher 85.4 ms).

**(b) 저장소만 clone한 환경** — 별도 디렉터리로 clone → 서브모듈 init → `build.sh`, **검증 21단계 PASS**
- clone 트리의 심볼릭 링크는 `src/results → ../results` 하나뿐이고 저장소 내부를 가리킨다. 깨진 링크 0개.
- pristine 서브모듈에 로컬 패치 2개가 그대로 적용되고 `[BUILD-DONE]`까지 완주.
- 체크포인트 없이도 저장 결과의 재산출·대조가 전부 수행된다. 자산이 필요한 GPU 단계만 자동으로 건너뛴다.

## GitHub 업로드

```bash
cd habitvla2_release
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin master
```
third_party는 서브모듈(LIBERO `8f1084e`, openvla-oft `e4287e9`, 각자 MIT 라이선스)로 업스트림 URL을 가리키며,
로컬 패치 2개는 `src/configs/*.patch`에 있고 `build.sh`가 적용한다.
