# HabitVLA-2 — Amortized Inference via Habit Formation (release)

LIBERO 시뮬레이션에서 VLA(OpenVLA-OFT) 교사의 반복 경험으로 클러스터별 경량 습관 정책(ACT)이 형성되고,
관할·성숙도 2단 gate가 위험 보장 하에 VLA 호출을 선택적으로 생략함을 보인 연구의 **실제 동작 소스와 결과**를
두 폴더로 나눠 정리한 것이다 (원본 작업 저장소 커밋 `5c12f9b`, 2026-08-31 → 릴리스 2026-09-03).

**Author** DoGeun Lee ([@Dogeun99](https://github.com/Dogeun99)) · 연구 방향은 지도교수 지도 아래 진행됐다 ([`PROJECT_STORY.md`](PROJECT_STORY.md) §4).
*커밋 `21a45b9`~`3f28db8`의 작성자가 `gnukim`으로 기록된 것은 공용 워크스테이션의 git 전역 설정이 그대로 쓰였기 때문이며, 실제 작업자는 위 저자다.*

**In one paragraph.** A vision-language-action (VLA) robot policy is accurate but costs about 85 ms per
decision. This project tests whether a robot can *amortize* that inference: as successful VLA trajectories
accumulate for a recurring situation, a lightweight habit policy (ACT, ~3.4 ms) is trained for it, and a
two-stage gate — is this situation familiar, and has the habit proven itself — decides when the large model
can be skipped. On the LIBERO benchmark across 3 seeds x 4,000-episode online streams, the VLA call rate
fell from 0.874 to 0.405 while task success stayed within -0.0021 of always calling the VLA (pre-registered
margin -0.03), with habit failure probability 0.0285 against a 0.2 ceiling. One hypothesis was rejected by
the data and one gate component was recorded as unsolved; both are documented rather than hidden.
Full narrative (Korean): [`PROJECT_STORY.md`](PROJECT_STORY.md).

| 문서 | 내용 |
|---|---|
| **[`PROJECT_STORY.md`](PROJECT_STORY.md)** | **먼저 읽을 것.** 왜 시작했고, 어떻게 진행됐고, 무엇이 나왔는지의 전체 서사. 막힌 자리에서 내린 판단과 한계, 인용하면 안 되는 산출물까지 한 문서에 정리돼 있다 |
| [`src/README.md`](src/README.md) | 소스 구조, 빌드 절차, 재현 명령 |
| [`results/README.md`](results/README.md) | 결과 수치와 각 값의 출처 파일 |
| [`results/VERIFICATION_REPORT.md`](results/VERIFICATION_REPORT.md) | 이 저장소만으로 결과가 재현되는지 검증한 기록 |
| [`src/log.md`](src/log.md) | 날짜별 연구 일지 49개 항목 (원기록) |

| 폴더 | 내용 |
|---|---|
| [`src/`](src/) | 빌드해서 바로 쓰는 메인 소스 (envs/teacher/habits/gates/experiments/tools/configs/docs + third_party 서브모듈 + 빌드·검증 스크립트) |
| [`results/`](results/) | 본 실험 E0–E5 결과, depth ablation, RGB-only 전체 재실행 데이터 패키지, 판정 보고서 25개, 그림 |

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
git remote add origin https://github.com/Dogeun99/HavitVLA.git   # 이미 설정돼 있으면 생략
git push -u origin main
```
third_party는 서브모듈(LIBERO `8f1084e`, openvla-oft `e4287e9`, 각자 MIT 라이선스)로 업스트림 URL을 가리키며,
로컬 패치 2개는 `src/configs/*.patch`에 있고 `build.sh`가 적용한다. 받는 쪽은 `--recurse-submodules`로 clone하면 된다.
