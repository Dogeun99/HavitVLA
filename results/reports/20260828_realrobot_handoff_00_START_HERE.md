# HabitVLA-2 실물 로봇 실험 — 인수인계 패키지

생성 2026-08-28. 시뮬레이션(LIBERO) 구현·실행 일체와 이식 가이드를 포함한다.

## 읽는 순서

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `paper/` (PDF를 넣어 전달) | RA-L 투고본 — 주장과 future work |
| 2 | **`01_PORTING_GUIDE.md`** | **실물 이식 시 무엇이 깨지는지 — 먼저 읽을 것** |
| 3 | `02_sim_baseline.json` | 실물 결과와 대조할 시뮬 기준선·상수 |
| 4 | `code/` | 전체 소스 (저장소 구조 그대로) |
| 5 | `prereg/preregistration.md` | 동결 상수와 그 근거·변경 이력 |
| 6 | `prereg/log.md` | 실패 이력 — **통독 말고 검색해 쓸 것** |
| 7 | `results/`, `raw/`, `figures/` | 시뮬 결과 전량 |

## 시뮬 결과 요약 (3 seed, 12,000 ep)

- VLA 호출률 **0.874±0.0261 → 0.405±0.0442** (Δ 0.469±0.0699)
- 비열등 diff **-0.0021±0.0015** (margin -0.03, paired 4,618 ep)
- 발화 위험 Pr(fail|fire) **0.0285±0.0068** (ε=0.2)
- 성숙 20.3±2.1/33, 소요 노출 중앙값 22.7±0.6회

## ★ 착수 전 반드시 결정할 두 가지

1. **성공 판정을 무엇으로 할 것인가** — 시뮬의 무료·정확한 predicate가 실물엔 없다. 이 신호가 학습 데이터·인증·위험 통제 셋에 동시에 들어가므로, 판정기 오차를 먼저 측정해야 한다. (가이드 §1)
2. **H4b 검정을 어떻게 다시 설계할 것인가** — 실물은 초기상태를 재현할 수 없어 paired replay가 불가능하다. 논문의 "paired full-VLA replay under identical episode specifications" 문구는 실물에서 그대로 쓸 수 없다. (가이드 §2)

## 제외된 것과 복원 방법

| 항목 | 크기 | 복원 |
|---|---|---|
| `checkpoints/` | 93 GB | 재학습으로 재생성 (spec 결정적) |
| `data/` HDF5 | 8.8 GB | teacher 재수집으로 재생성 |
| `third_party/` (LIBERO, openvla-oft) | — | 공개 저장소에서 설치 |
| OpenVLA-OFT 가중치 | ~16 GB | `moojink/openvla-7b-oft-finetuned-libero-*` |

## 코드를 돌려보려면

```bash
cp -r code/* <새-저장소>/     # code/ 내용을 저장소 루트로 옮기면 경로가 맞는다
#   (스크립트가 파일 기준 상위를 프로젝트 루트로 잡으므로 code/ 하위에서는 동작하지 않는다)
git clone <LIBERO>  third_party/LIBERO
git clone <openvla-oft> third_party/openvla-oft
```

## 환경

conda env 2개: `hv2_oft`(OpenVLA-OFT 추론) · `hv2_hab`(ACT 학습·분석). `export HF_HOME=<repo>/.hf_cache`로 캐시를 격리한다(공용 캐시 오염 방지). flash-attn은 sm_120 미빌드이므로 **attn=sdpa**를 쓰며, 모든 지연 수치에 이를 명기해야 한다.
