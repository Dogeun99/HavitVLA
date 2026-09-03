# [지시] HabitVLA-2 v10 → v11 패치 적용 — 원고 작업 환경용

> 작성: 실험 환경(`habitvla2`) 2026-08-28. 원고 소스는 이 환경에 없으므로 **적용은 원고 환경에서** 수행한다.
> 동봉 패치: `HabitVLA2_v10_patch_20260828_2.md` (P1–P13 + §D–§F)
> 본 지시서는 그 패치에 **실험 원장 실측으로 확정한 근거·정정 3건**을 더한 최종본이다.

---

## 0. 이 지시서가 패치 v2에 더하는 것

패치 v2 이후 실험 원장을 추가 대조해 **수치 정정 2건과 신규 항목 1건**을 찾았다.
아래 §A는 패치 v2에 없거나 다르게 적힌 내용이므로, **패치보다 이 지시서를 우선**한다.

| 항목 | 패치 v2 | 실측 확정 | 조치 |
|---|---|---|---|
| P3 standard 라우팅 | 0.5945 유지 | **0.594444** | **0.5944로 정정** (신규) |
| P12 형성 이벤트당 | 431 s | 431.2 s | 유지 (반올림 431) |
| **P14 학습 비용** | 181.8 s / 2,137 calls | **E5 조건 아님** | **신규 — 아래 §A-3** |

---

## A. 실측 근거 (원고 환경에서 재산출 불가 — 여기 수치를 사용)

### A-1. P3 정정 — 0.5945 → **0.5944**

standard 클러스터 라우팅률은 pooled와 seed별 평균이 **동일하게 0.594444**다.

```
pooled : 6420/10800 = 0.594444
seed별 : 0.598611 / 0.609167 / 0.575556 → 평균 0.594444
4자리 반올림 : 0.5944  (0.5945 아님)
```

cold-start 0.8017은 실측 0.801667로 **원고가 맞다**.

패치 v2는 "0.5945 유지(사용자·검증 확정)"로 적었으나 이는 잘못이다. 0.59444를 0.5945로
올림한 오기이므로 **0.5944로 고친다**. 본문 직접 수정이 아니라 매크로 소스에서 산출값을
확인·교체할 것(§B-2와 동일 원칙).

### A-2. P12 근거 — 431 s

```
seed별 이벤트당 : 433.9 / 427.3 / 432.3 s
총합/총횟수     : 65,537.9 s / 152 회 = 431.2 s
```

원고 434는 seed 0 단독값(433.9)의 반올림이다. 같은 문장의 50.7이 3-seed 평균이므로
단위가 섞인다. **431**로 정정한다.

### A-3. ★ P14 (신규) — "training alone is far cheaper, about 181.8 s"가 E5 조건이 아니다

**문제.** 181.8 s / 2,137 VLA-call 등가는 `results/e1/e1_latency.json`의 `anchor5`이고,
출처가 `checkpoints/libero_object_task0/train_summary.json (E2 C-L0, **warm-start from n=20**)`
— 즉 **E2 배치의 8,000 스텝 warm-start fit**이다. E5는 2026-08-17 판정(B-2)으로
**scratch 학습 + 배치 등가 스텝**(n=20 → 10,000 / n=80 → 28,000)을 쓰므로 조건이 다르다.

**결과적으로 논지가 뒤집힌다.** 현재 문장은 "이벤트 434 s 중 학습만 보면 181.8 s로 훨씬 싸다"로
읽혀, 형성 시간의 대부분이 probe·준비인 듯한 인상을 준다. 실측은 정반대다.

```
3-seed 통합 152 이벤트 (n=20 99회 / n=80 53회), 학습 속도 44.0 스텝/초(E2 앵커)

  n=20 : 이벤트 289 s = 학습 227 s (10,000 스텝) + probe·준비 62 s
  n=80 : 이벤트 697 s = 학습 636 s (28,000 스텝) + probe·준비 60 s
  가중평균 : 이벤트 431.2 s · 그중 학습 369.9 s  → **학습이 86%**

  VLA-호출 등가 (÷ 85.07 ms)
    학습만      4,348 calls
    이벤트 전체 5,068 calls
    [원고] E2 앵커 2,137 calls — E5 재학습 비용의 절반 이하
```

**권고 수정 (택1, 판정 필요).**

- **(a) E5 실측으로 교체** — 가장 정확하다.
  > Formation time covers each retraining event end to end—habit training, the $P=20$
  > off-stream certification probes, and data preparation—averaging about 431 s over 50.7
  > events, of which habit training accounts for about 86% (227 s at $n{=}20$, 636 s at
  > $n{=}80$); a single retraining event therefore costs roughly 5,000 VLA-call equivalents.

- **(b) 앵커 유지 + 범위 명시** — 최소 수정이나 손익분기 논의가 약해진다.
  > ... averaging about 431 s over 50.7 events. For reference, a warm-started $n{=}40$ batch
  > fit takes about 181.8 s ($\approx$2,137 VLA-call equivalents) under the batch schedule;
  > online retraining uses batch-equivalent step budgets and is correspondingly larger.

**(a)를 권한다.** V-F가 "operational-stage amortization이지 total-compute break-even이 아니다"를
주장하는 절이므로, 형성 비용을 과소 기재하면 그 주장의 보수성이 약해져 리뷰어에게 역효과다.

---

## B. 적용 절차 (순서 고정 — 단계 건너뛰기 금지)

### 1. 본문·수식 패치
패치 v2의 P1–P11·P13, §D 문장 교체표, §E 참고문헌을 `main.tex`에 적용한다.
**[16] 제거는 제외** — 4단계로 미룬다.
각 항목은 "현재 → 수정" 대조 형식이므로 **현재 문자열이 원고와 일치하는지 확인 후** 교체한다.
불일치 발견 시 **임의 조정 금지, 멈추고 해당 항목 보고**.

### 2. 수치 매크로 (P3·P12·P14) — 본문 직접 수정 절대 금지
수동 숫자 입력 0 원칙(CLAUDE.md §6)이 적용된다.

- `numbers.json`에서 해당 매크로의 **출처 필드를 먼저 열어 보고**한다
  (standard 라우팅률 / 형성 이벤트당 시간 / 학습 비용 — seed 0 단독 산출인지 확인).
- `build_numbers.py`의 집계 소스를 3-seed 기준으로 교체한다.
  기대값: standard 라우팅 **0.5944**, 이벤트당 **431.2 s**, 학습분 **369.9 s / 4,348 calls**.
- P14는 (a)/(b) 판정 후 적용한다. **판정 전에는 손대지 말 것.**

### 3. 검증 파이프라인
`build_numbers.py` → `verify_numbers.py` 전건 통과 확인.
v10 기준 주장·리터럴 수에서 증가가 있으면(P8·P13·P14의 문장 추가로 소폭 증가 예상)
**증가 내역을 항목별로 보고**한다. 패치에 대응하지 않는 증가는 멈추고 보고.

### 4. 참고문헌 [16] 제거 + 재배열
번호 재배열이 본문 인용 전체에 파급되므로 **다른 수정이 모두 검증 통과한 뒤 마지막에** 수행한다.
II-B를 `"Knowledge distillation [15] and speculative decoding [17] move expensive computation
onto cheaper paths, ..."`로 고친 뒤 [16] 제거 → 전체 재배열 → 본문 인용 일괄 갱신 → **verify 재실행**.

### 5. 초기 투고본 포맷
- `"Manuscript received August 27, 2026."` 제거 확인.
- `ieeeconf` 재컴파일 → **페이지 수, Fig. 2/3/4/5·Table I/II 배치 페이지를 보고**한다.
- 8쪽 초과 시 임의 압축 금지 — 초과량과 원인(플로트 이동·캡션 폭 등)만 보고하고 우선순위는 연구원이 정한다.

### 6. 산출물
`main_v11.tex` + 컴파일 PDF, 그리고 단계별 보고:
(1) 문자열 불일치 목록 (2) P3·P12·P14 매크로 출처 확인 결과 (3) verify 증감 내역
(4) [16] 재배열 후 재검증 (5) 페이지·플로트 배치

---

## C. 판정 대기 항목 (적용 전 연구원 확인 필요)

1. **P14 (a)/(b) 택1** — 형성 비용 서술 방식. (a) 권고.
2. **P3 0.5944 정정 승인** — 패치 v2가 "0.5945 유지"로 적었으나 실측은 0.594444다.

---

## D. 이미 종결된 확인 (재확인 불요)

실험 환경에서 원장·코드 실측으로 확정했다. 원고 환경에서 다시 검증할 필요가 없다.

| 항목 | 확정 내용 |
|---|---|
| 검정명 `p=0.0968` | `fisher_exact(alternative="less")` → **one-sided Fisher exact test** |
| τ 스코프 | **per-cluster** (`ClusterState`마다 `ACIRiskController`, 타 클러스터 τ 변화 0건) |
| ever-matured | (22, 18, 21) → **20.3 ± 2.1** / final-M (19, 16, 20) → 18.3 ± 2.1 |
| cold-start 라우팅 | **full-stream episode-weighted**에서 일치 (last-1000은 0.6684/0.3717로 불일치) |
| split-conformal | **진짜 split** — 결정적 셔플 반반 분할, μ·Σ는 fit 절반 / 분위수는 disjoint calib 절반, ⌈(n+1)(1−α)⌉ 보정 |
| between-category share | censored ranks 위 $SS_{between}/SS_{total}$ = **순위 기반 $\eta^2$** |
| 절단 1개 | `libero_10_task0` (**long 스위트** → Fig. 2(b) 22셀 밖) |
| 73.47% | seed별 비율(0.7277, 0.7643, 0.7120)의 **비가중 평균** — 원고가 맞음 (pooled는 73.66%) |
| 그 외 원고 수치 | 재학습 50.7±0.6 · probe 0.3838/0.5262 · 노출 22.7±0.6 · 강등 11/재성숙 5 · X·I 0.2983/0.1067 · X 7–10개 · 형성가능 6–9 · teacher 성공 56–60 · 운영 9.34±0.24 h · 형성 6.07±0.12 h · 시스템 0.9608±0.0042 vs 0.9629±0.0052 — **전건 일치** |

---

## E. 행동 규칙

- 패치·지시서와 실제 원고가 어긋나면 **멈추고 보고**한다. try/except 우회, 임의 단순화,
  "되게 만들기" 금지 (CLAUDE.md §0).
- verify 실패 시 실패 항목의 원고 문장과 JSON 값을 나란히 제시한다.
- 이 지시서에 없는 개선 사항을 발견하면 **적용하지 말고 목록으로만 보고**한다.
