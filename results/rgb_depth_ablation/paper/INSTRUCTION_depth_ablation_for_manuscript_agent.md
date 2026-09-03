# [지시] depth ablation 절 삽입 — 원고 작업 환경용

> 작성 2026-08-28. 원고 소스는 이 환경에 없다. **판정 완료 후에만 적용**한다.
> 근거 원자료: `ablation_summary.json` · 감사 보고서: `RGB_DEPTH_ABLATION_AUDIT.md`

## 0. 선행 조건 — 판정 전 적용 금지

자동 판정은 **CASE B**이고 실험자 권고는 **B(ablation 절 추가)**이지만, 지시서 §11의 CASE B
문구는 **C(25 클러스터 재실행)** 를 지정한다. **연구원이 B를 택한 경우에만** 아래를 적용한다.
C를 택하면 본 산출물은 6 클러스터 예비 결과로 보관하고 재실행 결과로 대체한다.

## 1. 파일

| 파일 | 용도 |
|---|---|
| `SECTION_depth_ablation.tex` | 본문 절. 삽입 위치 = **V-F Scope and Limitations 직전** (새 V-F가 되고 기존 V-F는 V-G로 밀린다) |
| `TABLE_depth_ablation.tex` | Table III (Table I·II 다음) |
| `fig_depth_ablation.pdf` / `.png` | 2-패널 double-column 그림. Fig. 6 |
| `ablation_numbers.json` | 매크로 21개 — value·formatted·**source 필드** |

## 2. 수치 주입 — 본문 직접 숫자 입력 금지 (CLAUDE.md §6)

`SECTION_depth_ablation.tex`의 모든 수치는 `\Num{Abla*}` 매크로다. 그대로 두고
`build_numbers.py`에 `ablation_numbers.json`을 소스로 등록한 뒤 `verify_numbers.py`를 재실행한다.
각 매크로의 `source` 필드가 `ablation_summary.json`의 어느 경로에서 나왔는지 명시돼 있으므로,
**등록 시 그 경로를 그대로 옮겨 적는다.** 값을 손으로 옮겨 적지 않는다.

주의가 필요한 두 매크로:

- `AblaDropEightyExLong` = **1.60** — long 클러스터를 뺀 n=80 평균 *감소폭*이다.
  원자료의 Δ는 −0.0160(부호가 반대)이며 산출식에서 부호를 뒤집는다. 본문은 "the mean drop is
  1.60 pp"로 읽히므로 부호를 다시 뒤집지 말 것.
- `AblaParamPct` = **0.0066** — 퍼센트 값이다. 본문에 `\%`가 이미 붙어 있다.

## 3. 참조 갱신

- 절 번호: 새 절이 V-F가 되면 기존 V-F(Scope and Limitations) 참조가 전부 밀린다.
  `\label`/`\ref` 사용 중이면 자동 갱신되지만, 본문에 절 번호를 문자로 적은 곳이 있으면 **보고**한다.
- Table/Fig 번호: Table III·Fig. 6이 마지막 번호인지 확인하고, 아니면 **임의 조정하지 말고 보고**한다.

## 4. V-F(Scope) 본문 한 문장 추가

새 절이 다루지 않는 잔여 범위를 Scope 절에 남긴다. 위치는 센서 구성 관련 문장 뒤.

> The screen in Section~\ref{sec:depth_ablation} covers \Num{AblaClusters} clusters at the batch
> stage; whether depth affects the online lifecycle---where retraining budgets and probe
> outcomes interact---is untested.

## 5. 검증

`build_numbers.py` → `verify_numbers.py` 전건 통과. 리터럴/주장 수 증가는 본 절 추가분과
일치해야 하며, **대응하지 않는 증가는 멈추고 보고**한다.

## 6. 행동 규칙

패치와 원고가 어긋나면 멈추고 보고한다. 임의 단순화·수치 조정·"되게 만들기" 금지.
본 지시서에 없는 개선 사항은 적용하지 말고 목록으로만 보고한다.
