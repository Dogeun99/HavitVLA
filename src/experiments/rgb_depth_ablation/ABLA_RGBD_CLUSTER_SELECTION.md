# RGB-D vs RGB-only 스크리닝 — 클러스터 선정 (결과 산출 전 고정)

작성 2026-08-28. **RGB-only 학습·평가를 시작하기 전에** 확정한다.
목적: 결과를 본 뒤 유리한 클러스터를 고르는 것을 원천 차단.

## 선정 규칙 (지시서 §5)

구성: object 2 · goal 2 · spatial 1 · long 1 = **6 클러스터**.
depth가 도움될 가능성이 높은 **spatial과 long을 반드시 포함**한다.

난이도 층화는 기존 E3 배치 형성 결과의 N*를 사용한다(`results/e3/e3_curves.json`).
규칙은 다음 순서로 기계적으로 적용하며, 동점은 task_id 최소로 깬다.

1. **object**: N* 최소 1개(easy) + N* 최대 1개(difficult)
2. **goal**: N* 최소 1개(easy) + N* 최대 1개(difficult)
3. **spatial**: 2개 중 N* 최대 1개 — depth 의존 가능성이 큰 쪽을 택한다
4. **long**: 3개 중 N* 최대 1개 — 우측절단(>80)이 있으면 그것을 택한다

## 적용 결과

| 클러스터 | 스위트 | N* | ŝ(80) | 난이도군 | 선정 사유 |
|---|---|---|---|---|---|
| `libero_object_task1` | object | 10 | 0.75 | easy | object 내 N* 최소 · 동점 중 task_id 최소 |
| `libero_object_task0` | object | 80 | 0.95 | difficult | object 내 N* 최대 |
| `libero_goal_task1` | goal | 10 | 1.00 | easy | goal 내 N* 최소 · 동점 중 task_id 최소 |
| `libero_goal_task0` | goal | 80 | 0.85 | difficult | goal 내 N* 최대 |
| `libero_spatial_task1` | spatial | 20 | 0.95 | medium | spatial 2개 중 N* 최대 (필수 포함) |
| `libero_10_task0` | long | **>80** | 0.75 | censored | long 3개 중 N* 최대(우측절단) (필수 포함) |

난이도 분포: easy 2 · medium 1 · difficult 2 · censored 1 — 층화 요건 충족.

## 고정 사항

- 위 6개 외 클러스터는 Stage 1에서 사용하지 않는다.
- 결과가 불리해도 클러스터를 교체하지 않는다. 교체가 필요하다고 판단되면
  **사유를 기록하고 연구원 판정을 받는다**(임의 교체 금지).
- `libero_10_task0`은 E3에서 n=80에서도 기준 미달(우측절단)이다. RGB-only에서도
  미달일 가능성이 높으며, 그 경우 ΔN*는 "둘 다 절단"으로 처리한다(Paper 1 규칙 승계).

## 평가 규모

지시서 §7에 따라 **held-out 50 trials / cluster / n**. E3 배치 평가는 20이었으므로
RGB-D도 50으로 **재평가**하며, 앞 20개 부분집합에서 기존 E3 결과와 대조해
재현성을 확인한다(지시서 §17 마지막 조건).

`heldout_specs(suite, task, n)`은 앞에서부터 n개를 취하므로 20은 50의 부분집합이다.
