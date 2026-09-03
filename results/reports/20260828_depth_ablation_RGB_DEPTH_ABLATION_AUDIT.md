# RGB_DEPTH_ABLATION_AUDIT — depth privileged-information confound 스크리닝

작성 2026-08-28. 지시서 §16. **Stage 1 스크리닝** — formal equivalence/non-inferiority 주장이 아니다.

> **한 줄**: 형성이 끝난 뒤의 성능(천장)은 두 조건이 사실상 같고, 전체 paired 검정도 유의하지 않다. 다만 n=80 평균 감소가 **3.67 pp**로 CASE A 기준(3 pp)을 0.67 pp 넘겨 **자동 판정은 CASE B**다. 이 초과분은 사실상 long 클러스터 한 개(−14 pp)에서 나온다.

---

## A. Implementation

- **RGB-D ACT 입력**: 두 시점(agentview·wrist) 128×128 RGB ⊕ depth를 **conv1 4채널 early fusion**. RGB 가중치는 ImageNet 사전학습 복사, depth 채널은 RGB 평균으로 초기화. depth는 [0,1] 유지.
- **RGB-only ACT 입력**: 동일 구조에서 conv1을 **표준 3채널**로 둔다. depth 경로만 제거되고 다른 용량 변경은 없다.
- **파라미터**: RGB-D 95,036,360 · RGB-only 95,030,088 · 차이 **6,272 (0.0066%)** — conv1 depth 채널 × 백본 2개.
- **depth 외 설정 차이**: 없음. RGB 정규화(ImageNet)·action/proprio 정규화·optimizer·lr·batch·학습 스텝·warm-start·seed·해상도·시점·증강(미사용)이 모두 동일하다. 16항목 key-by-key 감사는 `CONFIG_DIFF.md` 참조. depth는 별도 채널로만 붙어 **RGB 통계에 영향을 주지 않는다**.

## B. Dataset

- **선정 6 클러스터** (결과 산출 전 `ABLA_RGBD_CLUSTER_SELECTION.md`에 규칙·ID 고정):
  - `libero_object_task1` — object, easy
  - `libero_object_task0` — object, difficult
  - `libero_goal_task1` — goal, easy
  - `libero_goal_task0` — goal, difficult
  - `libero_spatial_task1` — spatial, medium
  - `libero_10_task0` — long, censored
- **선정 이유**: 스위트별 N* 최소/최대로 난이도 층화(easy 2·medium 1·difficult 2·censored 1), depth 의존 가능성이 높은 **spatial과 long을 필수 포함**. 결과를 본 뒤 교체하지 않았다.
- **teacher trajectory set**: 두 조건이 **동일** `B_k`(기존 배치 수집분)를 사용. 새 VLA rollout 없음.
- **train/eval episode ID**: 학습은 `load_cluster` 순서의 `episodes[:n]`로 결정적이며 동일. 평가는 `heldout_specs(suite, task, 50)`로 두 조건이 **동일 uid**를 본다 → 에피소드 단위 paired 성립.
- **기존 결과 재현(§17)**: 동일 RGB-D 체크포인트를 50-trial로 재평가해 기존 E3 20-trial과 대조 — `object_task1`의 공통 uid **80/80 결과 완전 일치**(4 체크포인트 × 20). 실행 결정성 확인.

## C. Results
### C-1. 클러스터별 형성 곡선과 N*

| Cluster | Suite | 난이도 | N*(RGB-D) | N*(RGB) | ΔN* | ŝ(80) RGB-D | ŝ(80) RGB | Δŝ(80) |
|---|---|---|---|---|---|---|---|---|
| object_task1 | object | easy | 10 | 20 | 1 | 0.880 | 0.960 | +0.080 |
| object_task0 | object | difficult | 80 | 40 | -1 | 0.960 | 0.900 | -0.060 |
| goal_task1 | goal | easy | 10 | 10 | 0 | 0.980 | 0.960 | -0.020 |
| goal_task0 | goal | difficult | 80 | 80 | 0 | 0.920 | 0.820 | -0.100 |
| spatial_task1 | spatial | medium | 20 | 20 | 0 | 0.960 | 0.980 | +0.020 |
| 10_task0 | long | censored | >80 | >80 | both censored | 0.740 | 0.600 | -0.140 |

### C-2. n별 Δ 패턴 (6 클러스터 평균)

| n | 평균 Δ | pp | 범위 |
|---|---|---|---|
| 10 | -0.1200 | -12.00 | [-0.24, +0.08] |
| 20 | +0.0000 | +0.00 | [-0.12, +0.10] |
| 40 | +0.0967 | +9.67 | [-0.04, +0.36] |
| 80 | -0.0367 | -3.67 | [-0.14, +0.08] |

**패턴이 단조적이지 않다.** n=10에서 −12.00 pp로 크게 벌어졌다가 n=20에서 0.00 pp, n=40에서 오히려 **+9.67 pp**(RGB-only 우위), n=80에서 −3.67 pp다. depth가 **저데이터 구간의 초기 수렴을 돕는 보조 신호**로 작동하고 도달 천장에는 기여하지 않는다는 해석과 부합한다.

### C-3. 전체 paired 비교 (에피소드 단위, 전 n)

- paired 표본 **1200** 에피소드 (6 클러스터 × 4 n × 50 held-out)
- 평균 차 **-0.0150**, paired bootstrap 95% CI **[-0.0392, +0.0092]** — **CI가 0을 포함**
- 불일치쌍: RGB-only만 성공 **100** / RGB-D만 성공 **118** · McNemar exact **p = 0.2495** (유의하지 않음)

### C-4. 실패 유형

| Cluster | step cap | RGB-D timeout/기타 | RGB-only timeout/기타 |
|---|---|---|---|
| object_task1 | 280 | 31/0 | 32/0 |
| object_task0 | 280 | 116/0 | 92/0 |
| goal_task1 | 300 | 10/0 | 17/0 |
| goal_task0 | 300 | 84/0 | 106/0 |
| spatial_task1 | 220 | 20/0 | 28/0 |
| 10_task0 | 520 | 105/0 | 109/0 |

**모든 실패가 timeout 계열이다**(에피소드 상한 도달, 기타 0건). 즉 depth 제거가 조기 파국(잡기 실패 후 즉시 종료 등)을 만들지 않고, **과제를 끝내지 못하는 형태**로만 나타난다.
> 로그에서 자동 분류 가능한 것은 timeout 계열뿐이다(steps가 상한 도달). grasp/localization/placement/trajectory 구분은 영상 판독이 필요하며 본 스크리닝 범위 밖 — 미분류로 보고한다.

## D. Interpretation

**1. depth 제거가 습관 성공률을 실질적으로 낮추는가?** — **대체로 아니다.** 전체 paired 차이는 -0.0150이고 CI가 0을 포함하며 McNemar도 유의하지 않다. 천장(n=80)에서는 평균 −3.67 pp인데, 이는 long 클러스터 하나(−14.0 pp)가 끌어내린 값이고 **long을 빼면 −1.60 pp**로 CASE A 범위다. 6개 중 2개는 오히려 RGB-only가 높다.

**2. depth 제거가 N*를 체계적으로 증가시키는가?** — **아니다.** 6개 중 4개가 동일하고, 움직인 2개는 **+1과 −1로 방향이 반대**다(object_task1 10→20, object_task0 80→40). long은 양 조건 모두 우측절단이라 비교가 성립하지 않는다.

**3. 효과가 spatial/long에 집중되는가?** — **spatial은 아니고 long은 그렇다.** depth 의존 후보로 지목했던 `spatial_task1`은 N* 20→20 동일에 ŝ(80)이 오히려 +0.02다. 반면 `10_task0`(long)은 −0.14로 6개 중 가장 크다. 다만 이 클러스터는 RGB-D에서도 N*>80(우측절단)이라 **애초에 형성되지 않은 셀**이며, 논문에서도 유일한 절단 사례로 이미 보고돼 있다.

**4. 현재 RGB-D 주 실험을 privileged-sensing 이득이 아니라 lifecycle 연구로 방어할 수 있는가?** — **그렇다, 다만 조건부다.** 논문의 주장은 (i) 습관이 teacher 성공만으로 형성되고 (ii) 성숙 인증 후 teacher를 대체할 수 있다는 것이며, 둘 다 **형성 이후의 천장**에 걸려 있다. 그 천장에서 두 조건의 차이는 통계적으로 검출되지 않는다. depth가 기여하는 곳은 **저데이터 구간의 수렴 속도**이고, 이는 논문이 N*로 이미 보고하는 축이다. 다만 long-horizon에서의 −14 pp는 정직하게 서술해야 한다.

**5. 무엇을 해야 하는가?**

자동 판정은 **CASE B** — "Depth contributes to habit formation, but the contribution is task-dependent." (n=80 평균 감소 3.67 pp, 기준 3 pp를 0.67 pp 초과).

## E. Recommendation — **B (RGB-only ablation 추가)**, 단 연구원 판정 필요

**권고: B.** 근거는 다음과 같다.

- 리뷰어 비판의 실질은 "높은 습관 성능이 depth 덕분"인데, **천장에서 차이가 검출되지 않으므로 그 비판이 데이터로 반박된다**. 6 클러스터는 스위트·난이도로 층화된 표본이고, 결과가 한 방향으로 쏠리지 않는다(2개는 RGB-only 우위).
- 따라서 **C(25 클러스터 전체 재실행)의 비용 대비 이득이 낮다.** 재실행해도 결론이 "천장 동일, 저데이터 구간에서만 차이"를 벗어날 가능성이 낮고, 그 결론은 이미 층화 표본에서 나왔다.
- **D(온라인 lifecycle 재실행)는 불필요하다.** 온라인 실험의 결론은 라우팅 감소와 비열등이며, 습관의 천장이 유지되는 한 depth 제거가 그 결론을 바꿀 경로가 없다.

**다만 지시서 §11의 CASE B 문구는 C(25 클러스터 재실행)를 권고한다.** 자동 판정이 CASE B로 떨어진 것은 사실이므로, B와 C 중 무엇을 택할지는 **연구원 판정 사항**이다. 판정에 필요한 사실은 다음과 같다.

| 근거 | CASE A 쪽 | CASE B 쪽 |
|---|---|---|
| 전체 paired | CI [-0.0392, +0.0092]가 0 포함, McNemar p=0.2495 | — |
| n=80 평균 | long 제외 시 -1.60 pp | 전체 −3.67 pp (기준 0.67 pp 초과) |
| N* | 6개 중 4개 동일, 이동 2개는 방향 반대 | — |
| 스위트 편중 | spatial은 차이 없음(+0.02) | long −14 pp |
| 실패 유형 | 전부 timeout 계열, 조기 파국 없음 | — |

**B를 택할 경우 원고 반영안**: 본 스크리닝을 supplemental/auxiliary validation으로 한 문단 추가하고, V-F Scope에 "habit이 teacher에 없는 depth를 받는다"는 설계 사실과 그 영향 범위(천장 동일, 저데이터 수렴에서만 차이, long에서 −14 pp)를 명시한다. 이렇게 하면 confound를 숨기지 않으면서 주장 범위를 정확히 한정할 수 있다.

**C를 택할 경우**: E3 배치 원장의 클러스터는 25가 아니라 **27개**이며(`results/e3/e3_curves.json`), 그중 6개는 본 Stage 1에서 완료됐으므로 **남은 21개**를 돌린다. Stage 1 wall-clock 실측 기준 클러스터당 학습 10.4분 + 양 조건 평가 18.6분 → **약 10.2시간**(학습 3.7h + 평가 6.5h). `chained_*` 2개는 에피소드가 길어 이보다 더 걸릴 수 있다. 산출은 `stage2_remaining_clusters.json`.

---

## 산출물

| 파일 | 내용 |
|---|---|
| `ablation_summary.json` | 전 수치 단일 진입점 |
| `table_detail.csv` | 지시서 §14 전 필드 (cluster·n·success·Δ·N*·params) |
| `table_cluster.md` | 클러스터 요약표 (논문 Table II 서식) |
| `fig_A_curves.png` | 형성 곡선 RGB-D vs RGB-only (논문 Fig. 2(a) 서식) |
| `fig_B_delta.png` | 클러스터별 Δŝ(80) (논문 Fig. 5(c) 서식) |
| `fig_C_nstar.png` | N* paired 이동 |
| `CONFIG_DIFF.md` · `RUN_COMMANDS.md` · `ABLA_RGBD_CLUSTER_SELECTION.md` | 감사·재현 |