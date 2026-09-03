# depth privileged-information confound 스크리닝 — 판정 요청

작성 2026-08-28 · Stage 1 · 실행 `experiments/rgb_depth_ablation/`

> **판정 요청 1건**: 자동 판정은 **CASE B**이고 지시서 §11은 CASE B에 **C(25 클러스터 재실행)** 를 지정한다. 그러나 증거는 경계선이며 실험자 권고는 **B(6 클러스터 결과를 ablation 절로 추가)** 다. **B와 C 중 택일이 필요하다.**

---

## 1. 무엇을 했나

habit이 teacher에 없는 depth를 받는다는 설계 사실이 "높은 습관 성능 = privileged sensing"이라는 리뷰어 비판을 부를 수 있다. 이를 데이터로 검사했다.

- **기존 실험 무수정.** 별도 경로(`experiments/rgb_depth_ablation/`, `results/rgb_depth_ablation/`, `checkpoints/rgb_only_ablation/`)에서만 작업했고 기존 RGB-D 체크포인트는 읽기 전용으로 재사용했다.
- **차이는 depth 하나뿐.** conv1을 4채널 → 3채널로 좁힌 것이 전부다 (파라미터 차 6,272 = 0.0066%). teacher 궤적·`B_k` 순서·optimizer·스케줄·seed·해상도·시점·평가 스펙이 모두 동일하다 (16항목 감사: `CONFIG_DIFF.md`).
- **클러스터는 결과 산출 전 고정.** 스위트별 N* 최소/최대 규칙으로 6개 (easy 2 · medium 1 · difficult 2 · censored 1), spatial·long 필수 포함 (`ABLA_RGBD_CLUSTER_SELECTION.md`).
- **기존 결과 재현 확인.** 동일 RGB-D 체크포인트를 50-trial로 재평가해 기존 E3 20-trial과 공통 uid **80/80 완전 일치**. 실행 결정성이 확인된 뒤에야 해석에 들어갔다.
- **에피소드 단위 paired.** 두 조건이 같은 held-out uid를 본다 → 1,200 paired 에피소드.

## 2. 결과
### 2.1 클러스터별

| Cluster | Suite | 난이도 | N*(RGB-D) | N*(RGB) | ŝ(80) RGB-D | ŝ(80) RGB | Δŝ(80) |
|---|---|---|---|---|---|---|---|
| object_task1 | object | easy | 10 | 20 | 0.880 | 0.960 | +0.080 |
| object_task0 | object | difficult | 80 | 40 | 0.960 | 0.900 | -0.060 |
| goal_task1 | goal | easy | 10 | 10 | 0.980 | 0.960 | -0.020 |
| goal_task0 | goal | difficult | 80 | 80 | 0.920 | 0.820 | -0.100 |
| spatial_task1 | spatial | medium | 20 | 20 | 0.960 | 0.980 | +0.020 |
| 10_task0 | long | censored | >80 | >80 | 0.740 | 0.600 | -0.140 |

### 2.2 전체 paired

- 평균 차 **-0.0150**, bootstrap 95% CI **[-0.0392, +0.0092]** — **0을 포함**
- 불일치쌍 RGB-only만 성공 100 / RGB-D만 성공 118 · exact McNemar **p = 0.2495**

### 2.3 n별 Δ (6 클러스터 평균, pp)

| n | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| Δ | -12.00 | +0.00 | +9.67 | -3.67 |

**비단조**. depth는 저데이터 구간(n=10)에서 12.00 pp를 벌어주고 n=40에서는 오히려 RGB-only가 +9.67 pp 앞선다.

### 2.4 실패 유형

두 조건 모두 **전 실패가 timeout 계열**(에피소드 상한 도달), 기타 0건. depth 제거가 조기 파국을 만들지 않는다.

> 로그에서 자동 분류 가능한 것은 timeout 계열뿐이다(steps가 상한 도달). grasp/localization/placement/trajectory 구분은 영상 판독이 필요하며 본 스크리닝 범위 밖 — 미분류로 보고한다.

## 3. 판정 요청 — B인가 C인가

지시서 §11의 자동 판정 기준은 "n=80 평균 감소 < 3 pp → CASE A"이고, 실측은 **3.67 pp**로 **0.67 pp 초과**해 CASE B로 떨어진다. §11의 CASE B 처방은 C(25 클러스터 재실행)다.

**다만 그 초과분의 출처가 한 곳이다.**

| 근거 | CASE A를 지지 | CASE B를 지지 |
|---|---|---|
| 전체 paired | CI [-0.0392, +0.0092] 0 포함, McNemar p=0.2495 | — |
| n=80 평균 | long 제외 시 감소 **1.60 pp** (기준 이내) | 전체 **3.67 pp** (기준 초과) |
| N* | 4/6 동일, 이동 2개는 **+1/−1 상쇄** | — |
| 스위트 | spatial +0.02 (차이 없음) | long **-0.14** |
| 실패 유형 | 전부 timeout, 조기 파국 없음 | — |

long 클러스터(`libero_10_task0`)는 **RGB-D에서도 N*>80으로 우측절단**된, 애초에 형성되지 않은 셀이다. 원고도 이를 유일한 절단 사례로 이미 보고한다.

**실험자 권고: B.** 리뷰어 비판의 실질은 "습관의 높은 성능이 depth 덕분"인데, 논문의 주장이 걸려 있는 **천장에서 차이가 검출되지 않는다**. 층화 표본에서 결과가 한 방향으로 쏠리지도 않는다(6개 중 2개는 RGB-only 우위). C를 해도 결론이 "천장 동일, 저데이터 구간에서만 차이"를 벗어날 가능성이 낮아 비용 대비 이득이 낮다. C 비용은 약 **15시간**(RGB-only 학습 4.4h + 양 조건 평가 10.4h).

**그러나 자동 판정이 CASE B로 떨어진 것은 사실이므로, 지시서 §18("결론은 데이터에 따라 결정한다")에 따라 실험자가 단독으로 기준을 완화하지 않고 판정을 요청한다.**

## 4. B 채택 시 즉시 사용 가능한 산출물

`paper/`에 논문(v11) 서식으로 준비돼 있다. 수동 숫자 입력 0 — 전 수치가 `\Num{Abla*}` 매크로다.

| 파일 | 내용 |
|---|---|
| `paper/SECTION_depth_ablation.tex` | 본문 절 (삽입 위치 = V-F Scope 직전) |
| `paper/TABLE_depth_ablation.tex` | Table III |
| `paper/fig_depth_ablation.pdf` | Fig. 6, 2-패널 double-column |
| `paper/ablation_numbers.json` | 매크로 21개 + 각 값의 source 경로 |
| `paper/INSTRUCTION_depth_ablation_for_manuscript_agent.md` | 적용 절차 |

## 5. 패키지 구성

| 경로 | 내용 |
|---|---|
| `RGB_DEPTH_ABLATION_AUDIT.md` | 지시서 §16 A~E 감사 보고서 |
| `results/rgb_depth_ablation/` | 원자료 12 JSON + summary + CSV + 그림 3종 |
| `experiments/rgb_depth_ablation/` | 실행·분석·산출 스크립트 전부 |
| `habits/` | ACT 구현 (in_ch 스위치 포함) |
| `configs/preregistration.md` · `log.md` | 전문 |
| `git_log.txt` | 커밋 이력 |