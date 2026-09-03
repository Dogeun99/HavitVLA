# depth privileged-information confound — Stage 1 결과·데이터 패키지

작성 2026-08-28 · **전체 재실험(Stage 2) 전 분석용** · `depth_ablation_pack_20260828.tar.gz`(판정 요청본)을 **대체**한다.

> 연구원 결정: 본 패키지를 분석한 뒤 **전체 재실험을 실시**한다. 따라서 본 문서는 판정 요청이 아니라 ①Stage 1이 무엇을 확정했고 ②무엇이 미해결로 남았으며 ③Stage 2가 무엇을 고정해야 하는지를 넘기는 인수인계다.

---

## 1. Stage 1이 확정한 것

6 클러스터 × n{10,20,40,80} × held-out 50 = **1,200 paired 에피소드**. 두 조건의 차이는 conv1 4채널→3채널 하나뿐이다(파라미터 차 6,272 = 0.0066%).

| 확정 사항 | 실측 |
|---|---|
| 전체 paired 차이 | **-0.0150**, 95% CI [-0.0392, +0.0092] — **0 포함** |
| exact McNemar | p = **0.2495** (불일치 100/118) |
| N* 불변 | **4/6**, 이동 2개는 +1/−1 상쇄 |
| n별 Δ (pp) | n=10 -12.00 · n=20 +0.00 · n=40 +9.67 · n=80 -3.67 — **비단조** |
| n=80 평균 감소 | 전체 **3.67 pp** / long 제외 **1.60 pp** |
| 실패 유형 | 양 조건 **전부 timeout 계열**, 기타 0건 |
| 기존 결과 재현 | 동일 RGB-D 체크포인트 50-trial 재평가 → 기존 E3 20-trial과 공통 uid **80/80 일치** |

**해석.** depth는 저데이터 구간(n=10, −12.00 pp)의 수렴을 돕고 **천장에는 기여하지 않는다**. 논문의 주장(성숙 인증 후 teacher 대체)은 천장에 걸려 있으므로 privileged-sensing 비판은 이 표본에서는 지지되지 않는다.

## 2. 미해결로 남은 것 — Stage 2가 답해야 할 질문

1. **long-horizon.** 유일한 long 클러스터 `libero_10_task0`가 Δ = -0.14로 가장 크다. 그런데 이 셀은 **RGB-D에서도 N*>80으로 우측절단**돼 애초에 형성되지 않는다. 형성 실패 위에서 측정한 차이라 해석이 성립하지 않는다. long 스위트에 형성되는 셀이 있는지, 있다면 거기서도 −14 pp가 재현되는지가 핵심이다.
2. **표본 크기.** n=80 평균 감소가 3.67 pp로 기준 3 pp를 0.67 pp 초과하는데, 이 초과가 long 1개에서 나온다. 6개 표본에서 클러스터 1개의 영향력이 지나치게 크다.
3. **spatial 가설 반증.** depth 의존 후보로 지목한 spatial이 Δ = +0.02에 N* 불변이었다. spatial 클러스터가 2개뿐이라 "기하 과제가 depth를 필요로 한다"는 가설을 6개로는 검정할 수 없다.
4. **실패 유형 분해.** 전 실패가 timeout이라는 사실만 확인했다. grasp/localization/placement 분리는 per-episode 비디오가 필요해 Stage 1 범위 밖이었다.
5. **온라인 lifecycle.** 배치 형성만 다뤘다. 재학습 예산·probe 결과와 depth가 상호작용하는지는 미검증.

## 3. Stage 2 규모 — ★ 25가 아니라 27이다

지시서 §11은 "25 클러스터 재실행"으로 적었으나, **E3 배치 원장의 실제 클러스터는 27개**다(`results/e3/e3_curves.json`의 `n_clusters_reported`, completeness missing=[] 확인). 그중 **6개는 Stage 1에서 이미 완료**됐고 **21개가 남는다**. Stage 2 착수 전에 이 수를 확정해야 한다.

**남은 클러스터 21개**

```
  libero_object_task2               libero_object_task3               libero_object_task4
  libero_object_task5               libero_object_task6               libero_object_task7
  libero_object_task8               libero_object_task9               libero_goal_task2
  libero_goal_task3                 libero_goal_task4                 libero_goal_task5
  libero_goal_task6                 libero_goal_task7                 libero_goal_task8
  libero_goal_task9                 libero_spatial_task0              libero_10_task2
  libero_10_task5                   chained_libero_object_task0       chained_libero_object_task5
```

### 실측 기반 비용 (Stage 1 wall-clock에서 산출, 하드코딩 없음)

| 항목 | 클러스터당 실측 | 남은 21개 |
|---|---|---|
| RGB-only 학습 (n 4개 합) | 10.4 분 (범위 10.4–10.5) | **3.7 시간** |
| RGB-only 평가 (4 ckpt × 50) | 9.4 분 | 3.3 시간 |
| RGB-D 재평가 (4 ckpt × 50) | 9.2 분 | 3.2 시간 |
| **합계** | | **10.2 시간** |

클러스터당 학습 1회는 10.4분인데 평가는 두 조건 합쳐 18.6분으로 **평가가 전체의 64%** 를 차지한다. `chained_*` 2개는 커스텀 래퍼라 에피소드가 길어 위 평균보다 더 걸릴 수 있다.

### Stage 2가 고정해야 할 것 (Stage 1과 동일하게 유지)

- **동일성 16항목** — `CONFIG_DIFF.md`. depth 외 어떤 것도 바꾸지 않는다 (width·encoder·steps·augmentation·lr·태스크별 튜닝 전부 금지).
- **paired 성립** — 두 조건이 같은 held-out uid를 본다. `heldout_specs(suite, task, 50)`.
- **기존 RGB-D 체크포인트 재사용** — 재학습하지 않는다. 재평가만 한다.
- **Stage 1 6개를 다시 돌리지 않는다** — 이미 완료됐고 결정성이 확인됐다. 체크포인트 SHA256이 `checkpoint_manifest.json`에 있다(8.5 GB, 24개).
- **클러스터 전수** — Stage 1처럼 층화 표집하지 않으므로 선택 편의 문제가 사라진다.

## 4. 패키지 구성

| 경로 | 내용 |
|---|---|
| `RGB_DEPTH_ABLATION_AUDIT.md` | 지시서 §16 A~E 감사 보고서 |
| `results/rgb_depth_ablation/*.json` | **원자료 12개** — 클러스터×조건, per-episode (uid·outcome·steps) 전량 |
| `results/rgb_depth_ablation/ablation_summary.json` | 분석 단일 진입점 |
| `results/rgb_depth_ablation/table_detail.csv` | 지시서 §14 전 필드 |
| `results/rgb_depth_ablation/fig_{A,B,C}*.png` | 형성 곡선 · Δŝ(80) · N* paired |
| `paper/` | 논문 서식 산출물 (본문 절·Table·Fig·매크로 21개·원고 지시서) |
| `logs/rgb_depth_ablation/` | **학습 6 + 평가 12 + run.log** = 19개 원본 로그 |
| `train_summaries/` | 클러스터별 final_l1 · train_seconds |
| `checkpoint_manifest.json` | RGB-only 체크포인트 24개 SHA256 (가중치는 디스크에 잔류) |
| `stage2_remaining_clusters.json` | 남은 클러스터 목록 + 실측 비용 |
| `experiments/rgb_depth_ablation/` | 실행·분석·산출 스크립트 전부 |
| `habits/` | ACT 구현 (`in_ch` 스위치 포함) |
| `configs/preregistration.md` · `log.md` · `CLAUDE.md` | 전문 |
| `git_log.txt` | 커밋 이력 |

**체크포인트 가중치는 포함하지 않았다** — 8.5 GB로 패키지에 넣을 수 없다. `checkpoints/rgb_only_ablation/`에 그대로 있고 SHA256으로 대조 가능하다.
