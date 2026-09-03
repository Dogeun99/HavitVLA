# CONFIG_DIFF — RGB-D vs RGB-only (key-by-key)

지시서 §15. **허용되는 차이는 depth 관련 키뿐**이어야 한다.

## 코드 경로 차이 (전수)

| 위치 | RGB-D (기존) | RGB-only (ablation) | 성격 |
|---|---|---|---|
| `habits/act.py: build_backbone(in_ch)` | `in_ch=4`, conv1을 4채널로 확장 후 depth 채널을 RGB 평균으로 초기화 | `in_ch=3`, 표준 ResNet18 conv1 그대로 | **depth 경로** |
| `habits/act.py: ACTPolicy(in_ch)` | 4 | 3 | **depth 경로** |
| `habits/dataset.py: make_frame_tensor(use_depth)` | RGB(ImageNet 정규화) ⊕ depth[0,1] → (4,H,W) | RGB(ImageNet 정규화) → (3,H,W) | **depth 경로** |
| `habits/dataset.py: ClusterDataset(use_depth)` | True | False | **depth 경로** |
| `habits/policy.py: HabitPolicy` | 체크포인트 `use_depth`로 자동 분기 (미기재 시 True) | 동일 코드 | 공통 |

**그 외 차이 없음.** 아래는 두 조건에서 동일하다.

## 동일성 감사 (지시서 §12)

| 항목 | 값 | 동일? |
|---|---|---|
| ACT 아키텍처 | d_model 512, nhead 8, enc 4, dec 7, ffn 3200, latent 32, chunk 8 | ✓ |
| backbone | ResNet18 + FrozenBatchNorm2d, ImageNet 사전학습, avgpool 제거 | ✓ |
| **RGB 정규화** | ImageNet mean/std — **두 조건 동일** | ✓ |
| **depth 정규화** | [0,1] 유지, **별도 채널로만 부착** → RGB 통계에 영향 없음 | ✓ (순수 ablation) |
| action/proprio 정규화 | `compute_stats`의 mean/std — 이미지와 무관, 두 조건 동일 | ✓ |
| optimizer | AdamW, lr 1e-5, lr_backbone 1e-5, weight_decay 1e-4 | ✓ |
| batch size | 8 | ✓ |
| 학습 스텝 | n별 {10:4000, 20:6000, 40:8000, 80:10000} (warm-start 체인) | ✓ |
| warm-start | n=10→20→40→80 승계 | ✓ |
| 정규화 통계 산출 | max-n 풀에서 1회 산출 후 전 단계 동결 | ✓ |
| **증강** | **미사용** (코드에 augmentation 없음) | ✓ |
| random seed | HP["seed"] = 0, `torch.manual_seed` + `np.random.seed` | ✓ |
| 이미지 해상도 | 128×128 두 시점(agentview, wrist) | ✓ |
| proprio | `proprio_vector(obs)` 8차원 | ✓ |
| action | 7차원, chunk K=8 | ✓ |
| 평가 스펙 | `heldout_specs(suite, task, 50)` — 두 조건 **동일 uid** | ✓ |

## 파라미터 수 (실측)

| 조건 | 파라미터 | conv1 |
|---|---|---|
| RGB-D | 95,036,360 | (64, 4, 7, 7) |
| RGB-only | 95,030,088 | (64, 3, 7, 7) |
| 차이 | **6,272 (0.0066%)** | depth 채널 × 백본 2개 |

용량 차이가 무시할 수준이므로, 성능 차이를 capacity로 설명할 수 없다.

## 회귀 확인

기존 RGB-D 체크포인트에는 `use_depth` 키가 없다. `HabitPolicy`가 **기본값 True**로 읽으므로
기존 체크포인트·기존 결과의 동작은 변하지 않는다. 실측으로 확인함
(`libero_object_task1/act_n80.pt` → `use_depth=True, in_ch=4` 정상 로드).
