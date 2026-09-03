# RUN_COMMANDS — RGB-D vs RGB-only 스크리닝 (지시서 §15)

모든 명령을 실행 순서대로 기록한다. 기존 결과는 별도 경로로 분리되어 수정되지 않는다.

## 0. 산출 경로 (기존 결과와 분리)

```
experiments/rgb_depth_ablation/     스크립트·문서
results/rgb_depth_ablation/         평가·분석 산출
checkpoints/rgb_only_ablation/      RGB-only 체크포인트
logs/rgb_depth_ablation/            실행 로그
```

기존 `checkpoints/<cluster>/`, `results/e2|e3/`, `data/`는 **읽기만** 한다.

## 1. 클러스터 선정 (결과 산출 전)

`ABLA_RGBD_CLUSTER_SELECTION.md` 작성 — 기계적 규칙 고정 후 적용.

## 2. RGB-only 학습 (6 클러스터)

RGB-D는 기존 체크포인트를 재사용하므로 학습하지 않는다(동일 split·seed·HP 확인, `CONFIG_DIFF.md`).

```bash
python habits/train.py --h5 data/e3/<cluster>.hdf5 --cluster <cluster> \
    --n-grid 10 20 40 80 --out checkpoints/rgb_only_ablation --no-depth
```

`libero_object_task0`만 `data/e2/`에 있다(E2 파일럿 클러스터).

## 3. 평가 (두 조건 × 6 클러스터, 동일 held-out 50)

```bash
# RGB-D (기존 체크포인트)
python habits/evaluate.py --suite <suite> --task <task> --cluster <cluster> \
    --ckpt-dir checkpoints/<cluster> --n-heldout 50 --out results/rgb_depth_ablation
# → <cluster>_rgbd_h50.json

# RGB-only
python habits/evaluate.py --suite <suite> --task <task> --cluster <cluster> \
    --ckpt-dir checkpoints/rgb_only_ablation/<cluster> --n-heldout 50 --out results/rgb_depth_ablation
# → <cluster>_rgb_h50.json
```

`heldout_specs(suite, task, 50)`은 결정적이므로 두 조건이 **동일 uid**를 평가한다 → 에피소드 단위 paired.

## 4. 분석·그림·표

```bash
python experiments/rgb_depth_ablation/analyze_ablation.py
```

산출: `ablation_summary.json` · `table_detail.csv` · `table_cluster.md` ·
`fig_A_curves.png` · `fig_B_delta.png` · `fig_C_nstar.png`

## 5. 일괄 실행

```bash
bash experiments/rgb_depth_ablation/run_ablation.sh     # 2·3단계 (재실행 시 완료분 건너뜀)
```

## 실행 이력

| 시각 | 내용 |
|---|---|
| 2026-08-28 19:30 | ablation 착수 (tmux `abla`) |
| 19:30–20:3x | RGB-only 6 클러스터 학습 (약 10.5분/클러스터, 44.4 steps/s) |
| 20:3x– | 평가 12건 (2,400 에피소드) |

### 착수 시 실행 환경 문제 2건 (실험 설계와 무관)

1. `tee` 대상 로그 디렉토리가 스크립트 내부에서 생성되어 첫 기동 실패 → 사전 생성으로 해결.
2. `ls "a" "b"`가 한쪽 미존재 시 exit 2를 내고 `set -o pipefail`이 전파해 스크립트 종료
   → 경로 후보를 `[ -f ]` 루프로 교체.

둘 다 학습·평가 설정에는 영향이 없으며, 발생 시점에 산출물이 없었다.
