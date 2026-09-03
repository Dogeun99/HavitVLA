# 최종 검증 절차

기준 커밋: `4d06731438bf234ffa6ce6588e5e22059222b163 2026-08-21 03:27:42 +0900 E5 complete: all three seeds pass every registered test`

## 0. 이 패키지로 검증할 수 있는 것

| 범위 | 가능 여부 | 방법 |
|---|---|---|
| 판독 재산출 (H4a·H4b·위험) | **가능** | 원자료 + `e5_analyze.py` (아래 1) |
| 3 seed 종합 재산출 | **가능** | `e5_seed_synthesis.py` (아래 2) |
| 그림 재생성 | **가능** | `fig_e5_reading.py` (아래 3) |
| 부적격 사후분석 재산출 | **가능** | `e5_ineligible_postmortem.py` |
| 원자료 무결성 | **가능** | `checksums.sha256` (아래 0-1) |
| 스트림 전체 재실행 | 코드는 포함, **데이터는 재생성 필요** | 아래 4 |

### 0-1. 무결성 확인
```bash
sha256sum -c checksums.sha256
```

## 1. 판독 재산출 (seed별 H4a·H4b)

```bash
# 패키지 루트 = 저장소 구조 미러이므로 스크립트가 그대로 실행된다.
for i in 0 1 2; do
  gunzip -c raw/e5_stream_$i.jsonl.gz > results/e5/stream_$i.jsonl
  gunzip -c raw/e5_cf_$i.jsonl.gz     > results/e5/cf_$i.jsonl
  gunzip -c raw/e5_cf_queue_$i.jsonl.gz > results/e5/cf_queue_$i.jsonl
done
for i in 0 1 2; do python experiments/e5_analyze.py --seed-idx $i; done
```
판독기는 **사전등록에 등재된 규칙만** 집행하며, 그림자 관할 예측치도 하드코딩이 아니라
`prereg/preregistration.md` §5 원문에서 정규식으로 추출한다. 산출된 `reading_{i}.json`을
동봉된 `results/e5/reading_{i}.json`과 비교하면 일치해야 한다.

## 2. 3 seed 종합
```bash
python experiments/e5_seed_synthesis.py
```
기대 출력: `3/3 seed 전 항목 PASS` · H4a 0.874±0.0261 → 0.405±0.0442 · H4b -0.0021±0.0015 · 위험 0.0285±0.0068

## 3. 그림
```bash
for i in 0 1 2; do python experiments/fig_e5_reading.py --seed-idx $i; done
```

## 4. 스트림 재실행 (선택 — 장시간)

체크포인트(93 GB)와 수집 HDF5(8.8 GB)는 용량상 **제외**했다. 다만 모든 에피소드가
`(suite, task, seed, base_idx, w, noise_seed)` 여섯 원소로 **완전히 결정적**이므로
(CF 결정성 사전 검증이 seed마다 5/5 통과) 재실행으로 동일 데이터를 재생성할 수 있다.

```bash
# 환경: conda env 2개 (hv2_oft = OpenVLA-OFT, hv2_hab = ACT/분석)
export HF_HOME=<repo>/.hf_cache   # 공용 캐시 오염 방지
python experiments/e5_driver.py --seed-idx 0        # 약 16.7 h/seed (RTX 5090)
python experiments/e5_counterfactual.py --seed-idx 0  # 약 4 h/seed
```
드라이버는 재학습마다 `assert_retrain_contract()`로 (a) 정규화 stats가 자기 학습
데이터에서 산출됐는지 (b) 스텝이 배치 등가값인지 (c) `|B_k|`가 HDF5와 3중 일치하는지를
검증하고, 위반 시 즉시 정지한다. 본 실행의 통과 기록은 `results/runtime_gate_assertions.txt`.

## 5. 제외 항목과 이유

| 항목 | 크기 | 이유 |
|---|---|---|
| `checkpoints/` | 93 GB | ACT 체크포인트. 재학습으로 재생성 가능(결정적 seed) |
| `data/` | 8.8 GB | 수집·스트림 HDF5(RGB-D 프레임). 재실행으로 재생성 가능 |
| `results/e5/seed0_normstats_invalid/` | — | **무효 실행** — 인용 금지 대상이라 의도적 제외 |
| OpenVLA-OFT 가중치 | ~16 GB | 공개 체크포인트 `moojink/openvla-7b-oft-finetuned-libero-*` |