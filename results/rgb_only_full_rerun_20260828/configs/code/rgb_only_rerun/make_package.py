"""§19~§21 최종 데이터 패키지. 논문/그림/표는 만들지 않는다 — 분석 가능한 DATA PACKAGE만.

체크포인트 가중치는 용량 때문에 패키지에 넣지 않고 CHECKPOINT_MANIFEST.csv(path·size·sha256·
seed·cluster·training_round)로 대체한다 (§21 허용).
산출: HabitVLA_RGB_only_full_rerun_<DATE>.tar.gz + 동명 디렉터리
실행: hv2_hab python -u experiments/rgb_only_rerun/make_package.py
마커: [PACKAGE-DONE]
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(HABIT2)
sys.path.insert(0, HABIT2)

from experiments.rgb_only_rerun.runner import PY_HAB, ROOT  # noqa: E402

DATE = "20260828"
PKG = f"HabitVLA_RGB_only_full_rerun_{DATE}"
SEEDS = (0, 1, 2)
CK_ROOTS = ("checkpoints/rgb_only_rerun/batch", "checkpoints/rgb_only_rerun/online")


def seed_dir(s):
    return f"{ROOT}/0{s + 2}_online_seed{s}"


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checkpoint_manifest(dst):
    rows = []
    for pat in (f"{CK_ROOTS[0]}/*/act_n*.pt", f"{CK_ROOTS[1]}/*/*/act_n*.pt"):
        for p in sorted(glob.glob(pat)):
            parts = p.split(os.sep)
            n = int(os.path.basename(p).replace("act_n", "").replace(".pt", ""))
            if "online" in parts:
                seed = parts[parts.index("online") + 1].replace("e5_s", "")
                cluster = parts[-2]
                stage = "online"
            else:
                seed, cluster, stage = "", parts[-2], "batch"
            rows.append({"path": p, "stage": stage, "size_bytes": os.path.getsize(p),
                         "sha256": sha256(p), "seed": seed, "cluster": cluster,
                         "training_round": n})
    if rows:
        with open(f"{dst}/CHECKPOINT_MANIFEST.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


DICT = """# DATA_DICTIONARY

각 파일의 **1행 = 무엇 하나**인지와 열의 의미. 값은 전부 실측이며 파생값은 출처를 표기한다.

## batch_episode_results.csv — 1행 = 배치 held-out 평가 에피소드 1개
| 열 | 의미 |
|---|---|
| cluster_id / suite / task_id | 클러스터 식별 |
| n | 학습에 쓴 teacher 성공 궤적 수 (10/20/40/80) |
| training_seed | ACT 학습 seed (전 클러스터 동일, 동결) |
| eval_episode_id / initial_state_id | EpisodeSpec uid (6원소 해시 — 초기상태 명세의 결정적 식별자) |
| success | 1/0, LIBERO 공식 predicate |
| outcome | success / fail / infra_error (인프라 오류는 성공·실패 어느 쪽에도 산입하지 않음) |
| steps | 소요 스텝 |
| checkpoint_path | 이 행을 만든 체크포인트 |
| n_heldout_protocol | 이 클러스터의 held-out 규모 (20 또는 50 — 기존 프로토콜 그대로) |
| in_e3_view | 1이면 E3 관점(앞 20 스펙) 집계에 포함된 행 |

## batch_summary.csv — 1행 = (클러스터, n)
num_trials·num_success·success_rate·n_infra_error. success_rate = num_success/num_trials.

## NSTAR_RESULTS.csv — 1행 = 클러스터
N_star = ŝ(n) ≥ 0.8을 처음 만족하는 n. 미달이면 ">80"(우측절단, right_censored=1).
formable = N_star가 그리드 안에서 정의됨. wilson_80_* = ŝ(80)의 Wilson 95% 구간.

## batch_statistics.json — §7 통계
decomposition_L(순위 분산 분해·Kruskal–Wallis) · regression_*(순위 OLS + 순열 p, B=10⁴) ·
horizon_T1_vs_T3(단측) · controlled_chain_product_baseline(곱 기준선 이항 + 모수 부트스트랩) ·
intermediate_inputs(재계산용 입력 테이블). 추정량은 기존 e3_* 스크립트 함수를 그대로 import.

## ONLINE_EPISODE_LEDGER_seedX.csv — 1행 = 스트림 에피소드 1개 (seed당 4,000)
| 열 | 의미 |
|---|---|
| seed / episode(t) | seed 인덱스, 스트림 내 0-based 위치 |
| cluster_id / suite / task_id | 클러스터 |
| cold_start | 이 클러스터가 novel 주입 풀(Spatial-b) 소속인가 |
| is_novel_injection | 이 에피소드가 novel 주입분인가 |
| spec_uid / initial_state_id / episode_seed / observation_noise_seed / perturbation_width | 에피소드 명세 6원소 (재현 키) |
| controller / executor | habit 또는 vla |
| decision_reason | fire / immature / unknown_cluster / habit_ineligible / infra |
| state_before / state_after | lifecycle 4상태 U·I·M·X |
| B_k_size (bc_pool) | 그 시점 BC 풀 크기 (teacher 성공 궤적 수) |
| training_triggered / training_round | 재학습 발생 여부와 정책 버전 |
| probe_triggered / probe_success_count / probe_failure_count | 재학습 직후 P=20 probe 결과 |
| sigma_k / phi_k | A_mat 사후 계수 (습관 출처 probe+fire만 산입) |
| tau_k | 그 시점 ACI 임계 |
| habit_fired / habit_success | 발화 여부와 그 결과 (비발화면 null) |
| teacher_used / teacher_success | teacher 실행 여부와 결과 (발화면 null) |
| demotion / rematuration / transition_to_X | 이 에피소드에서 일어난 상태 전이 |
| episode_success | 시스템 관점 성공 (infra_error면 null) |
| VLA_calls / habit_calls | 이 에피소드의 chunk 질의 횟수 |
| episode_latency / wall_s | 에피소드 벽시계 초 |
| wall_clock_time | 에피소드 시작 시각 |
| shadow_jur_* | 그림자 관할 기록 (불개입 — 발화 결정에 쓰이지 않음) |
| aci_* | ACI 누적 상태 |

## LIFECYCLE_EVENTS_LONG.csv — 1행 = 상태 전이 이벤트 1건
event_type ∈ {first_exposure, first_training, retraining, first_maturity, rematuration,
demotion, transition_X}. 에피소드 원장의 전이 플래그와 건수가 일치해야 한다(무결성 검사 항목).

## LIFECYCLE_CLUSTER_SUMMARY.csv — 1행 = (seed, 클러스터)
first_exposure/first_training/second_training/first_maturity/first_X 에피소드 인덱스,
num_firings·num_failures·num_demotions·num_rematurations, final_state, final_B_k_size.

## ONLINE_SUMMARY_seedX.json / ONLINE_SUMMARY_ALL_SEEDS.json — §10
vla_routing_rate(full/first_1000/last_1000/200-ep 창) · system_success · final_lifecycle(M/I/X/U) ·
lifecycle(ever mature·demotion·rematuration·probe 라운드별 통과) · risk(Pr(fail|fire)) ·
cold_start · late_traffic_last1000 · call_accounting. ALL_SEEDS는 완료된 seed의 mean/sd(ddof=1).

## PAIRED_REPLAY_EPISODES.csv — 1행 = 발화 에피소드 1개의 paired 비교
system_success · habit_success · full_vla_success · difference(system − full_vla) + 명세 6원소.

## PAIRED_REPLAY_SUMMARY.json
per_seed에 두 구성이 함께 있다:
  (a) 발화 집합 paired (§11 문면) — n_paired_episodes = 발화 수
  (b) full_stream_noninferiority — 발화분은 CF 재현, 비발화분은 VLA 실측 (논문 H4b와 동일 구성)
bootstrap B=10,000, seed 0. bootstrap_seed{S}.npy / bootstrap_fullstream_seed{S}.npy에
재표집 분포 전체를 저장했으므로 CI를 다시 그릴 수 있다.

## FAMILIARITY_*
DEPENDENCY_AUDIT = 지표별 (habit modality 의존 여부 / 재계산 여부 / 출처).
EPISODES.csv 1행 = 역량 지도 에피소드 1개 (섭동 폭 w, habit 성공, Mahalanobis 점수,
기각 여부, kNN k=5·10 점수, ID/boundary/OOD 라벨). teacher_hidden_score는 재계산 대상이
아니어서 null이며 원 출처는 audit에 기록돼 있다.

## LATENCY_RAW.csv — 1행 = 측정 샘플 1개 (anchor, sample_idx, ms)
anchor ∈ {act_forward_rgb_only, act_forward_rgbd, gate_path, teacher_oft_chunk_forward}.
warmup 10 + 측정 100, cuda.synchronize 경계, attn=sdpa.

## FORMATION_TIMING_RAW.csv — 1행 = 온라인 재학습 이벤트 1건
train_wall_s(학습만) · probe_and_prep_wall_s · formation_event_wall_s(합) ·
probe_success_count/probe_failure_count · n · probe_round.

## OLD_VS_NEW_NUMERIC.csv — 1행 = 지표 1개
metric · seed_or_cluster · old_rgbd · new_rgb · absolute_change · relative_change ·
source_old · source_new. **내부 검증용이며 해석 문장은 없다.**

## DATA_INTEGRITY_AUDIT.json — §14
검사별 status(PASS/FAIL)와 detail. FAIL이 하나라도 있으면 overall=INVALID.

## CHECKPOINT_MANIFEST.csv
가중치는 패키지에 넣지 않았다. path·size_bytes·sha256·seed·cluster·training_round로 대조한다.
"""


def readme(env, integ, status, ck_rows, sizes):
    pf = env.get("source", {})
    e = env.get("environment", {})
    L = [f"# README_RESULTS — {PKG}", "",
         "RGB-only full rerun의 **데이터 패키지**다. 논문 해석문·그림·표는 들어 있지 않다.", "",
         "## 1. run ID", f"- `{os.path.basename(ROOT)}`",
         f"- 패키지: `{PKG}`", "",
         "## 2. git commit",
         f"- commit `{pf.get('git_commit')}`  branch `{pf.get('git_branch')}`",
         f"- working tree clean: {pf.get('git_status_clean')}",
         "- RGB-only 관련 파일 sha256은 `ENVIRONMENT.json`의 `source.rgb_only_relevant_files`", "",
         "## 3. 실행 환경",
         f"- {e.get('gpu_name')} · driver/CUDA: `{e.get('nvidia_smi')}`",
         f"- torch {e.get('torch')} (cuda {e.get('torch_cuda')}) · python {str(e.get('python','')).split()[0]}",
         f"- attention: {e.get('attn_implementation')}",
         "- conda env: ACT=`hv2_hab`, teacher=`hv2_oft`", "",
         "## 4. RGB-only 변경 내용",
         "- **depth 제거 하나뿐이다.** ACT 백본 conv1을 4채널 → 3채널로 좁혔다.",
         "- teacher(OpenVLA-OFT)·teacher 궤적·클러스터 집합·train/eval split·에피소드 명세·",
         "  seed 대역·n grid·재학습 지점·P=20·K=8·optimizer·lr·batch·steps·augmentation·",
         "  RGB 정규화·action 정규화·proprio 표현·게이트 상수(τ·δ·γ·ε·c)는 전부 동결.",
         "- key-by-key 대조는 `CONFIG_DIFF.json` (허용 차이 = depth 관련 키뿐).",
         "- 런타임 depth 미사용 증명은 `RGB_ONLY_INPUT_AUDIT.json`.", ""]
    L += ["## 5. 완료된 experiment", ""]
    for k, v in status.get("stages", {}).items():
        L.append(f"- {k}: **{v}** {status.get('progress', {}).get(k, '')}")
    fails = status.get("errors", [])
    L += ["", "## 6. 실패한 experiment", ""]
    L += [f"- `{f['job']}` exit={f['exit_code']} log=`{f['log']}`" for f in fails] or ["- 없음"]
    L += ["", "## 7~8. 파일 위치와 1행의 단위", "",
          "열 단위 정의는 `DATA_DICTIONARY.md`에 전부 있다. 주요 진입점:", "",
          "| 파일 | 1행 = |", "|---|---|",
          "| `01_batch_formation/batch_episode_results.csv` | 배치 평가 에피소드 |",
          "| `01_batch_formation/NSTAR_RESULTS.csv` | 클러스터 |",
          "| `01_batch_formation/batch_statistics.json` | (통계 묶음) |",
          "| `0X_online_seedS/ONLINE_EPISODE_LEDGER_seedS.csv` | 스트림 에피소드 |",
          "| `derived/LIFECYCLE_EVENTS_LONG.csv` | 상태 전이 이벤트 |",
          "| `derived/LIFECYCLE_CLUSTER_SUMMARY.csv` | (seed, 클러스터) |",
          "| `derived/ONLINE_SUMMARY_ALL_SEEDS.json` | (3-seed 집계) |",
          "| `05_paired_replay/PAIRED_REPLAY_EPISODES.csv` | 발화 에피소드의 paired 비교 |",
          "| `06_familiarity/FAMILIARITY_EPISODES.csv` | 역량 지도 에피소드 |",
          "| `07_latency_cost/LATENCY_RAW.csv` | 레이턴시 측정 샘플 |",
          "| `07_latency_cost/FORMATION_TIMING_RAW.csv` | 재학습 이벤트 |",
          "| `08_statistics/OLD_VS_NEW_NUMERIC.csv` | 대조 지표 |", "",
          "원자료 JSONL(`raw/`)이 모든 CSV의 상위 출처다.", "",
          "## 9. 재현 command", "",
          "```bash",
          "# 전 단계 (marker 기반 resume — 이미 끝난 stage는 건너뛴다)",
          "hv2_hab python -u experiments/rgb_only_rerun/preflight.py",
          "hv2_hab python -u experiments/rgb_only_rerun/smoke.py",
          "hv2_hab python -u experiments/rgb_only_rerun/run_all.py",
          "",
          "# 개별 stage",
          "hv2_hab python -u experiments/rgb_only_rerun/run_batch.py",
          "hv2_oft python -u experiments/e5_driver.py --seed-idx S --n 4000 --no-depth \\",
          "    --out-root <ROOT>/0X_online_seedS --ck-root checkpoints/rgb_only_rerun/online \\",
          "    --data-root data/rgb_only_rerun/online",
          "hv2_oft python -u experiments/e5_counterfactual.py --seed-idx S \\",
          "    --queue-root <ROOT>/0X_online_seedS --out-root <ROOT>/05_paired_replay",
          "```", "",
          "## 10. 무결성 검사 결과", ""]
    if integ:
        L += [f"- **overall = {integ['overall']}** (검사 {integ['n_checks']}건, FAIL {integ['n_fail']}건)"]
        for c in integ["checks"]:
            if c["status"] == "FAIL":
                L.append(f"  - FAIL `{c['check']}`: {json.dumps(c['detail'], ensure_ascii=False)[:200]}")
    else:
        L += ["- `DATA_INTEGRITY_AUDIT.json` 미생성"]
    L += ["", "## 11. old vs new 수치 대조",
          "- `08_statistics/OLD_VS_NEW_NUMERIC.csv` (행마다 source_old·source_new 포함)",
          "- **내부 검증용이다. 해석은 이 패키지를 받는 쪽에서 한다.**", "",
          "## 12. 체크포인트", ""]
    tot = sum(r["size_bytes"] for r in ck_rows) / 2**30 if ck_rows else 0
    L += [f"- 가중치 {len(ck_rows)}개 · {tot:.1f} GB 는 패키지에 넣지 않았다 (§21).",
          "- `CHECKPOINT_MANIFEST.csv`의 path·sha256으로 원본 디렉터리에서 대조한다.",
          f"- 원본 경로: `{CK_ROOTS[0]}/`, `{CK_ROOTS[1]}/`", "",
          "## 13. 이 패키지가 하지 않은 것", "",
          "논문 문장·LaTeX·PDF·Fig.·publication table을 만들지 않았다 (지시 §18·§23).",
          "필요한 것은 이후 환경에서 이 데이터로 전부 재구성할 수 있다.", ""]
    return "\n".join(L)


def main():
    dst = os.path.join(HABIT2, PKG)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    def cp(src, rel):
        if not os.path.exists(src):
            return False
        d = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy(src, d)
        return True

    # ---- preflight / audit
    for f in ("ENVIRONMENT.json", "CONFIG_DIFF.json", "RGB_ONLY_INPUT_AUDIT.json",
              "DATASET_CHECK.json", "PREFLIGHT_STATUS.json", "SMOKE.json"):
        cp(f"{ROOT}/00_preflight/{f}", f"00_preflight/{f}")
    # ---- batch
    for f in ("batch_episode_results.csv", "batch_summary.csv", "NSTAR_RESULTS.csv",
              "batch_statistics.json"):
        cp(f"{ROOT}/01_batch_formation/{f}", f"01_batch_formation/{f}")
    for p in glob.glob(f"{ROOT}/01_batch_formation/curves/*.json"):
        cp(p, f"raw/batch_curves/{os.path.basename(p)}")
    # ---- online
    for s in SEEDS:
        sd = f"0{s + 2}_online_seed{s}"
        for f in (f"ONLINE_EPISODE_LEDGER_seed{s}.csv", f"ONLINE_SUMMARY_seed{s}.json",
                  f"summary_{s}.json"):
            cp(f"{seed_dir(s)}/{f}", f"{sd}/{f}")
        for f in (f"stream_{s}.jsonl", f"lifecycle_events_{s}.jsonl", f"cf_queue_{s}.jsonl"):
            cp(f"{seed_dir(s)}/{f}", f"raw/{f}")
    for f in ("LIFECYCLE_EVENTS_LONG.csv", "LIFECYCLE_CLUSTER_SUMMARY.csv",
              "ONLINE_SUMMARY_ALL_SEEDS.json"):
        cp(f"{ROOT}/derived/{f}", f"derived/{f}")
    # ---- paired / familiarity / latency / stats / integrity
    for p in glob.glob(f"{ROOT}/05_paired_replay/*"):
        if os.path.isfile(p):
            sub = "raw" if p.endswith(".jsonl") else "05_paired_replay"
            cp(p, f"{sub}/{os.path.basename(p)}")
    for p in glob.glob(f"{ROOT}/06_familiarity/*.json") + glob.glob(f"{ROOT}/06_familiarity/*.csv"):
        cp(p, f"06_familiarity/{os.path.basename(p)}")
    for f in ("LATENCY_RAW.csv", "FORMATION_TIMING_RAW.csv", "COMPUTE_SUMMARY.json"):
        cp(f"{ROOT}/07_latency_cost/{f}", f"07_latency_cost/{f}")
    for p in glob.glob(f"{ROOT}/08_statistics/*"):
        cp(p, f"08_statistics/{os.path.basename(p)}")
    cp(f"{ROOT}/09_integrity/DATA_INTEGRITY_AUDIT.json", "09_integrity/DATA_INTEGRITY_AUDIT.json")
    # ---- 운영 기록
    for f in ("WEEKEND_RUN_STATUS.md", "WEEKEND_RUN_STATUS.json", "JOBS_LEDGER.jsonl",
              "FAILED_JOBS.json"):
        cp(f"{ROOT}/{f}", f)
    if not os.path.exists(f"{dst}/FAILED_JOBS.json"):
        json.dump([], open(f"{dst}/FAILED_JOBS.json", "w"))
    for p in glob.glob(f"{ROOT}/logs/*.log"):
        cp(p, f"logs/{os.path.basename(p)}")
    # ---- configs (동결본 + 본 run 스크립트)
    cp("configs/preregistration.md", "configs/preregistration.md")
    cp("CLAUDE.md", "configs/CLAUDE.md")
    for p in glob.glob("experiments/rgb_only_rerun/*.py"):
        cp(p, f"configs/code/rgb_only_rerun/{os.path.basename(p)}")
    for p in ("habits/act.py", "habits/dataset.py", "habits/train.py", "habits/policy.py",
              "habits/evaluate.py", "experiments/e5_driver.py",
              "experiments/e5_counterfactual.py", "experiments/e4r_competence_map.py",
              "envs/stream.py", "envs/libero_env.py", "gates/two_stage.py",
              "experiments/e3_collect.py", "experiments/e3_h2_analysis.py"):
        cp(p, f"configs/code/{p}")

    ck = checkpoint_manifest(dst)
    open(f"{dst}/DATA_DICTIONARY.md", "w").write(DICT)
    env = json.load(open(f"{ROOT}/00_preflight/ENVIRONMENT.json"))
    integ = (json.load(open(f"{ROOT}/09_integrity/DATA_INTEGRITY_AUDIT.json"))
             if os.path.exists(f"{ROOT}/09_integrity/DATA_INTEGRITY_AUDIT.json") else None)
    status = (json.load(open(f"{ROOT}/WEEKEND_RUN_STATUS.json"))
              if os.path.exists(f"{ROOT}/WEEKEND_RUN_STATUS.json") else {})
    sizes = {}
    open(f"{dst}/README_RESULTS.md", "w").write(readme(env, integ, status, ck, sizes))

    subprocess.run(f"tar czf {PKG}.tar.gz {PKG}", shell=True, check=True, cwd=HABIT2)
    nf = sum(len(f) for _, _, f in os.walk(dst))
    mb = os.path.getsize(f"{PKG}.tar.gz") / 1e6
    print(f"[PACKAGE-BUILT] {PKG}.tar.gz · {mb:.1f} MB · {nf} 파일 · 체크포인트 {len(ck)}개(매니페스트)")

    # ---- §22 최종 검증: 새 프로세스에서 패키지만 읽어 숫자 복원
    r = subprocess.run([PY_HAB, "-u", "experiments/rgb_only_rerun/verify_package.py",
                        "--package", dst], cwd=HABIT2)
    if r.returncode != 0:
        print("[PACKAGE-VERIFY-FAIL] 패키지만으로 숫자 복원 실패")
        sys.exit(1)
    print("[PACKAGE-DONE]")


if __name__ == "__main__":
    main()
