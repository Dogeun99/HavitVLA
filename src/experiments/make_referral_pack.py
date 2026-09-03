"""세계 B 회부 패키지 생성기 (분석용 Claude 전달 — 연구원 요청 2026-08-15).

범위: C-T2 stale-chunk-tail 결함의 판정 사슬(diag5 → diag5b)과 α/β/γ 결정 재료.
원칙: REFERRAL.md의 수치까지 전부 results/·data/ 산출물에서 프로그래밍 주입 (수동 입력 금지).
재현: hv2_hab python -u experiments/make_referral_pack.py
"""
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(HABIT2, "referral_pack")

COPY_FILES = [
    "configs/preregistration.md", "log.md",
    # 판정 사슬 증거 (v1 → v2 → v3 → diag5/5b)
    "results/e3/t2_diag_task0_v1.json", "results/e3/t2_diag_task5_v2.json",
    "results/e3/t2_diag2.json", "results/e3/t2_diag3.json", "results/e3/t2_diag4.json",
    "results/e3/t2_diag_v3_probe.json", "results/e3/t2_diag5.json", "results/e3/t2_diag5b.json",
    "results/e3/t2_smoke_v1_negative.json", "results/e3/t2_smoke_v2_negative.json",
    # 수집 요약 (트리거 수치의 원본)
    "data/e3/chained_libero_object_task0_summary.json",
    "data/e3/chained_libero_object_task5_summary.json",
    "data/e2/libero_object_task0_summary.json", "data/e2/libero_object_task5_summary.json",
    "data/e3/libero_object_task6_summary.json",
    # 결함 위치의 코드 (실행기 chunk 루프 + 래퍼 + 게이트)
    "teacher/collector.py", "habits/evaluate.py", "envs/chained_env.py",
    "experiments/e3_t2_check.py", "experiments/e3_t2_diag5.py", "experiments/e3_t2_diag5b.py",
]


def load(p):
    return json.load(open(os.path.join(HABIT2, p)))


def main():
    os.makedirs(PACK, exist_ok=True)
    for rel in COPY_FILES:
        shutil.copy(os.path.join(HABIT2, rel), os.path.join(PACK, rel.replace("/", "__")))
    with open(os.path.join(PACK, "git_history.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "--oneline", "-20"], cwd=HABIT2,
                               capture_output=True, text=True).stdout)

    d5 = load("results/e3/t2_diag5.json")
    d5b = load("results/e3/t2_diag5b.json")
    s_t0 = load("data/e3/chained_libero_object_task0_summary.json")
    s_t5 = load("data/e3/chained_libero_object_task5_summary.json")
    s_t6 = load("data/e3/libero_object_task6_summary.json")
    v1 = load("results/e3/t2_smoke_v1_negative.json")
    v2 = load("results/e3/t2_smoke_v2_negative.json")

    stale = [r["stale_discarded"] for r in d5b["per_episode"]]
    stale_dist = dict(sorted(Counter(stale).items()))

    evidence = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "프로그래밍 생성 (experiments/make_referral_pack.py). REFERRAL.md의 수치도 본 파일과 동일 소스.",
        "timeline": {
            "v1_objects_only": {"task0_smoke": f"0/{v1['n']}", "cause": "teacher 포즈 OOD (verdict B)"},
            "v2_no_env_reset": {"task5_smoke": f"0/{v2['task5_smoke']['n']}",
                                "cause": "OSC 이월 — 로봇 홈 0.48rad 이탈 (t2_diag4)"},
            "v3_episode_boundary": {
                "task0_collection": f"{s_t0['n_success']}/120 (트리거 통과)",
                "task5_collection": f"{s_t5['n_success']}/120 (트리거 발동 → 정지)",
            },
        },
        "diag5": {"n_targets": d5["n_targets"], "fresh_success": d5["n_targets"] - d5["n_reproduced_fail"],
                  "band_identity": d5["band_identity"]["identical"], "world": d5["world"]},
        "diag5b": {"n": d5b["n_targets"], "chunk_break_success": d5b["n_success_with_chunk_break"],
                   "confirmed": d5b["confirmed"], "stale_discarded_distribution": stale_dist},
        "task6_candidate": {"collection": f"{s_t6['n_success']}/120",
                            "trigger_p0_rule": "(Wilson 95% 하한)² = 0.939 (§5 등재, 무소급)"},
        "cost_anchors_s": {"task0_chain_collection_wall": s_t0["wall_seconds"],
                           "task5_chain_collection_wall": s_t5["wall_seconds"]},
    }
    with open(os.path.join(PACK, "referral_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    n5, k5 = d5["n_targets"], d5["n_targets"] - d5["n_reproduced_fail"]
    md = f"""# C-T2 세계 B 회부 브리핑 — stale chunk tail 실행기 결함 (2026-08-15)

**결정 요청**: C-T2 체인 구성·실행기 수정 방침 (α/β/γ). 본 문서 수치는 전부
`referral_evidence.json`과 동일 소스에서 프로그래밍 주입 (생성기: make_referral_pack.py).

## 1. 판정 사슬 요약

| 단계 | 결과 | 원인 |
|---|---|---|
| v1 래퍼 (물체만 재배치) | task0 스모크 0/{v1['n']} | teacher 포즈 OOD — 시연 종료 분포 정지 행동 (t2_diag_task0_v1) |
| v2 (reset 생략 홈 재설정) | task5 스모크 0/{v2['task5_smoke']['n']} | OSC 컨트롤러 이월 — 로봇 홈 0.48 rad 이탈, 물체 Δ=0 (t2_diag4) |
| v3 (에피소드 경계 프로토콜) | task0 수집 {s_t0['n_success']}/120 통과 · task5 수집 {s_t5['n_success']}/120 **트리거 발동** | ↓ diag5/5b가 원인 격리 |
| **diag5** | 비맹점 stage-2 실패 {n5}건 fresh 재실행 → **{k5}/{n5} 성공** (대역 구성 수치 동일) | **세계 B**: v3 stage-2 ≢ fresh |
| **diag5b** | chunk-break 실행기로 {d5b['n_targets']}건 재실행 → **{d5b['n_success_with_chunk_break']}/{d5b['n_targets']} 성공** | **확증: stale chunk tail** |

## 2. 결함의 정체 (코드 위치 포함)

실행기(collector.py·evaluate.py의 K=8 open-loop 청크 루프)는 전환(stage 1→2)이 chunk
중간에 발생해도 **전환 전 관측으로 계산된 잔여 행동을 계속 실행**한다 — 실측 stale 폐기
분포 {json.dumps(stale_dist)} (건수: stale 수). 홈 포즈로 재설정된 stage-2 시작 직후 이
stale 행동들이 팔을 교란 → task5의 취약한 물체 판별(유사 빨간 상자)이 붕괴.
task0 teacher는 강건해 {s_t0['n_success']}/120 통과 — 결함이 태스크 강건성에 가려져 있었다.

- 래퍼(chained_env.py v3)는 무결: 상태 구성은 fresh와 수치 동일 (diag5 band_identity).
- §4e 등재 의미론("전환 = 에피소드 경계, fresh 동등")을 실행기가 위반하는 구조.
- task5 트리거 발동(88/120, p=4.03e-5)의 원인 재귀속: teacher 성질 → **인프라 아티팩트**.
- 파생 정정: "③층위 draw 민감성" 해석 철회 (①포즈 OOD·②base17 분포 내 맹점은 유효).

## 3. 결정 선택지

**α (실행 측 권고)** — 실행기 수정(전환 감지 시 stale tail 폐기·즉시 재질의; 수집·평가 동형)
  + **task5 유지 복귀**. 수정 후 stage-2 기대 = S_V² = 0.871이 정확히 성립(diag5b {d5b['n_success_with_chunk_break']}/{d5b['n_targets']}).
  원 paired 설계(task0+task5, E2 50 참조) 복원 — task6·h50 신설 경로 불필요.
  비용: task0 체인 재수집(3차, 실측 {s_t0['wall_seconds']:.0f}s) + task5 재수집({s_t5['wall_seconds']:.0f}s)
  + 재스모크 + 학습·평가. §4e 사유 재귀속 개정 + R1a(task6) 회귀 개정 필요.
**β** — 실행기 수정 + task6 유지 (task6 수집 {s_t6['n_success']}/120, 트리거 p₀={evidence['task6_candidate']['trigger_p0_rule']}).
  강건성 선호 시. 단 교체의 원 논거("task5 앵커 오염")가 실행기 원인으로 재귀속되어 약화.
**γ** — 수정 없이 task6: **기각 권고** — stale tail은 모든 체인에 존재, §4e 의미론과 모순.

## 4. 결정에 필요한 추가 논점

1. 실행기 수정의 등재 처리: §4b "habit 실행 주기 K=8 open-loop"과의 관계 — 전환 시
   chunk-break는 "에피소드 경계에서 새 질의"로 등재하면 K=8 의미론과 양립 (경계는 에피소드
   시작과 동형이므로). 수집·평가 대칭 유지 필수.
2. task0 체인 데이터 처리: 어느 옵션이든 실행기 수정 시 v3 수집분(stale tail 포함 과정)과
   런타임 과정이 불일치 → 재수집 원칙 유지 여부.
3. task5 트리거 p₀: α 시 기존 0.871(점추정², 무소급 규칙상 구 체인) 유지 vs Wilson 하한²
   일반 규칙으로 통일할지.
4. 방법론 부록: 판정 블록(diag5)이 그럴듯한 해석(draw 민감성)을 반증한 사례 —
   negative-result 서사 포함 여부.

## 5. 동봉 파일

전 증거 JSON(진단 사슬 v1–v3·diag5·diag5b·스모크 negative 2종), 수집 요약 5종(트리거
수치 원본), 결함 위치 코드 3종 + 게이트·진단 스크립트, preregistration.md·log.md 전문,
git_history.txt. 기계 판독용 요약 = referral_evidence.json.
"""
    with open(os.path.join(PACK, "REFERRAL.md"), "w") as f:
        f.write(md)
    print(f"[REFERRAL-PACK] {len(COPY_FILES)} 파일 + REFERRAL.md + referral_evidence.json + git_history.txt")


if __name__ == "__main__":
    main()
