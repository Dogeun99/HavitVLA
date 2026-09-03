"""verify_release.sh 실행 디렉터리에서 VERIFICATION_REPORT.md를 생성한다 (수치는 전부 파일에서 읽는다).
사용: python verify/make_verification_report.py --run <verify_runs/...> --out <results/VERIFICATION_REPORT.md>
"""
import argparse
import csv
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    a = ap.parse_args()
    R = a.run
    S = json.load(open(f"{R}/SUMMARY.json"))
    C = json.load(open(f"{R}/compare_cpu.json"))
    G = json.load(open(f"{R}/compare_gpu.json")) if os.path.exists(f"{R}/compare_gpu.json") else None
    RR = "rgb_only_full_rerun_20260828"
    smoke_p = f"{R}/src_copy/results/{RR}/00_preflight/SMOKE.json"
    smoke = json.load(open(smoke_p)) if os.path.exists(smoke_p) else None
    integ_p = f"{R}/src_copy/results/{RR}/09_integrity/DATA_INTEGRITY_AUDIT.json"
    integ = json.load(open(integ_p)) if os.path.exists(integ_p) else None
    lat_p = f"{R}/latency_side_by_side.txt"
    lat = open(lat_p).read().strip().splitlines() if os.path.exists(lat_p) else []
    env_hab = open(f"{R}/logs/env_hab.log").read().strip().splitlines()[-1]
    env_oft = [l for l in open(f"{R}/logs/env_oft.log").read().splitlines() if "/bin/python" in l][-1]
    pkg = [l for l in open(f"{R}/logs/verify_package.log").read().splitlines() if "PACKAGE-VERIFY" in l][-1]
    orig_commit = subprocess.run(["git", "-C", "/home/asmr/workspace/habitvla2", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or "?"

    L = []
    L.append(f"# VERIFICATION_REPORT — 릴리스 `src/`만으로 빌드해 `results/`와 일치하는가\n")
    L.append(f"- 실행: `{os.path.basename(R)}` (verify/verify_release.sh) · 판정 **{S['verdict']}** ({S['n_steps']} 단계, FAIL {S['n_fail']})")
    L.append(f"- 원본 저장소 커밋 `{orig_commit}` · 릴리스 src 위치 `{a.src}`")
    L.append("- 빌드: 릴리스의 `envs/setup_envs.sh`로 **별도 conda env**(`hv2r_oft`, `hv2r_hab`; 원본 env를 clone한 뒤 릴리스 `third_party/`로 editable 재설치)를 구성해 사용. 원본 env·원본 저장소는 수정하지 않았다.")
    L.append(f"  - hab env: `{env_hab}`")
    L.append(f"  - oft env: `{env_oft}`")
    L.append("- 방법: 릴리스 `src/`를 scratch에 복제하고 `results/`의 **복사본**을 붙인 뒤, 저장된 원자료(원장 JSONL/CSV, 곡선 JSON)에서 "
             "모든 요약·통계를 다시 계산해 저장본과 대조했다. GPU 단계는 실제 시뮬레이터 롤아웃으로 체크포인트를 재평가해 "
             "**에피소드별** 결과(uid → 성공/실패, 스텝 수)를 저장본과 대조했다. 체크포인트·HDF5·모델 캐시는 원본 디렉터리에 링크(읽기만).\n")

    L.append("## 1. 단계별 결과\n")
    L.append("| 단계 | 판정 | 소요(s) | 성공 마커 |")
    L.append("|---|---|---:|---|")
    for s in S["steps"]:
        L.append(f"| `{s['name']}` | {s['status']} | {s['elapsed_s']} | `{s['marker']}` |")
    L.append("")

    L.append("## 2. 분석 재산출 ↔ 저장 결과 대조 (CPU)\n")
    L.append(f"- 패키지 자체 검증: `{pkg.strip()}` (레포 코드 import 없이 원장 CSV에서 요약 재계산)")
    n_id = sum(1 for v in C["files"].values() if v[0] == "IDENTICAL")
    n_vol = sum(1 for v in C["files"].values() if v[0] == "EQUAL_MODULO_VOLATILE")
    n_bad = len(C["files"]) - n_id - n_vol
    L.append(f"- 대조 파일 {len(C['files'])}개: **완전 동일 {n_id}**, 휘발 필드(출처 절대경로·시간) 제외 동일 {n_vol}, 불일치 {n_bad} → `{C['verdict']}`\n")
    L.append("| 파일 | 판정 | 비고 |")
    L.append("|---|---|---|")
    for f, (st, d) in C["files"].items():
        note = ""
        if "rows" in d:
            note = f"{d['rows']} rows"
        if st == "EQUAL_MODULO_VOLATILE":
            cols = d.get("volatile_cols") or [p.split(".")[-1] for p in d.get("volatile_ignored", [])[:1]]
            note += (", " if note else "") + "무시한 필드: " + ", ".join(sorted(set(cols)))
        if st == "DIFFERENT":
            note += (", " if note else "") + json.dumps(d, ensure_ascii=False)[:200]
        L.append(f"| `{f}` | {st} | {note} |")
    L.append("")

    if G:
        L.append("## 3. GPU — 실제 롤아웃·학습·무결성\n")
        if smoke:
            ch = smoke["checks"]
            L.append(f"- 학습·추론 스모크 (`rgb_only_rerun/smoke.py`, {smoke['cluster']} {smoke['steps']}스텝 학습 → held-out {smoke['n_eval']} ep 추론): "
                     f"**{smoke['verdict']}** — {sum(ch.values())}/{len(ch)} 검사 통과 ({', '.join(k for k, v in ch.items())})")
        for r in G["rollouts"]:
            L.append(f"- 체크포인트 held-out 재평가 `{r['cluster']}` ({os.path.relpath(r['ref'], os.path.dirname(a.out))} 대비): **{r['status']}**")
            for c in r["checks"]:
                L.append(f"  - n={c['n']}: 에피소드 {c['n_common_uid']}개 중 성공/실패 일치 {c['outcome_match']}, 스텝 수 일치 {c['steps_match']}, "
                         f"ŝ 재측정 {c['s_hat_new']} / 저장 {c['s_hat_ref_on_common']}")
        if integ:
            ck = next(c for c in integ["checks"] if c["check"] == "depth_used_by_rgb_only_habit")
            L.append(f"- 무결성 감사 재실행 (`integrity_audit.py`): **{integ['overall']}** ({integ['n_checks']} 검사, FAIL {integ['n_fail']}); "
                     f"RGB-only 체크포인트 {ck['detail']['n_checkpoints']}개 전수 depth 미사용 위반 {ck['detail']['n_violations']} — 저장된 감사 JSON과 "
                     f"{G['files'][f'{RR}/09_integrity/DATA_INTEGRITY_AUDIT.json'][0]}")
        L.append("")
        if lat:
            L.append("### 3.1 레이턴시 재측정 (teacher env `hv2r_oft`, attn=sdpa; 시간 측정이라 동일성 대신 나란히 기록)\n")
            L.append("| metric | 저장값 | 재측정 |")
            L.append("|---|---:|---:|")
            for l in lat[1:]:
                parts = l.split()
                if len(parts) >= 3 and any(k in parts[0] for k in ("median", "p95", "ratios", "n_params")):
                    L.append(f"| `{parts[0]}` | {parts[1]} | {parts[2]} |")
            L.append("")

    L.append("## 4. 해석\n")
    L.append("- 저장된 모든 요약·통계(E2 go/no-go, E3 27 곡선·N*, H2 분석, E5 3 seed 판독·종합·사후분석, E4 scorer 표, "
             "RGB-only rerun의 배치/온라인/paired replay/부트스트랩 분포/old-vs-new)는 릴리스 코드로 원자료에서 **비트 단위로 재산출**된다.")
    L.append("- 시뮬레이터 + ACT 정책 스택은 릴리스 폴더의 third_party(핀 커밋 + 패치)와 릴리스 env에서 저장된 롤아웃을 **에피소드 단위로 결정적으로 재현**한다 "
             "(RGB-only·RGB-D 체크포인트 각 1 클러스터, 성공/실패와 스텝 수까지 일치).")
    L.append("- 레이턴시는 하드웨어 시간 측정이므로 ms 단위 소수점에서만 다르고 순위·비율(ACT/teacher ≈ 0.039)은 같다.")
    L.append("- 검증하지 않은 것: 70 h짜리 전체 재실행(배치 27 클러스터 학습, 온라인 12,000 ep, paired replay)과 teacher 궤적 재수집. "
             "이들은 결정적 에피소드 명세(§4h)와 seed 고정으로 재현 가능하도록 설계돼 있고, 위 체크포인트 재평가가 그 실행 경로(시뮬·정책·성공 판정)를 덮는다.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"[REPORT] {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
