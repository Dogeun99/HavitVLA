"""E1 태스크별 S_V 재유도 — (suite, task_id) 키 (통합 지시서 §4).

배경: `e1_sv_collect.py`는 콘솔 로그의 **절단된 태스크명**으로 그룹핑해 libero_spatial의
3개 태스크가 한 행(n=75)으로 병합되는 버그 보유 (검증 발견 — 회귀 사용 금지 판정).
본 스크립트는 공식 eval의 EVAL-*.txt(전체 태스크명 + 에피소드별 Success 라인)를
**순차 파싱**해 태스크 경계를 전체 문자열 전환으로 식별하고, 공식 task order [0..9]에
따라 task_id 키를 부여한다. e1_sv.json은 불변 — 산출은 별도 파일.

산출: results/e1/e1_sv_per_task.json
교차 검증: 스위트 합계가 e1_sv.json의 S_V 분자·분모와 일치해야 PASS.

실행: $HV2_HAB_PY -u experiments/e1_sv_per_task.py
"""
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGDIR = os.path.join(ROOT, "logs", "e1_sv")
OUT = os.path.join(ROOT, "results", "e1", "e1_sv_per_task.json")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TRIALS = 25
N_TASKS = 10


def parse_eval_txt(path):
    """순차 파싱: 'Task: <full>' 전환으로 태스크 경계 식별 → [(full_desc, [succ, ...])]."""
    tasks = []
    cur_desc, cur = None, []
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("Task: "):
            desc = line[len("Task: "):].strip()
            if desc != cur_desc:
                if cur_desc is not None:
                    tasks.append((cur_desc, cur))
                cur_desc, cur = desc, []
        elif line.startswith("Success: "):
            cur.append(line == "Success: True")
    if cur_desc is not None:
        tasks.append((cur_desc, cur))
    return tasks


def main():
    ref = json.load(open(os.path.join(ROOT, "results", "e1", "e1_sv.json")))
    out = {
        "note": "(suite, task_id) 키 — EVAL-*.txt 순차 파싱, task_id = 공식 task order 순번. "
        "e1_sv.json spatial per_task의 절단명 병합 버그 회피본 (통합 지시서 §4).",
        "suites": {},
        "status": "FAIL",
    }
    for suite in SUITES:
        paths = glob.glob(os.path.join(LOGDIR, f"EVAL-{suite}-openvla-*.txt"))
        assert len(paths) == 1, f"{suite}: EVAL txt {len(paths)}개 (1개 기대)"
        tasks = parse_eval_txt(paths[0])
        # 완결성: 10 태스크 × 25 ep — 미달 시 무조건 FAIL (부분 로그 PASS 금지)
        assert len(tasks) == N_TASKS, f"{suite}: 태스크 그룹 {len(tasks)} != {N_TASKS}"
        rows = {}
        for tid, (desc, eps) in enumerate(tasks):
            assert len(eps) == TRIALS, f"{suite} task{tid}: {len(eps)} ep != {TRIALS}"
            rows[str(tid)] = {"task": desc, "n": len(eps), "k": sum(eps),
                              "rate": round(sum(eps) / len(eps), 4)}
        # 교차 검증: 스위트 합계 = e1_sv.json S_V
        k_sum = sum(r["k"] for r in rows.values())
        n_sum = sum(r["n"] for r in rows.values())
        ref_sv = ref["suites"][suite]["S_V"]
        assert abs(k_sum / n_sum - ref_sv) < 1e-6, (
            f"{suite}: 재유도 {k_sum}/{n_sum} != e1_sv.json S_V {ref_sv}")
        out["suites"][suite] = rows
        print(f"[{suite}] {k_sum}/{n_sum} = {k_sum/n_sum:.4f} (e1_sv.json 일치)", flush=True)

    # 알려진 앵커 재확인: libero_10 task8 = moka pots 21/25
    assert out["suites"]["libero_10"]["8"]["k"] == 21, "libero_10 task8 앵커 불일치"
    out["status"] = "PASS"
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E1-PERTASK-PASS] -> {OUT}")


if __name__ == "__main__":
    main()
