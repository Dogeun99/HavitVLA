"""릴리스 검증 — 재산출된 결과 파일을 저장된(참조) 결과와 대조한다.

사용:
  python verify/compare_outputs.py --ref <참조 results 루트> --new <재산출 results 루트> \
      --files a.json b.csv ... [--rollout new_curve.json ref_curve.json] [--out report.json]

규칙
  * JSON: 깊은 비교. float은 |a-b| <= 1e-9. 휘발 키(시간·경로·git 등, VOLATILE_KEYS)는 값이
    달라도 무시하되 무시한 키를 보고한다.
  * CSV: 행 수·열 집합·셀 값 전수 비교 (휘발 열 제외).
  * NPY: numpy.array_equal.
  * --rollout: evaluate.py 산출 curve.json의 per_episode(uid → outcome, steps)를 참조 curve.json과 대조.
판정: 파일마다 IDENTICAL / EQUAL_MODULO_VOLATILE / DIFFERENT / MISSING. 마지막 줄 [COMPARE-PASS|FAIL].
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

VOLATILE_KEYS = {
    "wall_seconds", "wall_s", "wall_min", "elapsed_s", "elapsed", "generated_at", "generated",
    "timestamp", "time", "date", "start", "end", "done_at", "created", "run_at", "finished_at",
    "git_commit", "git_branch", "git_status_porcelain", "git_status_clean", "git_diff_stat",
    "git_diff_full_sha256", "hostname", "python_executable", "platform", "python", "pip_freeze",
    "source", "log", "cmd", "note", "attempt", "pass_marker", "exit_code", "ok", "job",
    "package", "out", "path_abs", "root", "HABIT2", "source_old", "source_new",
}
VOLATILE_PATTERNS = re.compile(r"(_at|_time|_sec|_s|_wall|_path|_dir|_root)$")
FTOL = 1e-9


def is_volatile(key):
    return key in VOLATILE_KEYS or bool(VOLATILE_PATTERNS.search(str(key)))


def deep_diff(a, b, path="", diffs=None, ignored=None):
    diffs = [] if diffs is None else diffs
    ignored = [] if ignored is None else ignored
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                (ignored if is_volatile(k) else diffs).append((p, "missing-in-" + ("new" if k not in b else "ref")))
                continue
            if is_volatile(k):
                if a[k] != b[k]:
                    ignored.append((p, "volatile-differs"))
                continue
            deep_diff(a[k], b[k], p, diffs, ignored)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)} != {len(b)}"))
            return diffs, ignored
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", diffs, ignored)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
            return diffs, ignored
        if abs(float(a) - float(b)) > FTOL:
            diffs.append((path, f"{a} != {b}"))
    else:
        if a != b:
            diffs.append((path, f"{a!r} != {b!r}"))
    return diffs, ignored


def cmp_json(ref, new):
    a, b = json.load(open(ref)), json.load(open(new))
    diffs, ignored = deep_diff(a, b)
    if diffs:
        return "DIFFERENT", {"n_diff": len(diffs), "examples": diffs[:8], "n_volatile_ignored": len(ignored)}
    if ignored:
        return "EQUAL_MODULO_VOLATILE", {"volatile_ignored": [p for p, _ in ignored][:12], "n_volatile_ignored": len(ignored)}
    return "IDENTICAL", {}


def cmp_csv(ref, new):
    A = list(csv.DictReader(open(ref, newline="")))
    B = list(csv.DictReader(open(new, newline="")))
    if len(A) != len(B):
        return "DIFFERENT", {"rows_ref": len(A), "rows_new": len(B)}
    cols_a = set(A[0].keys()) if A else set()
    cols_b = set(B[0].keys()) if B else set()
    if cols_a != cols_b:
        return "DIFFERENT", {"cols_only_ref": sorted(cols_a - cols_b), "cols_only_new": sorted(cols_b - cols_a)}
    vol = {c for c in cols_a if is_volatile(c)}
    ncell = nbad = nvol = 0
    ex = []
    for i, (ra, rb) in enumerate(zip(A, B)):
        for c in cols_a:
            va, vb = ra[c], rb[c]
            if va == vb:
                continue
            try:
                if abs(float(va) - float(vb)) <= 1e-6:
                    continue
            except ValueError:
                pass
            if c in vol:
                nvol += 1
                continue
            nbad += 1
            if len(ex) < 8:
                ex.append((i, c, va, vb))
        ncell += len(cols_a)
    if nbad:
        return "DIFFERENT", {"rows": len(A), "cells_differ": nbad, "examples": ex}
    if nvol:
        return "EQUAL_MODULO_VOLATILE", {"rows": len(A), "volatile_cells_differ": nvol, "volatile_cols": sorted(vol)}
    return "IDENTICAL", {"rows": len(A), "cells": ncell}


def cmp_npy(ref, new):
    a, b = np.load(ref), np.load(new)
    if a.shape != b.shape:
        return "DIFFERENT", {"shape_ref": a.shape, "shape_new": b.shape}
    if np.array_equal(a, b):
        return "IDENTICAL", {"shape": list(a.shape)}
    return "DIFFERENT", {"max_abs_diff": float(np.max(np.abs(a - b)))}


def cmp_rollout(new_curve, ref_curve):
    """evaluate.py per_episode 대조: 같은 n의 같은 uid → 같은 outcome·steps."""
    N, R = json.load(open(new_curve)), json.load(open(ref_curve))
    ref_by_n = {c["n"]: {e["uid"]: (e["outcome"], e["steps"]) for e in c["per_episode"]} for c in R["curve"]}
    out = {"cluster": N["cluster"], "checks": []}
    ok_all = True
    for c in N["curve"]:
        rb = ref_by_n.get(c["n"], {})
        common = [e for e in c["per_episode"] if e["uid"] in rb]
        same_outcome = sum(1 for e in common if rb[e["uid"]][0] == e["outcome"])
        same_steps = sum(1 for e in common if rb[e["uid"]][1] == e["steps"])
        ref_shat = next((x["s_hat"] for x in R["curve"] if x["n"] == c["n"]), None)
        rec = {"n": c["n"], "n_new": c["n_eval"], "n_common_uid": len(common),
               "outcome_match": same_outcome, "steps_match": same_steps,
               "s_hat_new": c["s_hat"], "s_hat_ref_full": ref_shat,
               "s_hat_ref_on_common": (round(sum(1 for e in common if rb[e["uid"]][0] == "success") / len(common), 4) if common else None)}
        rec["pass"] = bool(common) and same_outcome == len(common)
        ok_all &= rec["pass"]
        out["checks"].append(rec)
    return ("IDENTICAL" if ok_all and all(r["steps_match"] == r["n_common_uid"] for r in out["checks"])
            else "OUTCOMES_MATCH_STEPS_DIFFER" if ok_all else "DIFFERENT"), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--rollout", nargs=2, action="append", default=[], metavar=("NEW_CURVE", "REF_CURVE"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = {"ref": a.ref, "new": a.new, "files": {}, "rollouts": []}
    n_fail = 0
    for rel in a.files:
        r, n = os.path.join(a.ref, rel), os.path.join(a.new, rel)
        if not os.path.exists(r) or not os.path.exists(n):
            report["files"][rel] = ("MISSING", {"ref": os.path.exists(r), "new": os.path.exists(n)}); n_fail += 1
            continue
        try:
            if rel.endswith(".json"):
                st, d = cmp_json(r, n)
            elif rel.endswith(".csv"):
                st, d = cmp_csv(r, n)
            elif rel.endswith(".npy"):
                st, d = cmp_npy(r, n)
            else:
                same = open(r, "rb").read() == open(n, "rb").read()
                st, d = ("IDENTICAL" if same else "DIFFERENT"), {"bytes_ref": os.path.getsize(r), "bytes_new": os.path.getsize(n)}
        except Exception as e:  # noqa: BLE001 — 비교 실패도 FAIL로 기록
            st, d = "ERROR", {"error": repr(e)}
        report["files"][rel] = (st, d)
        if st not in ("IDENTICAL", "EQUAL_MODULO_VOLATILE"):
            n_fail += 1
        print(f"[{st:22s}] {rel}  {json.dumps(d, ensure_ascii=False)[:300]}")
    for new_c, ref_c in a.rollout:
        st, d = cmp_rollout(new_c, ref_c)
        report["rollouts"].append({"new": new_c, "ref": ref_c, "status": st, **d})
        if st == "DIFFERENT":
            n_fail += 1
        print(f"[{st:22s}] rollout {d['cluster']}: " + "; ".join(
            f"n={c['n']} outcome {c['outcome_match']}/{c['n_common_uid']} steps {c['steps_match']}/{c['n_common_uid']} "
            f"s_hat new {c['s_hat_new']} ref(common) {c['s_hat_ref_on_common']}" for c in d["checks"]))
    report["n_fail"] = n_fail
    report["verdict"] = "PASS" if n_fail == 0 else "FAIL"
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"[COMPARE-{report['verdict']}] files={len(a.files)} rollouts={len(a.rollout)} fail={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
