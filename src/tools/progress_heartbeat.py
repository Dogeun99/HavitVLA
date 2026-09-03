"""장기 작업 진행 heartbeat (연구원 지시 2026-08-16 — 공용, 작업별 plan 주입).

- plan JSON: {"job": str, "phases": [{"name", "units": int|null, "unit_cost_s": float|null,
  "pattern": regex, "count": "per_match"|"max_group"}]}
  * unit_cost_s는 호출측 plan 생성기가 **앵커 파일에서 프로그래밍 취득** (하드코딩 금지).
    앵커 부재 시 null → 균등 가중 + 세션 처리율 ETA만 사용 ("ETA 불확실" 태그).
  * units=null: 총량 미지 phase — phase 단위 % 강등 + "ETA 불확실".
- 동작: --interval(기본 600s)마다 --log 파싱 → 가중 진행률 + 이중 ETA
  (a: 계획 단가 / b: 세션 실측 처리율; 괴리 >20%면 b 우선 + 표기).
- 출력: [PROGRESS] 한 줄을 로그 append + --status 파일 갱신. 30분 무진전 → [PROGRESS-STALL].
- 종료: --until 정규식이 로그에 나타나면 최종 라인 후 종료. ETA 이력은 status 파일 옆
  <status>.history.jsonl에 적재 (캘리브레이션용).

사용: python tools/progress_heartbeat.py --plan p.json --log driver.log \
        --status logs/progress_status.txt --until 'PIPELINE-EXIT' [--interval 600]
"""
import argparse
import json
import re
import time
from datetime import datetime, timedelta


def parse_phase_done(text, ph):
    pat = re.compile(ph["pattern"], re.M)
    if ph.get("count") == "max_group":
        vals = [int(m.group(1)) for m in pat.finditer(text)]
        return max(vals) if vals else 0
    return len(pat.findall(text))


def fmt_td(sec):
    if sec is None:
        return "?"
    sec = max(0, int(sec))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--milestone-pct", type=float, default=0,
                    help="지정 시 그 %마다 [MILESTONE] 마커를 별도 출력 (연구원 보고용, 0=비활성)")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    phases = plan["phases"]
    known_cost = all(p.get("unit_cost_s") for p in phases if p.get("units"))
    total_cost = sum((p["units"] or 0) * (p.get("unit_cost_s") or 1.0) for p in phases)
    t0 = time.time()
    last_done_cost, last_change_t = -1.0, time.time()
    hist_path = args.status + ".history.jsonl"
    next_milestone = args.milestone_pct if args.milestone_pct > 0 else None

    while True:
        time.sleep(args.interval)
        try:
            text = open(args.log, errors="replace").read()
        except FileNotFoundError:
            text = ""
        done_cost, cur_detail, uncertain = 0.0, "", not known_cost
        for ph in phases:
            done = parse_phase_done(text, ph)
            units = ph.get("units")
            if units is None:
                uncertain = True
                if done and not cur_detail:
                    cur_detail = f"{ph['name']} 진행중(총량 미지)"
                continue
            done_c = min(done, units)
            done_cost += done_c * (ph.get("unit_cost_s") or 1.0)
            if done_c < units and not cur_detail:
                cur_detail = f"{ph['name']} {done_c}/{units}"
        pct = 100.0 * done_cost / total_cost if total_cost else 0.0
        elapsed = time.time() - t0
        finished = re.search(args.until, text) is not None
        if finished:
            pct, cur_detail = 100.0, "완료"

        eta_a = (total_cost - done_cost) if known_cost else None
        eta_b = elapsed * (total_cost - done_cost) / done_cost if done_cost > 0 else None
        eta, branch = eta_a, "정상"
        if eta_a and eta_b and abs(eta_a - eta_b) / max(eta_a, eta_b) > 0.2:
            eta, branch = eta_b, f"괴리(계획 {fmt_td(eta_a)}→실측 우선)"
        elif eta_b and not eta_a:
            eta = eta_b
        if uncertain or eta is None:
            branch = "ETA 불확실" if eta is None else branch + "·ETA 불확실"
        done_ts = (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M") if eta else "?"

        line = (f"[PROGRESS] {pct:.0f}% | {cur_detail or '대기'} | 경과 {fmt_td(elapsed)} | "
                f"ETA {fmt_td(eta)} (완료예상 {done_ts}) | 갈래: {branch}")
        if done_cost == last_done_cost and not finished and time.time() - last_change_t > 1800:
            line = f"[PROGRESS-STALL] 30분+ 무진전 | {cur_detail} | 경과 {fmt_td(elapsed)}"
        elif done_cost != last_done_cost:
            last_done_cost, last_change_t = done_cost, time.time()

        with open(args.log, "a") as f:
            f.write(line + "\n")
            # 10%-단위 마일스톤 마커 (연구원 보고 규약) — 경계를 넘긴 만큼 한 번만 찍는다
            if next_milestone is not None and (pct >= next_milestone or finished):
                reached = 100.0 if finished else (int(pct / args.milestone_pct) * args.milestone_pct)
                f.write(f"[MILESTONE] {reached:.0f}% | {cur_detail or '진행'} | 경과 {fmt_td(elapsed)} | "
                        f"ETA {fmt_td(eta)} (완료예상 {done_ts}) | 갈래: {branch}\n")
                next_milestone = None if finished else reached + args.milestone_pct
        with open(args.status, "w") as f:
            f.write(line + "\n")
        with open(hist_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "elapsed_s": round(elapsed), "pct": round(pct, 1),
                                "eta_s": round(eta) if eta else None, "branch": branch}) + "\n")
        if finished:
            break


if __name__ == "__main__":
    main()
