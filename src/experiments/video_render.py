"""영상 지시서 v2 — 렌더러 (CPU, cv2).

V1: 좌 teacher / 우 habit 나란히 + 지시문 배너 + 누적 추론 카운터(chunk당 teacher/habit ms —
    results/e1/e1_latency.json에서 취득, 하드코딩 금지) + 종료 화면 2초(합계·배율·basis).
V2/V3: 단일 패널 실패 재현 + 라벨/MEMO. V4: object_task0 벽시계 재구성(실측 앵커 명시).
index.json: cluster·편·uid·단언·추론 합계·파일 크기 + 앵커 출처.

실행: hv2_hab python -u experiments/video_render.py
"""
import json
import os
import sys

import cv2
import numpy as np

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
RAW = os.path.join(HABIT2, "results", "videos", "_raw")
OUT = os.path.join(HABIT2, "results", "videos")
FPS = 20
SIM_MS = 50.0  # 20Hz 제어 주기
PANEL = 512
BANNER_H, INFO_H = 64, 84
END_FRAMES = 2 * FPS


def load_anchors():
    """카운터 수치의 단일 출처 (검증 렌즈 1: 하드코딩 금지)."""
    lat = json.load(open(os.path.join(HABIT2, "results", "e1", "e1_latency.json")))
    teacher_ms = lat["anchor1_oft_chunk_forward"]["median_ms"]
    habit_ms = lat["anchor2_act_forward"]["median_ms"] + lat["anchor3_gate_path"]["median_ms"]
    return teacher_ms, habit_ms


def put(img, text, xy, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def wrap(text, width=64):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines[:2]


def load_npz(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return z["frames"], z["queries"], json.loads(str(z["meta"]))


def banner(canvas, cl, lang):
    put(canvas, cl, (12, 24), 0.55, (180, 220, 255), 1)
    for i, line in enumerate(wrap(f'"{lang}"')):
        put(canvas, line, (12, 44 + 18 * i), 0.42, (255, 255, 255), 1)


def panel_frame(frames, i):
    f = frames[min(i, len(frames) - 1)]
    return cv2.resize(f, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def info_text(canvas, x0, y0, label, step, total_steps, inf_ms, done, color):
    put(canvas, label, (x0, y0), 0.5, color, 1)
    put(canvas, f"step {step:>3}/{total_steps}", (x0, y0 + 22), 0.45)
    put(canvas, f"inference {inf_ms/1000:6.2f}s", (x0, y0 + 44), 0.45, (120, 255, 120))
    if done:
        put(canvas, "DONE", (x0 + 200, y0), 0.5, (120, 255, 120), 2)


def writer(path, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))


def stage_badge(canvas, x0, frame_idx, stage1_end):
    """chained: stage 2 진입 마커 (지시서 — STAGE 2 진입 마커 유지)."""
    if stage1_end is not None and frame_idx >= stage1_end:
        put(canvas, "STAGE 2", (x0 + PANEL - 110, BANNER_H + 26), 0.55, (80, 200, 255), 2)


def render_v1(cl, lang, t_data, h_data, t_ms, h_ms, out_path):
    tf, tq, tm = t_data
    hf, hq, hm = h_data
    W, H = 2 * PANEL, BANNER_H + PANEL + INFO_H
    vw = writer(out_path, W, H)
    T = max(len(tf), len(hf))
    for i in range(T):
        canvas = np.zeros((H, W, 3), np.uint8)
        banner(canvas, cl, lang)
        canvas[BANNER_H:BANNER_H + PANEL, :PANEL] = panel_frame(tf, i)[:, :, ::-1]
        canvas[BANNER_H:BANNER_H + PANEL, PANEL:] = panel_frame(hf, i)[:, :, ::-1]
        stage_badge(canvas, 0, min(i, len(tf) - 1), tm.get("stage1_end"))
        stage_badge(canvas, PANEL, min(i, len(hf) - 1), hm.get("stage1_end"))
        y = BANNER_H + PANEL + 22
        ti, hi = min(i, len(tf) - 1), min(i, len(hf) - 1)
        info_text(canvas, 12, y, "Teacher: OFT 7B", ti, tm["steps"], tq[ti] * t_ms,
                  i >= len(tf) - 1, (120, 180, 255))
        info_text(canvas, PANEL + 12, y, "Habit: ACT 95M", hi, hm["steps"], hq[hi] * h_ms,
                  i >= len(hf) - 1, (140, 255, 180))
        vw.write(canvas)
    t_tot, h_tot = tm["n_queries"] * t_ms / 1000, hm["n_queries"] * h_ms / 1000
    for _ in range(END_FRAMES):
        canvas = np.zeros((H, W, 3), np.uint8)
        put(canvas, cl, (12, 30), 0.6, (180, 220, 255))
        put(canvas, f"Inference: {t_tot:.2f}s (teacher) vs {h_tot:.2f}s (habit)  -- {t_tot/h_tot:.1f}x",
            (12, H // 2 - 10), 0.85, (255, 255, 255), 2)
        put(canvas, f"basis: per-chunk, gate counted every chunk (conservative floor); "
                    f"anchors {t_ms:.2f}ms vs {h_ms:.2f}ms (e1_latency.json, attn=sdpa)",
            (12, H // 2 + 24), 0.42, (200, 200, 200))
        vw.write(canvas)
    vw.release()
    return {"teacher_inference_s": round(t_tot, 3), "habit_inference_s": round(h_tot, 3),
            "ratio": round(t_tot / h_tot, 1)}


def render_single(cl, lang, data, ms, label, sub, out_path, memo=None, color=(255, 160, 120)):
    fr, q, m = data
    W, H = PANEL, BANNER_H + PANEL + INFO_H
    vw = writer(out_path, W, H)
    for i in range(len(fr)):
        canvas = np.zeros((H, W, 3), np.uint8)
        banner(canvas, cl, lang)
        canvas[BANNER_H:BANNER_H + PANEL, :] = panel_frame(fr, i)[:, :, ::-1]
        stage_badge(canvas, 0, i, m.get("stage1_end"))
        y = BANNER_H + PANEL + 22
        put(canvas, label, (12, y), 0.5, color, 2)
        put(canvas, sub, (12, y + 20), 0.4, color)
        put(canvas, f"step {min(i, len(fr)-1):>3}/{m['steps']}  inference {q[min(i, len(fr)-1)]*ms/1000:5.2f}s",
            (12, y + 42), 0.45)
        if memo:
            put(canvas, memo, (12, y + 62), 0.38, (120, 200, 255))
        vw.write(canvas)
    vw.release()
    return {"inference_s": round(m["n_queries"] * ms / 1000, 3)}


def render_v4(cl, lang, t_data, h_data, t_ms, h_ms, out_path):
    """벽시계 재구성: sim 50ms/frame + 쿼리 경계마다 실측 추론 지연만큼 정지 프레임 삽입."""
    def timeline(frames, queries, ms):
        idx, wall_ms, acc = [], [], 0.0
        for i in range(len(frames)):
            if i > 0 and queries[i] > queries[i - 1]:
                acc += ms
                while acc >= SIM_MS:
                    idx.append(i - 1)          # 추론 대기 = 직전 프레임 정지
                    wall_ms.append(None)
                    acc -= SIM_MS
            idx.append(i)
            wall_ms.append(None)
        return idx

    tf, tq, tm = t_data
    hf, hq, hm = h_data
    ti_l, hi_l = timeline(tf, tq, t_ms), timeline(hf, hq, h_ms)
    T = max(len(ti_l), len(hi_l))
    W, H = 2 * PANEL, BANNER_H + PANEL + INFO_H
    vw = writer(out_path, W, H)
    for i in range(T):
        canvas = np.zeros((H, W, 3), np.uint8)
        banner(canvas, cl + "  [WALL-CLOCK MODE]", lang)
        a = ti_l[min(i, len(ti_l) - 1)]
        b = hi_l[min(i, len(hi_l) - 1)]
        canvas[BANNER_H:BANNER_H + PANEL, :PANEL] = panel_frame(tf, a)[:, :, ::-1]
        canvas[BANNER_H:BANNER_H + PANEL, PANEL:] = panel_frame(hf, b)[:, :, ::-1]
        y = BANNER_H + PANEL + 22
        put(canvas, f"Teacher wall {(i if i < len(ti_l) else len(ti_l)) * SIM_MS / 1000:5.2f}s",
            (12, y), 0.5, (120, 180, 255))
        put(canvas, f"Habit wall {(i if i < len(hi_l) else len(hi_l)) * SIM_MS / 1000:5.2f}s",
            (PANEL + 12, y), 0.5, (140, 255, 180))
        if i >= len(hi_l) - 1:
            put(canvas, "DONE", (PANEL + 212, y), 0.5, (140, 255, 180), 2)
        if i >= len(ti_l) - 1:
            put(canvas, "DONE", (212, y), 0.5, (120, 180, 255), 2)
        put(canvas, "reconstructed from measured latency anchors (e1_latency.json, attn=sdpa)"
                    " -- pauses inserted at query boundaries", (12, y + 44), 0.4, (200, 200, 200))
        vw.write(canvas)
    vw.release()


def main():
    t_ms, h_ms = load_anchors()
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    chosen = json.load(open(os.path.join(RAW, "chosen.json")))
    index = {"anchors_source": "results/e1/e1_latency.json",
             "teacher_ms_per_query": t_ms, "habit_ms_per_query": h_ms,
             "basis": "per-chunk; habit = gate(anchor3) + ACT(anchor2) — conservative floor",
             "rollout_failures": chosen.get("failures", []) + chosen.get("habit_failures", []),
             "videos": []}

    for cl, v in man["clusters"].items():
        lang = v["language"]
        cdir = os.path.join(OUT, cl)
        t1 = load_npz(f"{cl}_V1_teacher.npz")
        h1 = load_npz(f"{cl}_V1_habit.npz")
        if t1 and h1:
            assert t1[2]["uid"] == h1[2]["uid"] == chosen["chosen_v1"][cl], f"{cl}: uid 불일치"
            p = os.path.join(cdir, f"{cl}_V1.mp4")
            stats = render_v1(cl, lang, t1, h1, t_ms, h_ms, p)
            index["videos"].append({"cluster": cl, "kind": "V1", "uid": t1[2]["uid"],
                                    "assertion": "both-success", **stats,
                                    "bytes": os.path.getsize(p)})
        h2 = load_npz(f"{cl}_V2_habit.npz")
        if h2:
            p = os.path.join(cdir, f"{cl}_V2.mp4")
            stats = render_single(cl, lang, h2, h_ms, "HABIT FAILURE",
                                  "(recorded in held-out eval, n=80)", p)
            index["videos"].append({"cluster": cl, "kind": "V2", "uid": h2[2]["uid"],
                                    "assertion": "failure-reproduced", **stats,
                                    "bytes": os.path.getsize(p)})
        t3 = load_npz(f"{cl}_V3_teacher.npz")
        if t3:
            p = os.path.join(cdir, f"{cl}_V3.mp4")
            stats = render_single(cl, lang, t3, t_ms, "TEACHER FAILURE",
                                  "(recorded in collection)", p, memo=t3[2].get("memo"))
            index["videos"].append({"cluster": cl, "kind": "V3", "uid": t3[2]["uid"],
                                    "assertion": "failure-reproduced", **stats,
                                    "bytes": os.path.getsize(p)})

    # V4 — 대표 1편 (object_task0)
    t1 = load_npz("libero_object_task0_V1_teacher.npz")
    h1 = load_npz("libero_object_task0_V1_habit.npz")
    if t1 and h1:
        p = os.path.join(OUT, "libero_object_task0", "libero_object_task0_V4_wallclock.mp4")
        render_v4("libero_object_task0", man["clusters"]["libero_object_task0"]["language"],
                  t1, h1, t_ms, h_ms, p)
        index["videos"].append({"cluster": "libero_object_task0", "kind": "V4",
                                "uid": t1[2]["uid"], "assertion": "reconstruction",
                                "bytes": os.path.getsize(p)})

    index["deferred"] = man["deferred"]
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=2, ensure_ascii=False)
    kinds = {}
    for e in index["videos"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    total_mb = sum(e["bytes"] for e in index["videos"]) / 1e6
    print(f"[VIDEO-RENDER-PASS] {kinds} 총 {len(index['videos'])}편 {total_mb:.0f}MB -> results/videos/")


if __name__ == "__main__":
    main()
