"""E0-4a: OFT 4 체크포인트를 프로젝트 로컬 HF_HOME으로 다운로드 (전역 캐시 불사용)."""
import os, sys, time, json
HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: script-relative
assert os.environ.get("HF_HOME", "").startswith(HABIT2), "HF_HOME must be project-local"
from huggingface_hub import snapshot_download

SUITES = ["spatial", "object", "goal", "10"]
out = {}
for s in SUITES:
    repo = f"moojink/openvla-7b-oft-finetuned-libero-{s}"
    t0 = time.time()
    p = snapshot_download(repo)
    dt = time.time() - t0
    out[repo] = {"path": p, "seconds": round(dt, 1)}
    print(f"[DL-DONE] {repo} -> {p} ({dt:.0f}s)", flush=True)
json.dump(out, open(os.path.join(HABIT2, "results", "e0", "e0_4a_download.json"), "w"), indent=2)
print("[E0-PASS] item=E0-4a status=PASS json=results/e0/e0_4a_download.json")
