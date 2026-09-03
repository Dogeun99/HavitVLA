#!/usr/bin/env bash
# 대용량 로컬 자산(체크포인트·궤적 HDF5·모델 캐시)을 원본 작업 디렉터리에서 심볼릭 링크로 연결한다.
# 릴리스 저장소에는 이 자산들이 들어 있지 않다 (checkpoints 192 GB, data 15 GB, .hf_cache 60 GB).
#
#   사용: setup/link_local_assets.sh [ORIGINAL_ROOT]   (기본 /home/asmr/workspace/habitvla2)
#
# 규칙: checkpoints/·data/는 **하위 디렉터리 단위**로 링크한다 — 새 클러스터/새 run은 릴리스 쪽에
# 새로 생기고, 기존 것은 원본을 가리킨다. checkpoints/rgb_only_rerun/{batch,online}도 같은 규칙.
# .hf_cache·.torch_cache는 통째로 링크한다 (읽기 전용 캐시).
set -euo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ORIG=${1:-/home/asmr/workspace/habitvla2}
[ -d "$ORIG" ] || { echo "[LINK-FAIL] original root not found: $ORIG"; exit 1; }

link_children() {  # link_children <orig_dir> <dst_dir>
  local src=$1 dst=$2
  mkdir -p "$dst"
  for d in "$src"/*; do
    [ -e "$d" ] || continue
    local name; name=$(basename "$d")
    [ -e "$dst/$name" ] || ln -s "$d" "$dst/$name"
  done
}
# checkpoints: 1단계 하위 링크, rgb_only_rerun과 rgb_only_ablation은 2단계까지
mkdir -p "$HABIT2/checkpoints" "$HABIT2/data"
for d in "$ORIG"/checkpoints/*; do
  name=$(basename "$d")
  case "$name" in
    rgb_only_rerun|rgb_only_ablation) link_children "$d" "$HABIT2/checkpoints/$name" ;;
    *) [ -e "$HABIT2/checkpoints/$name" ] || ln -s "$d" "$HABIT2/checkpoints/$name" ;;
  esac
done
link_children "$ORIG/data" "$HABIT2/data"
for c in .hf_cache .torch_cache; do
  [ -e "$HABIT2/$c" ] || ln -s "$ORIG/$c" "$HABIT2/$c"
done
echo "[LINK-DONE] checkpoints=$(ls "$HABIT2/checkpoints" | wc -l) entries, data=$(ls "$HABIT2/data" | wc -l) entries, caches linked"
