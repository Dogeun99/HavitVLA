#!/usr/bin/env bash
# HabitVLA-2 환경 구축 스크립트 (release 판).
#
# 원본(E0-1/E0-2, 2026-08-15 실측 확정 상태, results/e0/e0_1_envs.json)을 그대로 재현한다:
#   $OFT_ENV (기본 hv2_oft): py3.10, torch 2.7.0+cu128, transformers 4.40.1(moojink 포크), TF 2.15,
#             robosuite 1.4.1, mujoco 3.1.6, numpy 1.26.4, opencv 4.9.0.80,
#             LIBERO(pin) editable-compat, openvla-oft(third_party) editable
#   $HAB_ENV (기본 hv2_hab): py3.11, torch 2.7.0+cu128, torchvision 0.22.0, scikit-learn/transformers/einops/
#             matplotlib/pandas/h5py, robosuite 1.4.1, mujoco 3.1.6, numpy 1.26.4,
#             opencv 4.9.0.80, LIBERO(pin) editable-compat
#
# 사후 수동 수정이었던 것들을 전부 포함:
#   ISSUE-9  numpy/opencv 재핀   ISSUE-10 editable compat 모드   ISSUE-11 mujoco 3.1.6 핀
#   ISSUE-12 LIBERO torch.load 패치(git apply)   ISSUE-13 LIBERO_CONFIG_PATH config 생성
#   ISSUE-14 openvla-oft를 third_party 사본에서 설치
#
# 환경변수 (모두 선택):
#   HABIT2          프로젝트 루트 (기본: 이 스크립트의 상위 디렉터리)
#   CONDA           conda 실행 파일 (기본: ~/miniconda3/bin/conda)
#   OFT_ENV/HAB_ENV 생성할 env 이름 (기본: hv2_oft / hv2_hab)
#   CLONE_OFT_FROM  존재하면 이 env를 clone해서 OFT_ENV를 만든다 (기본: vla_oft — 선행 프로젝트의 sm_120 해결 env)
#   CLONE_HAB_FROM  존재하면 이 env를 clone해서 HAB_ENV를 만든다 (기본: 없음 → py3.11 신규 + pip)
#   SKIP_TORCH_INSTALL=1  이미 torch가 있는 env를 재사용할 때 다운로드 생략
#
# 정확한 패키지 버전 전체는 setup/hv2_hab.requirements.lock, setup/hv2_oft.requirements.lock 참조.
set -euo pipefail
HABIT2=${HABIT2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONDA=${CONDA:-$HOME/miniconda3/bin/conda}
OFT_ENV=${OFT_ENV:-hv2_oft}
HAB_ENV=${HAB_ENV:-hv2_hab}
CLONE_OFT_FROM=${CLONE_OFT_FROM:-vla_oft}
CLONE_HAB_FROM=${CLONE_HAB_FROM:-}
LIBERO_PIN=8f1084e3132a39270c3a13ebe37270a43ece2a01
OFT_PIN=e4287e94541f459edc4feabc4e181f537cd569a8
echo "HABIT2=$HABIT2  OFT_ENV=$OFT_ENV  HAB_ENV=$HAB_ENV"

apply_patch() {  # apply_patch <repo> <patch>  — 멱등 (이미 적용돼 있으면 skip)
  local repo=$1 patch=$2
  if git -C "$repo" apply --check "$patch" 2>/dev/null; then
    git -C "$repo" apply "$patch"; echo "$(basename "$patch") applied"
  elif git -C "$repo" apply --check --reverse "$patch" 2>/dev/null; then
    echo "$(basename "$patch") already applied"
  else
    echo "[SETUP-FAIL] $(basename "$patch") neither applies nor is applied"; exit 1
  fi
}

echo "=== [1/7] third_party 준비 (핀 체크아웃 + 로컬 패치) ==="
if [ ! -e "$HABIT2/third_party/LIBERO/setup.py" ]; then
  # git submodule로 받은 경우 비어 있을 수 있다 → 초기화, 아니면 클론
  git -C "$HABIT2" submodule update --init third_party/LIBERO 2>/dev/null || \
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$HABIT2/third_party/LIBERO"
fi
git -C "$HABIT2/third_party/LIBERO" checkout -q $LIBERO_PIN 2>/dev/null || \
  [ "$(git -C "$HABIT2/third_party/LIBERO" rev-parse HEAD)" = "$LIBERO_PIN" ] || { echo "[SETUP-FAIL] LIBERO pin"; exit 1; }
apply_patch "$HABIT2/third_party/LIBERO" "$HABIT2/configs/libero_local.patch"
if [ ! -e "$HABIT2/third_party/openvla-oft/pyproject.toml" ]; then
  git -C "$HABIT2" submodule update --init third_party/openvla-oft 2>/dev/null || \
    git clone https://github.com/moojink/openvla-oft.git "$HABIT2/third_party/openvla-oft"
fi
git -C "$HABIT2/third_party/openvla-oft" checkout -q $OFT_PIN 2>/dev/null || \
  [ "$(git -C "$HABIT2/third_party/openvla-oft" rev-parse HEAD)" = "$OFT_PIN" ] || { echo "[SETUP-FAIL] openvla-oft pin"; exit 1; }
apply_patch "$HABIT2/third_party/openvla-oft" "$HABIT2/configs/openvla_oft_local.patch"   # sm_120 cu128 재핀 포함

echo "=== [2/7] LIBERO 로컬 config 생성 (ISSUE-13: 대화식 프롬프트 회피 + 공용 ~/.libero 오염 방지) ==="
mkdir -p "$HABIT2/.libero" "$HABIT2/.hf_cache" "$HABIT2/.torch_cache" "$HABIT2/logs" "$HABIT2/data" "$HABIT2/checkpoints"
LR="$HABIT2/third_party/LIBERO/libero/libero"
cat > "$HABIT2/.libero/config.yaml" <<EOC
assets: $LR/assets
bddl_files: $LR/bddl_files
benchmark_root: $LR
datasets: $HABIT2/third_party/LIBERO/libero/datasets
init_states: $LR/init_files
EOC

echo "=== [3/7] conda env 생성 ==="
has_env() { $CONDA env list | awk '{print $1}' | grep -qx "$1"; }
if ! has_env "$OFT_ENV"; then
  if [ -n "$CLONE_OFT_FROM" ] && has_env "$CLONE_OFT_FROM"; then
    $CONDA create -n "$OFT_ENV" --clone "$CLONE_OFT_FROM" -y
  else
    $CONDA create -n "$OFT_ENV" python=3.10 -y
    $CONDA run -n "$OFT_ENV" pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
    $CONDA run -n "$OFT_ENV" pip install -e "$HABIT2/third_party/openvla-oft"  # 전체 의존성 (transformers 포크 포함)
  fi
fi
if ! has_env "$HAB_ENV"; then
  if [ -n "$CLONE_HAB_FROM" ] && has_env "$CLONE_HAB_FROM"; then
    $CONDA create -n "$HAB_ENV" --clone "$CLONE_HAB_FROM" -y
  else
    $CONDA create -n "$HAB_ENV" python=3.11 -y
  fi
fi

echo "=== [4/7] $OFT_ENV: LIBERO + 핀 (ISSUE-9/10/11) ==="
$CONDA run -n "$OFT_ENV" pip install -e "$HABIT2/third_party/LIBERO" --config-settings editable_mode=compat
$CONDA run -n "$OFT_ENV" pip install -r "$HABIT2/third_party/openvla-oft/experiments/robot/libero/libero_requirements.txt"
$CONDA run -n "$OFT_ENV" pip install "numpy==1.26.4" "opencv-python==4.9.0.80" "mujoco==3.1.6"
# openvla-oft는 반드시 프로젝트 사본에서 (ISSUE-14)
$CONDA run -n "$OFT_ENV" pip uninstall -q -y openvla-oft 2>/dev/null || true
$CONDA run -n "$OFT_ENV" pip install -e "$HABIT2/third_party/openvla-oft" --no-deps

echo "=== [5/7] $HAB_ENV: torch + ML 스택 ==="
if [ "${SKIP_TORCH_INSTALL:-0}" != "1" ]; then
  $CONDA run -n "$HAB_ENV" python -c "import torch" 2>/dev/null || \
    $CONDA run -n "$HAB_ENV" pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
fi
$CONDA run -n "$HAB_ENV" pip install scikit-learn transformers einops matplotlib pandas h5py scipy joblib

echo "=== [6/7] $HAB_ENV: LIBERO + 핀 ==="
$CONDA run -n "$HAB_ENV" pip install -e "$HABIT2/third_party/LIBERO" --config-settings editable_mode=compat
$CONDA run -n "$HAB_ENV" pip install -r "$HABIT2/third_party/openvla-oft/experiments/robot/libero/libero_requirements.txt"
$CONDA run -n "$HAB_ENV" pip install "numpy==1.26.4" "opencv-python==4.9.0.80" "mujoco==3.1.6"

echo "=== [7/7] 검증 ==="
for e in "$OFT_ENV" "$HAB_ENV"; do
  LIBERO_CONFIG_PATH="$HABIT2/.libero" $CONDA run -n $e python -c "
import numpy, torch, mujoco, robosuite, cv2, libero, os
from libero.libero import benchmark
assert benchmark.get_benchmark_dict()['libero_spatial']().n_tasks == 10
assert os.path.realpath(libero.__path__[0]).startswith(os.path.realpath('$HABIT2/third_party/LIBERO')), libero.__path__
print('$e OK:', 'numpy', numpy.__version__, 'mujoco', mujoco.__version__, 'robosuite', robosuite.__version__, 'libero@', libero.__path__[0])
" || { echo "[SETUP-FAIL] $e verification"; exit 1; }
done
LIBERO_CONFIG_PATH="$HABIT2/.libero" $CONDA run -n "$OFT_ENV" python -c "import prismatic, os; print('$OFT_ENV openvla-oft @', os.path.dirname(prismatic.__file__))"
echo "[E0-SETUP-DONE]"
