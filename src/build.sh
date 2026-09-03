#!/usr/bin/env bash
# HabitVLA-2 원클릭 빌드.
#   1) third_party(LIBERO, openvla-oft) 핀 커밋 체크아웃 + 로컬 패치
#   2) conda env 2개 구성 (기본 이름 hv2_oft / hv2_hab — envs/setup_envs.sh의 환경변수로 변경 가능)
#   3) 두 env에서 import·벤치마크 로드 검증
#   4) (선택) ORIG=<원본 작업 디렉터리> 를 주면 체크포인트·HDF5·모델 캐시를 심볼릭 링크로 연결
# 사용:  bash build.sh
#        OFT_ENV=hv2r_oft HAB_ENV=hv2r_hab CLONE_OFT_FROM=hv2_oft CLONE_HAB_FROM=hv2_hab bash build.sh
#        ORIG=/home/asmr/workspace/habitvla2 bash build.sh
set -euo pipefail
HABIT2=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HABIT2
bash "$HABIT2/envs/setup_envs.sh"
if [ -n "${ORIG:-}" ]; then bash "$HABIT2/setup/link_local_assets.sh" "$ORIG"; fi
echo "[BUILD-DONE]  다음: bash verify/verify_release.sh (저장 결과와의 일치 검증) 또는 README.md의 실행 예시"
