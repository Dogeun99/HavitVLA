"""학습된 ACT 체크포인트 → 실행 정책 (held-out 평가·E5 발화 공용).

입력 경로는 collector의 store_frame과 동일 전처리(180° 회전 + 128 리사이즈 + [0,1])를
LiberoEpisodeEnv의 원시 obs에서 재현한다 — 학습/실행 분포 일치.
"""
import os
import sys

import numpy as np
import torch

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HABIT2)
os.environ.setdefault("TORCH_HOME", os.path.join(HABIT2, ".torch_cache"))  # 공용 캐시 불침범

from habits.act import ACTPolicy  # noqa: E402
from habits.dataset import make_frame_tensor  # noqa: E402

STORE_RES = 128


def _prep_rgb(img):
    import cv2

    img = img[::-1, ::-1]
    return cv2.resize(img, (STORE_RES, STORE_RES), interpolation=cv2.INTER_AREA)


def _prep_depth(d):
    import cv2

    d = d[::-1, ::-1, 0]
    return cv2.resize(d, (STORE_RES, STORE_RES), interpolation=cv2.INTER_AREA)


class HabitPolicy:
    def __init__(self, ckpt_path, device="cuda"):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # pretrained=False: 체크포인트로 즉시 덮어쓰므로 ImageNet 가중치 다운로드 불필요 (검증 발견)
        # 체크포인트가 스스로 modality를 들고 다닌다 — RGB-only ablation 체크포인트를 로드하면
        # 추론 경로도 자동으로 depth를 빼므로, 실행 측이 조건을 기억할 필요가 없다.
        self.use_depth = bool(sd.get("use_depth", True))
        self.model = ACTPolicy(kl_weight=sd["hp"]["kl_weight"], pretrained=False,
                               in_ch=4 if self.use_depth else 3).to(device)
        self.model.load_state_dict(sd["model"])
        self.model.eval()
        self.device = device
        s = sd["stats"]
        self.am = np.array(s["action_mean"], dtype=np.float32)
        self.as_ = np.array(s["action_std"], dtype=np.float32)
        self.pm = np.array(s["proprio_mean"], dtype=np.float32)
        self.ps = np.array(s["proprio_std"], dtype=np.float32)
        self.meta = {k: sd[k] for k in ("n_episodes", "steps", "final_l1", "use_depth", "in_ch",
                                        "n_params") if k in sd}

    @torch.inference_mode()
    def act_chunk(self, obs):
        """LIBERO 원시 obs → K per-step 행동 (비정규화)."""
        from envs.libero_env import proprio_vector  # collector와 동일 proprio 구성 (TF 무의존)

        agent = make_frame_tensor(_prep_rgb(obs["agentview_image"]),
                                  _prep_depth(obs["agentview_depth"]), self.use_depth)
        wrist = make_frame_tensor(_prep_rgb(obs["robot0_eye_in_hand_image"]),
                                  _prep_depth(obs["robot0_eye_in_hand_depth"]), self.use_depth)
        proprio = (proprio_vector(obs) - self.pm) / self.ps

        images = [agent.unsqueeze(0).to(self.device), wrist.unsqueeze(0).to(self.device)]
        p = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        pred = self.model.act(images, p)[0].float().cpu().numpy()  # (K, A) 정규화 공간
        return pred * self.as_ + self.am
