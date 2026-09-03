"""teacher HDF5 → ACT 학습 데이터셋.

표본 단위 = requery 경계의 (관측, 이후 K per-step 행동) — teacher의 제어 주기와 동일.
정규화: 행동·proprio 모두 클러스터 학습 풀의 mean/std (통계는 체크포인트에 저장, 평가 시 재사용).
n_k 부분집합: 에피소드 순서는 수집 스트림 순서(성공 에피소드의 등장 순) — "경험 축적" 의미론과 일치.
"""
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

CHUNK = 8


def load_cluster(h5_path, n_episodes=None):
    """HDF5(schema v2) → 에피소드 리스트(수집 순서). n_episodes로 앞에서부터 절단 (성숙 곡선 n_k)."""
    episodes = []
    with h5py.File(h5_path, "r") as f:
        meta = json.loads(f["meta_json"][()])
        order = [m["uid"] for m in meta if m["outcome"] == "success"]
        if n_episodes is not None:
            order = order[:n_episodes]
        for uid in order:
            g = f[f"episodes/{uid}"]
            flat = g["actions_flat"][:]
            lens = g["chunk_lens"][:]
            offsets = np.concatenate([[0], np.cumsum(lens)])
            chunks = [flat[offsets[i] : offsets[i + 1]] for i in range(len(lens))]
            episodes.append(
                {
                    "uid": uid,
                    "agentview_rgb": g["agentview_rgb"][:],
                    "wrist_rgb": g["wrist_rgb"][:],
                    "agentview_depth": g["agentview_depth"][:],
                    "wrist_depth": g["wrist_depth"][:],
                    "proprio": g["proprio"][:],
                    "chunks": chunks,
                }
            )
    return episodes


def compute_stats(episodes):
    acts = np.concatenate([c for ep in episodes for c in ep["chunks"]])
    prop = np.concatenate([ep["proprio"] for ep in episodes])
    return {
        "action_mean": acts.mean(0).tolist(),
        "action_std": (acts.std(0) + 1e-6).tolist(),
        "proprio_mean": prop.mean(0).tolist(),
        "proprio_std": (prop.std(0) + 1e-6).tolist(),
    }


# ImageNet 사전학습 백본과 정합 (표준 ACT의 transforms.Normalize — 검증 워크플로우 발견 반영).
# depth 채널은 [0,1] 유지 (conv1 4번째 채널은 RGB 평균 가중치로 초기화되어 스케일 자유).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def make_frame_tensor(rgb, depth, use_depth=True):
    """(H,W,3) uint8 [+ (H,W) f16] → (4,H,W) 또는 (3,H,W) float32.

    RGB 정규화(ImageNet)는 두 조건에서 **동일**하다. depth는 [0,1]을 유지하며 별도 채널로만
    붙으므로, use_depth=False로 두어도 RGB 통계에 영향이 없다 — 순수 modality ablation.
    """
    rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    if not use_depth:
        return rgb
    d = torch.from_numpy(np.ascontiguousarray(depth.astype(np.float32))).unsqueeze(0)
    return torch.cat([rgb, d], dim=0)


class ClusterDataset(Dataset):
    def __init__(self, episodes, stats, use_depth=True):
        self.samples = []
        self.stats = stats
        self.use_depth = use_depth      # False = RGB-only ablation (depth 경로만 제거)
        am = np.array(stats["action_mean"], dtype=np.float32)
        as_ = np.array(stats["action_std"], dtype=np.float32)
        pm = np.array(stats["proprio_mean"], dtype=np.float32)
        ps = np.array(stats["proprio_std"], dtype=np.float32)
        for ep in episodes:
            T = len(ep["chunks"])
            for t in range(T):
                self.samples.append((ep, t))
        self._norm = (am, as_, pm, ps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep, t = self.samples[idx]
        am, as_, pm, ps = self._norm
        ud = getattr(self, "use_depth", True)
        agent = make_frame_tensor(ep["agentview_rgb"][t], ep["agentview_depth"][t], ud)
        wrist = make_frame_tensor(ep["wrist_rgb"][t], ep["wrist_depth"][t], ud)
        proprio = torch.from_numpy((ep["proprio"][t] - pm) / ps)
        chunk = ep["chunks"][t]
        K = CHUNK
        acts = np.zeros((K, chunk.shape[1]), dtype=np.float32)
        pad = np.ones(K, dtype=bool)
        n = min(len(chunk), K)
        acts[:n] = (chunk[:n] - am) / as_
        pad[:n] = False
        return {
            "agentview": agent,
            "wrist": wrist,
            "proprio": proprio.float(),
            "actions": torch.from_numpy(acts),
            "pad_mask": torch.from_numpy(pad),
        }
