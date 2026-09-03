"""관할 특징 추출 — DINOv2 ViT-S/14 임베딩 → PCA(d=32) (설계서 §2.4 2층).

- 입력: 초기 관측 I₀의 agentview RGB (teacher 전처리와 동일하게 180° 회전).
- DINOv2는 gate 경로 전용 경량 인코더 — 주 경로에 VLA forward 없음(설계서 §2.2).
- PCA는 전 클러스터 공용(수집 성공 에피소드 초기 프레임 풀로 적합), d=32 (preregistration §1).
"""
import os
import sys

import numpy as np
import torch

HABIT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(HABIT2, ".hf_cache"))

PCA_DIM = 32  # preregistration §1


GATE_INPUT_RES = 128  # 저장 프레임(수집)과 런타임(256 env obs) 모두 이 해상도로 통일


def prep_gate_rgb(raw_env_rgb):
    """LIBERO 원시 agentview_image → gate 입력 규격 (180° 회전 + 128 리사이즈).
    수집 저장본(store_frame)과 동일 규격 — fit(저장 프레임)/런타임(env obs) 분포 일치 보장."""
    import cv2

    img = raw_env_rgb[::-1, ::-1]
    if img.shape[0] != GATE_INPUT_RES:
        img = cv2.resize(img, (GATE_INPUT_RES, GATE_INPUT_RES), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


class DinoFeatureExtractor:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, rgb_batch):
        """rgb_batch: list of (128,128,3) uint8 — 반드시 prep_gate_rgb 규격.
        해상도 불일치는 즉시 오류 (fit/런타임 분포 불일치 방지 — 검증 발견)."""
        for r in rgb_batch:
            assert r.shape[:2] == (GATE_INPUT_RES, GATE_INPUT_RES), (
                f"gate input must be {GATE_INPUT_RES}px (use prep_gate_rgb), got {r.shape}"
            )
        inputs = self.processor(images=list(rgb_batch), return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0].float().cpu().numpy()


class SharedPCA:
    """전 클러스터 공용 PCA. fit은 **수집 성공 에피소드 초기 프레임 풀에서만** 1회.

    출처 규율: held-out(base 40–49)·novel 프레임을 fit에 혼입하면 평가 누수 — 호출측은
    수집 스트림(base 0–39) 프레임만 전달할 것. calibration 표본이 PCA fit에 포함되는 것은
    split conformal 교환성의 근사 위반으로 문서화된 한계 (엄밀 coverage → 근사 coverage)."""

    def __init__(self, dim=PCA_DIM):
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=dim, random_state=0)
        self.fitted = False

    def fit(self, X):
        self.pca.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        assert self.fitted
        return self.pca.transform(X)

    def save(self, path):
        import joblib

        joblib.dump(self.pca, path)

    @classmethod
    def load(cls, path):
        import joblib

        obj = cls()
        obj.pca = joblib.load(path)
        obj.fitted = True
        return obj
