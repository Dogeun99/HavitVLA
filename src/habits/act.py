"""ACT (Action Chunking with Transformers) — 클러스터별 습관 정책 (설계서 §2.3).

표준 ACT(Zhao et al., 2023) 구성: per-camera ResNet18 백본 → 토큰, proprio 토큰,
CVAE 잠재 z 토큰, transformer encoder-decoder, chunk 길이 K 질의 → L1 + β·KL.

본 구현의 결정 사항:
  - 입력 = RGB-D 4채널 × 2 카메라(agentview + wrist) + proprio 8차원.
    depth는 robosuite 정규화 [0,1] 그대로 사용(E0-3; 절대 미터가 아닌 상대 신호로 충분,
    변환 상수 불필요 — 전처리 단순화).
  - conv1을 4채널로 확장: RGB 가중치는 ImageNet 사전학습 복사, depth 채널은 RGB 평균으로 초기화.
  - K = 8: teacher(OFT)의 requery 주기와 동일 → 습관/teacher가 같은 제어 패턴으로 비교 가능.
  - HP는 C-L0에서만 튜닝 허용(설계서 §2.3), 이후 동결. 기본값은 ACT 공개 구현 표준.
"""
import math

import torch
import torch.nn as nn
import torchvision


def build_backbone(pretrained=True, in_ch=4):
    """ResNet18 → in_ch채널 입력, avgpool 제거(공간 토큰 유지).

    in_ch=4 (기본, RGB-D) / in_ch=3 (RGB-only ablation). in_ch=3이면 표준 ResNet conv1을
    그대로 두므로 depth 경로만 제거되고 나머지 용량은 동일하다 — 순수 modality ablation.

    FrozenBatchNorm2d 사용 — 표준 ACT(DETR 계보)와 동일. 소배치(batch 8)에서 BN 통계가
    불안정해지는 문제를 원천 차단 (검증 워크플로우 발견 반영)."""
    net = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d,
    )
    if in_ch != 3:                       # RGB-D: conv1을 확장하고 depth 채널을 RGB 평균으로 초기화
        old = net.conv1
        conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight[:, :3] = old.weight
            conv.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        net.conv1 = conv
    return nn.Sequential(*list(net.children())[:-2])  # (B, 512, H/32, W/32)


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, n):
        return self.pe[:n]


class ACTPolicy(nn.Module):
    def __init__(
        self,
        action_dim=7,
        proprio_dim=8,
        chunk=8,
        d_model=512,
        nhead=8,
        enc_layers=4,
        dec_layers=7,
        ffn=3200,
        latent_dim=32,
        kl_weight=10.0,
        dropout=0.1,
        pretrained=True,
        in_ch=4,          # 4 = RGB-D (기본) / 3 = RGB-only ablation
    ):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim

        self.in_ch = in_ch
        self.backbones = nn.ModuleList(
            [build_backbone(pretrained, in_ch), build_backbone(pretrained, in_ch)]
        )  # agentview, wrist
        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.pe2d_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
        )
        self.query_embed = nn.Embedding(chunk, d_model)
        self.action_head = nn.Linear(d_model, action_dim)
        self.pe = SinusoidalPE(d_model)
        # 잠재 z·proprio 토큰용 학습 위치 임베딩 (표준 ACT의 additional_pos_embed).
        # 이미지 토큰 PE는 1D 사인파(2D 아님) — 표준 대비 단순화, 문서화된 편차.
        self.additional_pos_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)

        # CVAE encoder: (proprio, action seq) → z
        self.cvae_action_proj = nn.Linear(action_dim, d_model)
        self.cvae_proprio_proj = nn.Linear(proprio_dim, d_model)
        self.cvae_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        cvae_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=dropout, batch_first=True
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, num_layers=4)
        self.cvae_latent_head = nn.Linear(d_model, latent_dim * 2)

    def encode_images(self, images):
        """images: list of (B,4,H,W) per camera → (B, N_tokens, d)."""
        tokens = []
        for bb, img in zip(self.backbones, images):
            f = self.input_proj(bb(img))  # (B, d, h, w)
            B, D, H, W = f.shape
            t = f.flatten(2).transpose(1, 2)  # (B, h*w, d)
            t = t + self.pe(H * W).unsqueeze(0) * self.pe2d_scale
            tokens.append(t)
        return torch.cat(tokens, dim=1)

    def cvae_encode(self, proprio, actions, pad_mask):
        """actions: (B,K,A), pad_mask: (B,K) True=pad → z 분포."""
        B = actions.shape[0]
        seq = torch.cat(
            [
                self.cvae_cls.expand(B, -1, -1),
                self.cvae_proprio_proj(proprio).unsqueeze(1),
                self.cvae_action_proj(actions),
            ],
            dim=1,
        )
        seq = seq + self.pe(seq.shape[1]).unsqueeze(0)
        mask = torch.cat(
            [torch.zeros(B, 2, dtype=torch.bool, device=actions.device), pad_mask], dim=1
        )
        h = self.cvae_encoder(seq, src_key_padding_mask=mask)[:, 0]
        mu, logvar = self.cvae_latent_head(h).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, images, proprio, actions=None, pad_mask=None):
        """학습: actions 제공 → (pred, mu, logvar). 추론: actions=None → pred (z=0)."""
        B = proprio.shape[0]
        if actions is not None:
            mu, logvar = self.cvae_encode(proprio, actions, pad_mask)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=proprio.device, dtype=proprio.dtype)

        img_tokens = self.encode_images(images)
        ctx = torch.cat(
            [
                self.latent_proj(z).unsqueeze(1) + self.additional_pos_embed[0],
                self.proprio_proj(proprio).unsqueeze(1) + self.additional_pos_embed[1],
                img_tokens,
            ],
            dim=1,
        )
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        h = self.transformer(ctx, queries)
        pred = self.action_head(h)  # (B, K, A)
        return pred, mu, logvar

    def loss(self, images, proprio, actions, pad_mask):
        pred, mu, logvar = self.forward(images, proprio, actions, pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        l1 = (torch.abs(pred - actions) * valid).sum() / valid.sum().clamp(min=1) / self.action_dim
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + self.kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}

    @torch.inference_mode()
    def act(self, images, proprio):
        pred, _, _ = self.forward(images, proprio)
        return pred  # (B, K, A) — 정규화 공간; 호출측에서 unnormalize
