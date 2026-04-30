"""EPRFormer model definition.

The implementation is self-contained and does not require custom CUDA
extensions.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _make_divisible(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_chans),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class LightweightLinearAttention2d(nn.Module):
    """EfficientViT-style ReLU linear attention with local multi-scale tokens."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        scales: Sequence[int] = (3, 5),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = _make_divisible(dim, heads) // heads
        self.inner_dim = self.head_dim * heads
        self.eps = eps

        self.qkv = nn.Conv2d(dim, self.inner_dim * 3, kernel_size=1, bias=False)
        self.ms_dw = nn.ModuleList(
            [
                nn.Conv2d(
                    self.inner_dim * 3,
                    self.inner_dim * 3,
                    kernel_size=k,
                    padding=k // 2,
                    groups=self.inner_dim * 3,
                    bias=False,
                )
                for k in scales
            ]
        )
        self.proj = ConvBNAct(self.inner_dim, dim, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        b, _, h, w = x.shape
        qkv = self.qkv(x)
        if self.ms_dw:
            qkv = qkv + sum(branch(qkv) for branch in self.ms_dw) / len(self.ms_dw)

        q, k, v = qkv.chunk(3, dim=1)
        # Linear attention is sensitive to fp16 underflow/overflow because the
        # denominator is a reduction over all spatial tokens. Keep this small
        # block in fp32 even when the surrounding model uses AMP.
        q = q.float()
        k = k.float()
        v = v.float()
        q = q.flatten(2).transpose(1, 2).reshape(b, h * w, self.heads, self.head_dim)
        k = k.flatten(2).transpose(1, 2).reshape(b, h * w, self.heads, self.head_dim)
        v = v.flatten(2).transpose(1, 2).reshape(b, h * w, self.heads, self.head_dim)

        q = F.relu(q).transpose(1, 2)
        k = F.relu(k).transpose(1, 2)
        v = v.transpose(1, 2)

        kv = torch.matmul(k.transpose(-2, -1), v)
        normalizer = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)).clamp_min(self.eps)
        out = torch.matmul(q, kv) / normalizer
        out = out.transpose(1, 2).reshape(b, h * w, self.inner_dim).transpose(1, 2)
        out = out.reshape(b, self.inner_dim, h, w).to(input_dtype)
        return self.proj(out)


def _window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = x.shape[-2:]
    x = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, window_size * window_size, c)
    return x, (hp, wp)


def _window_reverse(
    windows: torch.Tensor,
    padded_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
    window_size: int,
) -> torch.Tensor:
    b, num_windows, tokens, c = windows.shape
    hp, wp = padded_hw
    assert tokens == window_size * window_size
    assert num_windows == (hp // window_size) * (wp // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
    h, w = original_hw
    return x[:, :, :h, :w]


class PromptGuidedRoutingAttention(nn.Module):
    """BiFormer-style region routing adapted to prompt cross-attention.

    The implementation favors readability over speed. It is suitable as a first
    correctness prototype before writing a fused or vectorized training version.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        window_size: int = 4,
        topk: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.topk = topk

        assert dim % heads == 0, "dim must be divisible by heads"
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

    def _attention(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        input_dtype = q_tokens.dtype
        n_win, q_len, _ = q_tokens.shape
        kv_len = kv_tokens.shape[1]
        q = self.q(q_tokens).float().view(n_win, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_tokens).float().view(n_win, kv_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_tokens).float().view(n_win, kv_len, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        out = torch.matmul(attn.softmax(dim=-1), v)
        out = out.transpose(1, 2).reshape(n_win, q_len, self.dim).to(input_dtype)
        return self.proj(out)

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor]) -> torch.Tensor:
        if prompt is None:
            return x

        b, c, h, w = x.shape
        x_windows, padded_hw = _window_partition(x, self.window_size)
        p_windows, _ = _window_partition(prompt, self.window_size)

        x_desc = x_windows.mean(dim=2)
        p_desc = p_windows.mean(dim=2)
        score = torch.matmul(x_desc, p_desc.transpose(1, 2)) / math.sqrt(c)
        topk = min(self.topk, p_windows.size(1))
        routed = score.topk(k=topk, dim=-1).indices

        routed_out = []
        for batch_idx in range(b):
            selected = p_windows[batch_idx][routed[batch_idx]]
            selected = selected.reshape(x_windows.size(1), topk * self.window_size * self.window_size, c)
            routed_out.append(self._attention(x_windows[batch_idx], selected))
        y_windows = torch.stack(routed_out, dim=0)
        y = _window_reverse(y_windows, padded_hw, (h, w), self.window_size)
        gate = self.gate(torch.cat([x, y], dim=1))
        return x + gate * y


class MSCAConvBlock(nn.Module):
    """SegNeXt-style cheap multi-scale convolutional attention block."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.BatchNorm2d(dim)
        self.expand = ConvBNAct(dim, hidden, kernel_size=1)
        self.dw3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        self.dw5 = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False)
        self.dw7 = nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden, bias=False)
        self.attn = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.project = ConvBNAct(hidden, dim, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(self.norm(x))
        context = self.dw3(y) + self.dw5(y) + self.dw7(y)
        y = y + context * self.attn(context)
        return x + self.project(y)


class ContentAwareUpsample(nn.Module):
    """Content-aware lightweight upsampling for scale=2 feature pyramids."""

    def __init__(self, channels: int, scale: int = 2, kernel_size: int = 5, compressed_channels: int = 32) -> None:
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, compressed_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(compressed_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if size is None:
            x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        else:
            x_up = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        gate = self.encoder(x_up)
        return x_up + gate * self.refine(x_up)


class EPRFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        window_size: int = 4,
        topk: int = 2,
        scales: Sequence[int] = (3, 5),
        use_prompt_routing: bool = True,
    ) -> None:
        super().__init__()
        self.use_prompt_routing = use_prompt_routing
        self.prompt_attn = PromptGuidedRoutingAttention(dim, heads, window_size, topk)
        self.linear_attn = LightweightLinearAttention2d(dim, heads=heads, scales=scales)
        self.conv_attn = MSCAConvBlock(dim)
        self.ffn = nn.Sequential(
            ConvBNAct(dim, dim * 2, kernel_size=1),
            ConvBNAct(dim * 2, dim, kernel_size=1, act=False),
        )

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_prompt_routing:
            x = self.prompt_attn(x, prompt)
        x = x + self.linear_attn(x)
        x = self.conv_attn(x)
        return x + self.ffn(x)


class EPRFormerMixer(nn.Module):
    """Drop-in mixer with the same broad interface as the original Mamba wrapper."""

    def __init__(
        self,
        n_layer: int = 1,
        patch_size: int = 2,
        in_chans: int = 64,
        heads: int = 4,
        window_size: int = 4,
        topk: int = 2,
        use_prompt_routing: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = in_chans * patch_size * patch_size
        self.patch_embedding = nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList(
            [
                EPRFormerBlock(
                    self.embed_dim,
                    heads=heads,
                    window_size=window_size,
                    topk=topk,
                    use_prompt_routing=use_prompt_routing,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        coarse: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del coarse
        b, _, h, w = x.shape
        x = self.patch_embedding(x)
        if prompt is not None:
            prompt = self.patch_embedding(prompt)
        for layer in self.layers:
            x = layer(x, prompt)
        x = self.norm(x)
        x = x.reshape(b, self.embed_dim, -1)
        x = F.fold(x, output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)
        return x


class LitePFSM(nn.Module):
    def __init__(self, in_channel: int = 64, n_layer: int = 1, patch_size: int = 2) -> None:
        super().__init__()
        self.m5 = EPRFormerMixer(n_layer, patch_size, in_channel)
        self.m4 = EPRFormerMixer(n_layer, patch_size, in_channel)
        self.m3 = EPRFormerMixer(n_layer, patch_size, in_channel)
        self.m2 = EPRFormerMixer(n_layer, patch_size * 2, in_channel)
        self.m1 = EPRFormerMixer(n_layer, patch_size * 2, in_channel)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
        e5: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f5 = self.m5(e5, None, p5)
        f4 = self.m4(e4, None, p4)
        f3 = self.m3(e3, None, p3)
        f2 = self.m2(e2, None, p2)
        f1 = self.m1(e1, None, p1)
        return f1, f2, f3, f4, f5


class LitePointPredictor(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.up54 = ContentAwareUpsample(channels)
        self.up43 = ContentAwareUpsample(channels)
        self.up32 = ContentAwareUpsample(channels)
        self.up21 = ContentAwareUpsample(channels)
        self.d4 = MSCAConvBlock(channels)
        self.d3 = MSCAConvBlock(channels)
        self.d2 = MSCAConvBlock(channels)
        self.d1 = MSCAConvBlock(channels)
        self.fusion_shallow = MSCAConvBlock(channels)
        self.course_output = nn.Conv2d(channels, 2, kernel_size=3, padding=1)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f5: torch.Tensor,
        shallow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d4 = self.d4(self.up54(f5, f4.shape[-2:]) + f4)
        d3 = self.d3(self.up43(d4, f3.shape[-2:]) + f3)
        d2 = self.d2(self.up32(d3, f2.shape[-2:]) + f2)
        d1 = self.d1(self.up21(d2, f1.shape[-2:]) + f1)
        d0 = self.fusion_shallow(F.interpolate(d1, size=shallow.shape[-2:], mode="bilinear", align_corners=False) + shallow)
        return self.course_output(d0), d0


class ASPP(nn.Module):
    def __init__(self, dim: int, in_dim: int) -> None:
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        down_dim = in_dim // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, 1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 3, dilation=2, padding=2),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 3, dilation=4, padding=4),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 3, dilation=6, padding=6),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, 1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, 1), nn.BatchNorm2d(in_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.shape[-2:], mode="bilinear")
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), dim=1))


class PBDEncoder(nn.Module):
    def __init__(self, in_channel: int = 64, backbone: str = "resnet50d", pretrained: bool = True) -> None:
        super().__init__()
        self.bkbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        feature_channels = self.bkbone.feature_info.channels()

        self.aspp_x = ASPP(feature_channels[4], in_channel)
        self.dem4 = ConvBNAct(feature_channels[3], in_channel)
        self.dem3 = ConvBNAct(feature_channels[2], in_channel)
        self.dem2 = ConvBNAct(feature_channels[1], in_channel)
        self.dem1 = ConvBNAct(feature_channels[0], in_channel)

        self.aspp_p = ASPP(feature_channels[4], in_channel)
        self.dem4_p = ConvBNAct(feature_channels[3], in_channel)
        self.dem3_p = ConvBNAct(feature_channels[2], in_channel)
        self.dem2_p = ConvBNAct(feature_channels[1], in_channel)
        self.dem1_p = ConvBNAct(feature_channels[0], in_channel)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor):
        e1, e2, e3, e4, e5 = self.bkbone(x)
        p1, p2, p3, p4, p5 = self.bkbone(prompt)
        e5 = self.aspp_x(e5)
        e4 = self.dem4(e4)
        e3 = self.dem3(e3)
        e2 = self.dem2(e2)
        e1 = self.dem1(e1)
        p5 = self.aspp_p(p5)
        p4 = self.dem4_p(p4)
        p3 = self.dem3_p(p3)
        p2 = self.dem2_p(p2)
        p1 = self.dem1_p(p1)
        return e1, e2, e3, e4, e5, p1, p2, p3, p4, p5


class LinearContextBlock(nn.Module):
    def __init__(self, channels: int = 64, heads: int = 4, scales: Sequence[int] = (3, 5)) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.attn = LightweightLinearAttention2d(channels, heads=heads, scales=scales)
        self.ffn = nn.Sequential(
            ConvBNAct(channels, channels * 2, kernel_size=1),
            ConvBNAct(channels * 2, channels, kernel_size=1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        return x + self.ffn(x)


class PatchLinearRefiner(nn.Module):
    def __init__(self, channels: int = 64, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = channels * patch_size * patch_size
        self.patch_embedding = nn.Conv2d(channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.context = LinearContextBlock(self.embed_dim, heads=4, scales=(3, 5))
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = self.patch_embedding(x)
        x = self.norm(self.context(x))
        x = x.reshape(b, self.embed_dim, -1)
        x = F.fold(x, output_size=(hp, wp), kernel_size=self.patch_size, stride=self.patch_size)
        return x[:, :, :h, :w]


class CountingPredictor(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.regressor_fcn_neg = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.ReLU(inplace=True))
        self.regressor_fcn_pos = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.ReLU(inplace=True))
        self.regressor_neg = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.ReLU(inplace=True))
        self.regressor_pos = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, neg_input: torch.Tensor, pos_input: torch.Tensor, t5: torch.Tensor):
        neg = F.interpolate(neg_input, size=t5.shape[-2:], mode="bilinear")
        pos = F.interpolate(pos_input, size=t5.shape[-2:], mode="bilinear")
        neg = self.regressor_neg(self.regressor_fcn_neg(F.adaptive_avg_pool2d(t5 * neg, 1))).flatten()
        pos = self.regressor_pos(self.regressor_fcn_pos(F.adaptive_avg_pool2d(t5 * pos, 1))).flatten()
        return neg, pos


class LinePredictor(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.t2_t1_fusion = ConvBNAct(channels, channels)
        self.neg_line_pre = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.pos_line_pre = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, neg_map: torch.Tensor, pos_map: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, x: torch.Tensor):
        t12 = self.t2_t1_fusion(F.interpolate(t2, size=t1.shape[-2:], mode="bilinear") + t1)
        t12 = F.interpolate(t12, size=x.shape[-2:], mode="bilinear")
        line_neg = self.neg_line_pre(t12 * neg_map + t12)
        line_pos = self.pos_line_pre(t12 * pos_map + t12)
        return line_neg, line_pos


class PBDFeatureFusion(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        topk: int = 2,
    ) -> None:
        super().__init__()
        self.pg = nn.ModuleList(
            [
                PromptGuidedRoutingAttention(channels, window_size=4, topk=topk),
                PromptGuidedRoutingAttention(channels, window_size=4, topk=topk),
                PromptGuidedRoutingAttention(channels, window_size=4, topk=topk),
                PromptGuidedRoutingAttention(channels, window_size=4, topk=topk),
                PromptGuidedRoutingAttention(channels, window_size=4, topk=topk),
            ]
        )
        self.context = nn.ModuleList([LinearContextBlock(channels) for _ in range(5)])

    def forward(self, e_feats: Sequence[torch.Tensor], p_feats: Sequence[torch.Tensor]):
        outputs = []
        for idx, (e, p) in enumerate(zip(e_feats, p_feats)):
            e = self.pg[idx](e, p)
            e = self.context[idx](e)
            outputs.append(e)
        return tuple(outputs)


class EPRFormer(nn.Module):
    """Full EPRFormer with PG-BRA, LMLA, and CADD enabled."""

    def __init__(
        self,
        backbone: str = "resnet50d",
        pretrained: bool = True,
        pre_path: Optional[str] = None,
    ) -> None:
        del pre_path
        super().__init__()
        self.encoder = PBDEncoder(64, backbone=backbone, pretrained=pretrained)
        self.shallow_x = ConvBNAct(3, 64)
        self.fusion = PBDFeatureFusion()
        self.point_predictor = LitePointPredictor(64)
        self.counting_predictor = CountingPredictor(64)
        self.line_predictor = LinePredictor(64)
        self.refine_mixer = PatchLinearRefiner(64)
        self.point_refine = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor):
        shallow_x = self.shallow_x(x)
        e1, e2, e3, e4, e5, p1, p2, p3, p4, p5 = self.encoder(x, prompt)
        f1, f2, f3, f4, f5 = self.fusion((e1, e2, e3, e4, e5), (p1, p2, p3, p4, p5))
        point_course, d0 = self.point_predictor(f1, f2, f3, f4, f5, shallow_x)
        refined_feature = self.refine_mixer(d0)
        point_refine = self.point_refine(refined_feature + shallow_x)

        if self.training:
            neg_map = torch.sigmoid(point_course[:, 0:1])
            pos_map = torch.sigmoid(point_course[:, 1:2])
            regression_neg, regression_pos = self.counting_predictor(neg_map, pos_map, f5)
            line_neg, line_pos = self.line_predictor(neg_map, pos_map, f1, f2, x)
            return point_refine, point_course, regression_neg, regression_pos, line_neg, line_pos
        return point_refine


if __name__ == "__main__":
    model = EPRFormer()
    image = torch.randn(1, 3, 512, 512)
    prompt = torch.randn(1, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        output = model(image, prompt)
    print(tuple(output.shape))
