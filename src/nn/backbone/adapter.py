# -*- coding: utf-8 -*-
"""
MonaFreq2DAdapter (optimized version, same architecture)
- Mona 内容分支（C→D→C，DW 3/5/7 融合） + 频域低/高分支（共享DW3x3）
- 逐像素 Router 融合三路校正
- 适配 NCHW，AMP/半精度下的 FFT 数值安全
- 通道门控可选：SE（原版）/ ECA / ScaleOnlyGate（极致轻量），支持低/高频共享门控
- 小幅代码级优化：减少 expand/zeros_like/重复 view/cast
"""

from typing import Tuple
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# LayerNorm2d (NCHW)
# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            w = self.weight[:, None, None]
            b = self.bias[:,   None, None]
            x = x * w + b
        return x


# -----------------------------
# Mona 内容分支 (2D)
# -----------------------------
class MonaOp2d(nn.Module):
    """Depthwise 3/5/7 conv fusion + 1x1 projector, residual."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)
        self.projector = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # 三个 DWConv 平均 + 残差，再加 1x1 projector 残差
        conv_sum = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0
        x = conv_sum + identity
        return x + self.projector(x)


# -----------------------------
# SE（原版：两层 1x1）
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv2d(in_channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


# -----------------------------
# 轻量化门控 A：ECA（Efficient Channel Attention）
# -----------------------------
class ECABlock(nn.Module):
    """
    ECA: GAP -> 1D conv(k) on channel axis -> sigmoid
    将 GAP 后的 (B, C, 1, 1) 视作 (B, 1, C) 做一维卷积建模临近通道依赖
    """
    def __init__(self, in_channels: int, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "ECA kernel size must be odd"
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)                                # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)              # (B, 1, C)
        y = self.conv(y)                               # (B, 1, C)
        y = torch.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


# -----------------------------
# 轻量化门控 B：ScaleOnlyGate（极致轻量）
# -----------------------------
class ScaleOnlyGate(nn.Module):
    """
    GAP -> per-channel gate (sigmoid(alpha * mean + beta))
    仅逐通道缩放，无通道混合；数值稳定且极轻量。
    """
    def __init__(self, in_channels: int, use_bias: bool = True, init_scale: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((in_channels, 1, 1), init_scale))
        self.use_bias = use_bias
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        gate = self.alpha * m
        if self.use_bias:
            gate = gate + self.beta
        gate = torch.sigmoid(gate)
        return x * gate


# -----------------------------
# Util convs
# -----------------------------
def _conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    conv = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
    nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
    if bias:
        nn.init.zeros_(conv.bias)
    return conv

def _dw3x3(C: int) -> nn.Conv2d:
    conv = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=True)
    nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
    nn.init.zeros_(conv.bias)
    return conv


# -----------------------------
# Mona + 频域 三分支适配器
# -----------------------------
class MonaFreq2DAdapter(nn.Module):
    """
    三分支：content( Mona ) + low/high( 频域 ) + Router 逐像素融合
    - 预归一: y = LN2d(x)*gamma + x*gammax
    - content: 1x1(C->D) -> MonaOp2d(D) -> GELU+Dropout -> 1x1(D->C), project2 零初始化
    - low/high: FFT2 -> fftshift -> 方窗低频/其余高频 -> ifft2.real
                -> 共享DW3x3 -> （可选）门控(SE/ECA/ScaleOnly) + 静态β -> per-channel scale
    - Router: 1x1(C->3) + softmax(HW)，像素级融合三路 delta
    """
    def __init__(self,
                 in_channels: int,
                 bottleneck_dim: int = 128,
                 dropout_p: float = 0.1,
                 cutoff_ratio: float = 0.3,
                 scale_init: float = 0.1,
                 use_se: bool = True,
                 se_mode: str = "eca",      # "se" | "eca" | "scale"
                 eca_kernel: int = 3,       # ECA 的 k（需奇数）
                 share_se: bool = False,    # 低/高频是否共享同一门控
                 size_adaptive_gamma: bool = False   # 多尺度下自适应 γ
                 ):
        super().__init__()
        assert 0 < cutoff_ratio < 1, "cutoff_ratio should be in (0,1)"
        C = in_channels
        D = bottleneck_dim

        # Mona 预归一
        self.norm = LayerNorm2d(C)
        self.gamma  = nn.Parameter(torch.ones(C) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(C))

        self.size_adaptive_gamma = bool(size_adaptive_gamma)
        self._ref_hw = None  # 首次前向自动记录参考尺寸 (Hr, Wr)

        # content (Mona)
        self.proj1   = nn.Conv2d(C, D, 1, bias=True)
        self.mona_op = MonaOp2d(D)
        self.drop    = nn.Dropout(dropout_p)
        self.proj2   = nn.Conv2d(D, C, 1, bias=True)
        nn.init.constant_(self.proj2.weight, 0.0)
        if self.proj2.bias is not None:
            nn.init.constant_(self.proj2.bias, 0.0)

        # 频域
        self.cutoff_ratio = float(cutoff_ratio)
        self.freq_dw  = _dw3x3(C)

        # 门控
        self.use_se   = bool(use_se)
        self.se_mode  = se_mode
        self.share_se = bool(share_se)
        if self.use_se:
            if self.se_mode == "eca":
                se_factory = lambda: ECABlock(C, k_size=eca_kernel)
            elif self.se_mode == "scale":
                se_factory = lambda: ScaleOnlyGate(C, use_bias=True, init_scale=1.0)
            else:
                se_factory = lambda: SEBlock(C, reduction_ratio=16)

            if self.share_se:
                se_module = se_factory()
                self.se_low  = se_module
                self.se_high = se_module
            else:
                self.se_low  = se_factory()
                self.se_high = se_factory()
        else:
            self.se_low  = nn.Identity()
            self.se_high = nn.Identity()

        # 静态 β（可与门控叠加）
        self.beta_low     = nn.Parameter(torch.zeros(C, 1, 1))
        self.beta_high    = nn.Parameter(torch.zeros(C, 1, 1))

        # per-branch × per-channel scale
        self.scale_c = nn.Parameter(torch.full((C, 1, 1), float(scale_init)))
        self.scale_l = nn.Parameter(torch.full((C, 1, 1), float(scale_init)))
        self.scale_h = nn.Parameter(torch.full((C, 1, 1), float(scale_init)))

        # Router
        self.router = _conv1x1(C, 3, bias=True)

        self.act = nn.GELU()

    # ------- 频域工具 -------
    @torch.no_grad()
    def _low_mask(self, H: int, W: int, device) -> torch.Tensor:
        half = int(min(H, W) * self.cutoff_ratio // 2)
        cx, cy = H // 2, W // 2
        x0, x1 = max(cx - half, 0), min(cx + half, H)
        y0, y1 = max(cy - half, 0), min(cy + half, W)
        m = torch.zeros((H, W), device=device, dtype=torch.bool)
        m[x0:x1, y0:y1] = True
        return m

    def _fft_low_high(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AMP/半精度安全的 FFT：
        - 统一在 float32 下计算 FFT
        - 利用广播避免 expand(B,C,H,W) 和 zeros_like
        """
        orig_dtype = x.dtype
        ctx = nullcontext()
        if x.is_cuda:
            from torch.cuda.amp import autocast
            ctx = autocast(enabled=False)

        with ctx:
            x32 = x.to(torch.float32)
            F2  = torch.fft.fft2(x32, norm='ortho')
            F2s = torch.fft.fftshift(F2, dim=(-2, -1))
            _, _, H, W = F2s.shape

            # 低频 mask：形状 [1,1,H,W]，靠广播作用到 (B,C,H,W)
            m2d   = self._low_mask(H, W, x32.device)      # [H,W]
            low_m = m2d.view(1, 1, H, W)                  # broadcast to (B,C,H,W)
            high_m = ~low_m

            Fl = F2s * low_m
            Fh = F2s * high_m

            low  = torch.fft.ifft2(torch.fft.ifftshift(Fl, dim=(-2, -1)), norm='ortho').real
            high = torch.fft.ifft2(torch.fft.ifftshift(Fh, dim=(-2, -1)), norm='ortho').real

        if low.dtype != orig_dtype:
            low  = low.to(orig_dtype)
            high = high.to(orig_dtype)
        return low, high

    # ------- 前向 -------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) -> y: (B, C, H, W)
        """
        identity = x
        B, C, H, W = x.shape

        # 记录/获取参考尺寸（首次前向即为参考）
        if self.size_adaptive_gamma and (self._ref_hw is None):
            self._ref_hw = (H, W)

        # Mona 预归一（支持 size_adaptive_gamma）
        gamma  = self.gamma.view(1, -1, 1, 1)
        gammax = self.gammax.view(1, -1, 1, 1)
        if self.size_adaptive_gamma and (self._ref_hw is not None):
            Hr, Wr = self._ref_hw
            s_gamma = math.sqrt(max(1.0, Hr * Wr) / max(1.0, H * W))
            x_norm = self.norm(x) * (gamma * s_gamma) + x * gammax
        else:
            x_norm = self.norm(x) * gamma + x * gammax

        # content 分支（Mona）
        c = self.proj1(x_norm)
        c = self.mona_op(c)
        c = self.act(c)
        c = self.drop(c)
        c = self.proj2(c)                 # 回到 C 维
        c = self.scale_c * c              # per-channel scale

        # 频域分解 + 两分支（用原始 x 做频域）
        low, high = self._fft_low_high(x)
        yl = self.freq_dw(low)
        yh = self.freq_dw(high)

        # 门控（SE/ECA/ScaleOnly 或 Identity） + 静态 β
        dl = self.se_low(yl)  + self.beta_low
        dh = self.se_high(yh) + self.beta_high

        dl = self.scale_l * dl
        dh = self.scale_h * dh

        # Router：逐像素融合
        w = torch.softmax(self.router(x), dim=1)  # (B,3,H,W) -> content/low/high
        delta = w[:, 0:1] * c + w[:, 1:2] * dl + w[:, 2:3] * dh

        return identity + delta
