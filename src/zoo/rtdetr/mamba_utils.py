import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat
from timm.models.layers import trunc_normal_

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # 官方内核


# ----- Mamba SSM 1D -----
class SS1D(nn.Module):
    def __init__(
            self,
            # basic dims =============
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv =================
            d_conv=-1,  # < 2 means no conv
            conv_bias=True,
            # =======================
            dropout=0.,
            bias=False,
            # dt init ===============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # =======================
            softmax_version=False,
            # =======================
            **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}

        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # in proj: [*, d_model] -> [*, 2*d_inner]
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # depthwise conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ======================================
        # VMamba 原版 K=4，原始 SSM K=1；这里保持逻辑不变
        self.K = 1
        if self.forward_core == self.forward_corev0:
            self.K = 1

        # x_proj: inner -> (dt_rank + 2*d_state)
        x_proj_list = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj_list], dim=0)
        )  # (K, C_total, d_inner)
        del x_proj_list

        # dt_projs: rank -> inner
        dt_proj_list = [
            self.dt_init(
                self.dt_rank, self.d_inner,
                dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                **factory_kwargs
            )
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_proj_list], dim=0)
        )  # (K, d_inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_proj_list], dim=0)
        )  # (K, d_inner)
        del dt_proj_list

        # A, D =======================================
        if self.forward_core == self.forward_corev0:
            self.K2 = 1
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K*D, d_state)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K*D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    # ---------- static init helpers ----------
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 初始化权重，保持方差
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化 bias，使 softplus(bias) ∈ [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 保持 fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # ---------- core SSM ----------
    def forward_corev0(self, x: torch.Tensor):
        """
        x: [B, L, D_inner]
        """
        selective_scan = selective_scan_fn

        B, L, D_inner = x.shape
        K = self.K  # 目前=1

        # [B, L, D] -> [B, K(=1), D, L]
        xs = x.transpose(1, 2).unsqueeze(1)  # 不额外 contiguous，后面 float() 会复制

        # x_proj: (b,k,d,l) * (k,c,d) -> (b,k,c,l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)

        # 切分 dt / B / C
        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2
        )

        # dt proj: (b,k,r,l) * (k,d,r) -> (b,k,d,l)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # 展平 K 维，准备给 selective_scan
        xs = xs.float().view(B, -1, L)              # (B, K*D, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, K*D, L)
        Bs = Bs.float()                             # (B, K, d_state, L)
        Cs = Cs.float()                             # (B, K, d_state, L)

        As = -torch.exp(self.A_logs)                # (K*D, d_state), 已是 fp32
        Ds = self.Ds                                # (K*D)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (K*D)

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)                         # (B, K, D, L)

        y = out_y[:, 0].transpose(1, 2).contiguous()  # -> (B, L, D)
        if not self.softmax_version:
            y = self.out_norm(y)

        return y

    # 可以后面替换为其他 core 版本
    forward_core = forward_corev0  # ori mamba

    # ---------- public forward ----------
    def forward(self, x: torch.Tensor, **kwargs):
        """
        这里沿用你原来的注释： x: [B, C, HW] 或 [B, L, d_model]
        只要最后一维是 d_model，Linear 就没问题。
        """
        xz = self.in_proj(x)  # -> [..., 2*d_inner]

        if self.d_conv > 1:
            # conv 分支：假设 x: [B, H*W, d_model]，此分支目前在你 ChannelWeights 里没有用到
            x_part, z = xz.chunk(2, dim=-1)          # (B, H*W, d_inner)
            # 还原成 [B, d_inner, H, W] 的格式
            B = x_part.shape[0]
            HW = x_part.shape[1]
            D = x_part.shape[2]
            H = W = int(HW ** 0.5)                   # 简单假设，如果你原来有特定形状就按原来的来
            x_part = x_part.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

            x_part = self.act(self.conv2d(x_part))   # (B, D, H, W)
            x_part = x_part.flatten(2).transpose(1, 2).contiguous()  # (B, HW, D)

            y = self.forward_core(x_part)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)

        else:
            # 无卷积分支（你当前使用的分支）
            if self.softmax_version:
                x_part, z = xz.chunk(2, dim=-1)
                x_part = F.silu(x_part)
            else:
                xz = F.silu(xz)
                x_part, z = xz.chunk(2, dim=-1)

            y = self.forward_core(x_part)  # [B, L, d_inner]
            y = y * z

        out = self.dropout(self.out_proj(y))
        return out


# =======================
#   通道权重学习模块
# =======================
class ChannelWeights(nn.Module):
    def __init__(self, channel_dim, reduction=4, embed_dim=96):
        super().__init__()
        self.embed_dim = int(embed_dim)  # 固定特征维度 (替代 H*W)

        self.pre = nn.Identity()  # 预留位，保持接口
        self.ssm = SS1D(d_model=self.embed_dim, dropout=0, d_state=16)

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """
        x1, x2: [B, C, H, W]
        返回: [B, 2, C, 1, 1]
        """
        B, C, H, W = x1.shape

        # 拼接后展平空间维： [B,2C,HW]
        x = torch.cat((x1, x2), dim=1).reshape(B, 2 * C, H * W)

        # 自适应池化到 embed_dim: [B,2C,embed_dim]
        x = F.adaptive_avg_pool1d(x, self.embed_dim)

        # SSM 编码: [B,2C,embed_dim] -> [B,2C,embed_dim]
        x = self.ssm(x)

        # head: [B,2C,embed_dim] -> [B,2C,1]
        x = self.head(x).reshape(B, 2, C, 1, 1)  # 直接整理成 [B,2,C,1,1]

        return x


class ChannelRectifyModule(nn.Module):
    def __init__(self, channel_dim, reduction=16, lambda_c=0.5, lambda_s=0.5, embed_dim=96):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(channel_dim=channel_dim, embed_dim=embed_dim)

    def forward(self, x1, x2):
        """
        x1, x2: [B, C, H, W]
        """
        # w: [B, 2, C, 1, 1]
        w = self.channel_weights(x1, x2).to(x1.dtype)

        w1 = w[:, 0, :, :, :]  # [B,C,1,1]
        w2 = w[:, 1, :, :, :]  # [B,C,1,1]

        out_x1 = x1 + w1 * x1
        out_x2 = x2 + w2 * x2

        return out_x1, out_x2
