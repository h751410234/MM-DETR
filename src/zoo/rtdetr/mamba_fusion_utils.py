import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SS2D_RegionLite_2_Tiny(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        dropout: float = 0.0,
        use_norm: bool = True,
        eps: float = 1e-6,
        bc_groups: int = 2,      # 分组在线性
        share_AC: bool = False,  # A/D/dt_bias 按组共享
        bottleneck_r: int = 4,   # 低秩瓶颈倍率
        use_ss2d: bool = False,  # 默认 False（即 SS1D）
    ):
        super().__init__()
        assert d_model % bc_groups == 0, "d_model 必须能被 bc_groups 整除"
        self.C = d_model
        self.S = d_state
        self.G = bc_groups
        self.use_conv = d_conv is not None and d_conv >= 2
        self.share_AC = share_AC
        self.r = max(1, bottleneck_r)
        self.use_ss2d = use_ss2d  # ★ 非参数属性，不进 state_dict

        # 归一化
        self.pre_norm  = nn.GroupNorm(1, d_model, eps=eps) if use_norm else nn.Identity()
        self.post_norm = nn.GroupNorm(1, d_model, eps=eps) if use_norm else nn.Identity()

        # 深度可分卷积
        if self.use_conv:
            self.dw  = nn.Conv2d(
                d_model, d_model,
                kernel_size=d_conv,
                padding=d_conv // 2,
                groups=d_model,
                bias=True,
            )
            self.act = nn.SiLU()

        # 低秩 in/dt/out: C -> mid -> C
        mid = max(self.C // self.r, 8)
        self.in_proj1  = nn.Linear(self.C, mid, bias=False)
        self.in_proj2  = nn.Linear(mid, self.C, bias=True)
        self.dt_proj1  = nn.Linear(self.C, mid, bias=False)
        self.dt_proj2  = nn.Linear(mid, self.C, bias=True)
        self.out_proj1 = nn.Linear(self.C, mid, bias=False)
        self.out_proj2 = nn.Linear(mid, self.C, bias=True)
        nn.init.zeros_(self.out_proj2.weight)
        nn.init.zeros_(self.out_proj2.bias)

        # B/C 合并： [B,C,L] -> [B,2*C*S,L]
        self.bc_bottleneck = nn.Conv1d(self.C, mid, kernel_size=1, groups=self.G, bias=False)
        self.bc_head       = nn.Conv1d(mid, 2 * self.C * self.S, kernel_size=1, groups=1, bias=True)

        # A / D / dt_bias
        if self.share_AC:
            G = self.G
            with torch.no_grad():
                base  = torch.linspace(-0.7, 0.7, steps=self.S)
                Arow  = torch.exp(base)                        # (S,)
                Ainit = Arow.unsqueeze(0).repeat(G, 1)         # [G,S]
            self.A = nn.Parameter(-Ainit)                       # [G,S]
            self.D = nn.Parameter(torch.full((G,), 0.1))        # [G]
            self.dt_bias = nn.Parameter(torch.full((G,), -1.0)) # [G]
        else:
            with torch.no_grad():
                base  = torch.linspace(-0.7, 0.7, steps=self.S)
                Arow  = torch.exp(base)
                Ainit = Arow.unsqueeze(0).repeat(self.C, 1)     # [C,S]
            self.A = nn.Parameter(-Ainit)                        # [C,S]
            self.D = nn.Parameter(torch.full((self.C,), 0.1))    # [C]
            self.dt_bias = nn.Parameter(torch.full((self.C,), -1.0))

        self.res_scale = nn.Parameter(torch.tensor(1e-3))
        self.drop      = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ================== 工具函数 ==================

    def _expand_group_params(self, Bsz: int, L: int, device, dtype):
        """把按组的 A/D/dt_bias 展开到每通道，用于 CUDA 核"""
        if not self.share_AC:
            return (
                self.A.to(dtype=dtype),
                self.D.to(dtype=dtype),
                self.dt_bias.to(dtype=dtype),
            )

        # 组共享 -> 展开到每个通道
        gC = self.C // self.G
        A = self.A.unsqueeze(1).repeat(1, gC, 1).reshape(self.C, self.S)  # [C,S]
        D = self.D.unsqueeze(1).repeat(1, gC).reshape(self.C)             # [C]
        dtb = self.dt_bias.unsqueeze(1).repeat(1, gC).reshape(self.C)     # [C]
        return (
            A.to(dtype=dtype),
            D.to(dtype=dtype),
            dtb.to(dtype=dtype),
        )

    # ---------- SS2D 辅助：1D scan 封装 ----------
    def _scan_1d(self, u, delta, B, C_param, A_full, D_full, dtb_full):
        """
        u      : [B1, C, L]
        delta  : [B1, C, L]
        B      : [B1, C, S, L]
        C_param: [B1, C, S, L]
        """
        return selective_scan_fn(
            u, delta,
            A_full, B, C_param,
            D_full,
            None,          # z
            dtb_full,
            True,          # delta_softplus
            False          # return_last_state
        )

    # ---------- SS2D：双方向 H/W selective scan ----------
    def _ss2d_two_dir(
        self,
        u_T32, del_T32, B_T32, C_T32,
        A_full, D_full, dtb_full,
        H: int, W: int,
    ):
        """
        2D selective scan:
          - 水平方向：沿 W
          - 垂直方向：沿 H
        最后对两方向结果做平均。
        输入 / 输出都是 [B,C,L] (L=H*W)，内部暂时还原成 BCHW。
        """
        Bsz, C, L = u_T32.shape
        S = self.S
        assert L == H * W, f"L({L}) != H*W({H*W})"

        # 还原到 2D
        u_2d   = u_T32.view(Bsz, C, H, W).contiguous()
        del_2d = del_T32.view(Bsz, C, H, W).contiguous()
        B_2d   = B_T32.view(Bsz, C, S, H, W).contiguous()
        C_2d   = C_T32.view(Bsz, C, S, H, W).contiguous()

        # ---- 1) 水平方向：每一行做 1D scan ----
        # [B,C,H,W] -> [B*H, C, W]
        u_h   = u_2d.permute(0, 2, 1, 3).reshape(Bsz * H, C, W).contiguous()
        del_h = del_2d.permute(0, 2, 1, 3).reshape(Bsz * H, C, W).contiguous()
        B_h   = B_2d.permute(0, 3, 1, 2, 4).reshape(Bsz * H, C, S, W).contiguous()
        C_h   = C_2d.permute(0, 3, 1, 2, 4).reshape(Bsz * H, C, S, W).contiguous()

        y_h = self._scan_1d(u_h, del_h, B_h, C_h, A_full, D_full, dtb_full)  # [B*H,C,W]
        y_h = y_h.view(Bsz, H, C, W).permute(0, 2, 1, 3).contiguous()        # [B,C,H,W]

        # ---- 2) 垂直方向：每一列做 1D scan ----
        # [B,C,H,W] -> [B*W, C, H]
        u_v   = u_2d.permute(0, 3, 1, 2).reshape(Bsz * W, C, H).contiguous()
        del_v = del_2d.permute(0, 3, 1, 2).reshape(Bsz * W, C, H).contiguous()
        B_v   = B_2d.permute(0, 4, 1, 2, 3).reshape(Bsz * W, C, S, H).contiguous()
        C_v   = C_2d.permute(0, 4, 1, 2, 3).reshape(Bsz * W, C, S, H).contiguous()

        y_v = self._scan_1d(u_v, del_v, B_v, C_v, A_full, D_full, dtb_full)  # [B*W,C,H]
        y_v = y_v.view(Bsz, W, C, H).permute(0, 2, 3, 1).contiguous()        # [B,C,H,W]

        # ---- H / W 两个方向融合 ----
        y_2d = 0.5 * (y_h + y_v)  # 也可以改成加权

        # 展平成 [B,C,L]
        y_T = y_2d.view(Bsz, C, L).contiguous()
        return y_T

    # ================== 前向 ==================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        """
        B, C, H, W = x.shape
        assert C == self.C, f"Channel mismatch: {C} vs {self.C}"

        z = self.pre_norm(x)
        if self.use_conv:
            z = self.act(self.dw(z))

        # BCHW -> [B,L,C]
        L   = H * W
        seq = z.flatten(2).transpose(1, 2).contiguous()   # [B,L,C]

        # 低秩 in/dt: [B,L,C] -> [B,L,C]
        u         = self.in_proj2(self.in_proj1(seq))     # 驱动 u
        delta_pre = self.dt_proj2(self.dt_proj1(seq))     # delta

        # 生成 B/C： [B,L,C] -> [B,C,L] -> [B,2*C*S,L]
        seq_T = seq.transpose(1, 2).contiguous()          # [B,C,L]
        bc_mid = self.bc_bottleneck(seq_T)                # [B,mid,L]
        bc_all = self.bc_head(bc_mid)                     # [B,2*C*S,L]

        # 直接 reshape 成 B/C 两个张量：[B,2,C,S,L] -> [B,C,S,L] x2
        bc_all = bc_all.view(B, 2, C, self.S, L)          # [B,2,C,S,L]
        B_T = bc_all[:, 0]                                # [B,C,S,L]
        C_T = bc_all[:, 1]                                # [B,C,S,L]

        # u / delta: [B,L,C] -> [B,C,L]
        u_T   = u.transpose(1, 2)                         # [B,C,L]
        del_T = delta_pre.transpose(1, 2)                 # [B,C,L]

        # 展开 A/D/dt_bias 到每通道，并统一到 float32
        A_full, D_full, dtb_full = self._expand_group_params(
            B, L, u_T.device, torch.float32
        )

        # selective_scan_fn 里全用 fp32，避免重复 .to()
        with autocast(enabled=False):
            u_T32   = u_T.to(torch.float32)
            del_T32 = del_T.to(torch.float32)
            B_T32   = B_T.to(torch.float32)
            C_T32   = C_T.to(torch.float32)

            if self.use_ss2d:
                # SS2D 路径：双方向 H/W selective scan
                y_T32 = self._ss2d_two_dir(
                    u_T32, del_T32, B_T32, C_T32,
                    A_full, D_full, dtb_full,
                    H, W,
                )
            else:
                # SS1D
                y_T32 = selective_scan_fn(
                    u_T32,
                    del_T32,
                    A_full,        # 已经是 float32
                    B_T32,
                    C_T32,
                    D_full,        # 已经是 float32
                    None,
                    dtb_full,      # 已经是 float32
                    True,          # delta_softplus
                    False          # return_last_state
                )  # [B,C,L], fp32

        # 回到 BCHW + 低秩 out
        y_seq = y_T32.transpose(1, 2).contiguous()        # [B,L,C]
        y     = self.out_proj2(self.out_proj1(y_seq))     # [B,L,C]
        y     = self.drop(y * self.res_scale)
        y     = y.transpose(1, 2).reshape(B, C, H, W).contiguous()
        y     = self.post_norm(y)

        return x + y
