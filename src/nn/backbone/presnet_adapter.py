# -*- coding: utf-8 -*-
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from collections import OrderedDict

from .common import get_activation, FrozenBatchNorm2d
from ...core import register
from .adapter import MonaFreq2DAdapter

__all__ = ['PResNetAdapter']

# -----------------------------
# ResNet 配置
# -----------------------------
ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}

# 注意：与原始代码保持拼写一致（donwload_url）
donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


# -----------------------------
# 基础层
# -----------------------------
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        out = self.act(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        out = self.act(out)
        return out


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant, act=act
                )
            )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


# =========================
# 2) PResNetAdapter（带可插拔 Adapter，双路输出）
# =========================
@register()
class PResNetAdapter(nn.Module):
    def __init__(self,
                 depth,
                 variant='d',
                 num_stages=4,
                 return_idx=[0, 1, 2, 3],
                 act='relu',
                 freeze_at=-1,
                 freeze_norm=True,
                 pretrained=False,
                 # ===== Adapter相关超参（Mona-Freq）=====
                 enable_adapters=True,
                 mona_dim=128,
                 mona_dropout=0.1,
                 # ===== 频域分支的超参 =====
                 adapter_cutoff_ratio=0.5,
                 adapter_scale_init=0.1,
                 mona_use_se=True):
        super().__init__()

        self.enable_adapters = enable_adapters
        self.mona_dim = mona_dim
        self.mona_dropout = mona_dropout
        self.adapter_cutoff_ratio = adapter_cutoff_ratio
        self.adapter_scale_init = adapter_scale_init

        # ---- stem ----
        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(cin, cout, k, s, act=act))
            for cin, cout, k, s, name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]  # [256, 512, 1024, 2048]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i],
                       stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        # 冻结部分 stage（如果需要）
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        # 将 BN 替换为 FrozenBN（若设置）
        if freeze_norm:
            self._freeze_norm(self)

        # 预训练
        if pretrained:
            if isinstance(pretrained, bool) or 'http' in str(pretrained):
                state = torch.hub.load_state_dict_from_url(donwload_url[depth], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state, strict=False)
            print(f'Load PResNet{depth} state_dict')

        # ====== 构建 Mona-Freq Adapter（每个stage各自、两模态独立一套）======
        self.adapters_mod0 = nn.ModuleList()
        self.adapters_mod1 = nn.ModuleList()
        for n in range(num_stages):
            c = _out_channels[n]
            self.adapters_mod0.append(
                MonaFreq2DAdapter(
                    in_channels=c,
                    bottleneck_dim=self.mona_dim,
                    dropout_p=self.mona_dropout,
                    cutoff_ratio=self.adapter_cutoff_ratio,
                    scale_init=self.adapter_scale_init,
                    use_se=mona_use_se
                )
            )
            self.adapters_mod1.append(
                MonaFreq2DAdapter(
                    in_channels=c,
                    bottleneck_dim=self.mona_dim,
                    dropout_p=self.mona_dropout,
                    cutoff_ratio=self.adapter_cutoff_ratio,
                    scale_init=self.adapter_scale_init,
                    use_se=mona_use_se
                )
            )


        # self._apply_stage_mode(stage2=True)
        # #DEBUG
        # print("\n[Trainable parameters]")
        # for name, p in self.named_parameters():
        #     if p.requires_grad:
        #         print("  ✅", name)
        # print("\n[Frozen parameters]")
        # for name, p in self.named_parameters():
        #     if not p.requires_grad:
        #         print("  ❄️", name)

    # ----------------- 阶段切换 API -----------------
    def switch_to_stage1(self):
        """第一阶段（共享主干，不启用Adapter或不训练Adapter）"""
        self._apply_stage_mode(stage2=False)

    def switch_to_stage2(self):
        """第二阶段（冻结主干，启用Adapter，仅训练Adapter）"""
        self._apply_stage_mode(stage2=True)

    # ----------------- 权重加载（供二阶段使用） -----------------
    def load_backbone_from_stage1(self, ckpt_path: str):
        """
        从第一阶段权重中加载 backbone 参数（忽略 adapter），用于第二阶段开始前调用。
        """
        state = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = self.load_state_dict(state, strict=False)
        print('[Stage2] load backbone from Stage1')
        print('   missing keys (usually adapters.*):', [k for k in missing if 'adapters_' in k])
        if unexpected:
            print('   unexpected keys:', unexpected)

    # ----------------- 仅训练 Adapter：外部优化器使用 -----------------
    def adapter_parameters(self):
        """
        返回仅包含两路 Adapter 的可训练参数迭代器，便于在 solver 里：
            optimizer = AdamW(model.adapter_parameters(), ...)
        """
        for m in list(self.adapters_mod0) + list(self.adapters_mod1):
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    # ----------------- 前向：双模态输入，分路过 Adapter，再拼回 -----------------
    def forward(self, x0, x1):
        # 保证 batch 对齐
        assert x0.shape[0] == x1.shape[0], "x0/x1 的 batch 大小不一致"
        # [B, ...] + [B, ...] -> [2B, ...]
        x = torch.cat((x0, x1), dim=0)
        B_single = x.shape[0] // 2

        # stem
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        outs_0, outs_1 = [], []
        for idx, stage in enumerate(self.res_layers):
            # 主干第 idx 个 stage
            x = stage(x)

            # 切分两模态
            x0_s = x[:B_single, ...]
            x1_s = x[B_single:, ...]

            # 过各自 adapter
            if self.enable_adapters:
                x0_s = self.adapters_mod0[idx](x0_s)
                x1_s = self.adapters_mod1[idx](x1_s)

            # 收集多尺度输出（已过 adapter）
            if idx in self.return_idx:
                outs_0.append(x0_s)
                outs_1.append(x1_s)

            # 串联：拼回去，作为下一 stage 的输入
            x = torch.cat((x0_s, x1_s), dim=0)

        return outs_0, outs_1

    # ----------------- 内部小工具：开关参数的 requires_grad -----------------
    @staticmethod
    def _set_module_trainable(m: nn.Module, flag: bool):
        for p in m.parameters():
            p.requires_grad = flag

    # ----------------- 冻结 BN：不更新均值/方差，固定为 eval 行为 -----------------
    @staticmethod
    def _set_bn_eval_(m: nn.Module):
        """
        将模型中的 BatchNorm 设为 eval()，并关闭其追踪统计量更新。
        若 __init__ 中已用 FrozenBatchNorm2d 替换，这里相当于双保险。
        """
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm2d, )):
                mod.eval()
                mod.track_running_stats = False
                if hasattr(mod, 'momentum'):
                    mod.momentum = 0.0

    # ----------------- 内部：阶段逻辑与冻结 -----------------
    def _apply_stage_mode(self, stage2: bool):
        """
        stage2=False：关闭Adapter训练，解冻backbone（预训练/第一阶段）
        stage2=True ：开启Adapter训练，冻结backbone（第二阶段，仅训Adapter）
        """
        self.enable_adapters = stage2

        # Backbone 模块集合（conv1 + res_layers）
        backbone_modules = [self.conv1] + list(self.res_layers)

        if stage2:
            # 二阶段：冻结 backbone 参数
            for m in backbone_modules:
                self._set_module_trainable(m, False)
            # 冻结 BN 的统计（确保不更新）
            self._set_bn_eval_(self)

            # 仅 Adapter 可训练
            for m in list(self.adapters_mod0) + list(self.adapters_mod1):
                self._set_module_trainable(m, True)

            # 保持 train()，但确保 BN 仍为 eval（防外部 .train() 复写）
            super().train(True)
            self._set_bn_eval_(self)

        else:
            # 一阶段：允许 backbone 训练；默认不训练 adapter（可按需调整）
            for m in backbone_modules:
                self._set_module_trainable(m, True)

            for m in list(self.adapters_mod0) + list(self.adapters_mod1):
                # 若一阶段你想预热 adapter，可改为 True
                self._set_module_trainable(m, False)

            # 一阶段允许 BN 正常训练
            super().train(True)

    # ----------------- 冻结工具（给 freeze_at 用） -----------------
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m
