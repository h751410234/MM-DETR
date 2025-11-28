"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register

from copy import deepcopy

__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
                 backbone: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
       # self.print_encoder_breakdown()

    def forward(self, x0, x1, targets=None):  # 适配多模态
        x0, x1 = self.backbone(x0, x1)  # feature list [/8(512),/16(1024),/32(2048)] 每个模态
        x = self.encoder(x0, x1)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def print_model_parameters(self):
        """统计并打印各部分参数量"""
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        parts = {
            "Backbone": self.backbone,
            "Encoder": self.encoder,
            "Decoder": self.decoder,
        }

        print("\n===== Model Parameter Summary =====")
        total_all, trainable_all = 0, 0
        for name, module in parts.items():
            total, trainable = count_params(module)
            total_all += total
            trainable_all += trainable
            print(f"{name:10s} → Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")

        print("-----------------------------------")
        print(f"Total Parameters: {total_all/1e6:.2f}M | Trainable: {trainable_all/1e6:.2f}M")
        print("===================================\n")

    def print_encoder_breakdown(self, topk_tensors: int = 10, topk_leaf: int = 10):
        """
        打印 Encoder 内部参数量拆解（按模块族群 & 细到每个 block）。
        - topk_tensors: 打印 encoder 内部最大的若干个参数张量
        - topk_leaf   : 打印 encoder 内部按“叶子模块（nn.Conv2d/nn.Linear 等）”聚合的 Top-K
        """
        import re
        import torch.nn as nn

        enc: nn.Module = getattr(self, "encoder", None)
        assert isinstance(enc, nn.Module), "self.encoder 必须是 nn.Module"

        # ------- 1) 定义“族群”匹配器（按名字前缀做聚合） -------
        group_specs = [
            ("input_proj0", r"^input_proj0(\.|$)"),
            ("input_proj1", r"^input_proj1(\.|$)"),
            ("input_proj", r"^input_proj(\.|$)"),
            ("cross_mamba", r"^cross_mamba(\.|$)"),
            ("enh_td", r"^enh_td(\.|$)"),
            ("enh_bu", r"^enh_bu(\.|$)"),
            ("skip_low", r"^skip_low(\.|$)"),
            ("skip_high", r"^skip_high(\.|$)"),
            ("fpn_gate_low_ir", r"^fpn_gate_low_ir(\.|$)"),
            ("pan_gate_high_ir", r"^pan_gate_high_ir(\.|$)"),
            ("transformer", r"^encoder(\.|$)"),  # TransformerEncoder(ModuleList)
            ("lateral_convs", r"^lateral_convs(\.|$)"),
            ("fpn_blocks", r"^fpn_blocks(\.|$)"),
            ("downsample_convs", r"^downsample_convs(\.|$)"),
            ("pan_blocks", r"^pan_blocks(\.|$)"),
        ]
        group_specs = [(name, re.compile(pat)) for name, pat in group_specs]

        # ------- 2) 先把 encoder 的 named_parameters 抓出来 -------
        # 注意：只统计 encoder 内部
        enc_named_params = list(enc.named_parameters())

        # 总数 & 组聚合
        total_params = 0
        group_sums = {name: 0 for name, _ in group_specs}
        unmatched_sum = 0

        for name, p in enc_named_params:
            n = p.numel()
            total_params += n
            matched = False
            for gname, gpat in group_specs:
                if gpat.match(name):
                    group_sums[gname] += n
                    matched = True
                    break
            if not matched:
                unmatched_sum += n

        # ------- 3) 打印组汇总（从大到小） -------
        print("\n===== Encoder Parameter Breakdown (by group) =====")
        print(f"Encoder TOTAL : {total_params / 1e6:.2f} M params")
        pairs_sorted = sorted(group_sums.items(), key=lambda x: x[1], reverse=True)
        for gname, s in pairs_sorted:
            if s > 0:
                print(f"  {gname:18s}: {s / 1e6:8.3f} M  ({100.0 * s / total_params:5.1f}%)")
        if unmatched_sum > 0:
            print(
                f"  {'<unmatched>':18s}: {unmatched_sum / 1e6:8.3f} M  ({100.0 * unmatched_sum / total_params:5.1f}%)")
        print("--------------------------------------------------")

        # ------- 4) 细化：FPN/PAN 每个 block 的参数量 -------
        def agg_indexed_block(prefix):
            # 例如 prefix='fpn_blocks' 或 'pan_blocks'
            # 统计每个 index（如 fpn_blocks.0.*, fpn_blocks.1.*）的参数
            idx_sum = {}
            pat = re.compile(rf"^{prefix}\.(\d+)\.")
            for name, p in enc_named_params:
                m = pat.match(name)
                if m:
                    idx = int(m.group(1))
                    idx_sum[idx] = idx_sum.get(idx, 0) + p.numel()
            return idx_sum

        for prefix in ["fpn_blocks", "pan_blocks", "downsample_convs", "lateral_convs"]:
            idx_sum = agg_indexed_block(prefix)
            if idx_sum:
                print(f"\n-- {prefix} per-block --")
                for k in sorted(idx_sum.keys()):
                    v = idx_sum[k]
                    print(f"  {prefix}[{k}]: {v / 1e6:.3f} M  ({100.0 * v / total_params:4.1f}% of encoder)")

        # ------- 5) Top-K 最大参数张量 -------
        print("\n===== Encoder Top-{} Largest Tensors =====".format(topk_tensors))
        top_tensors = sorted(enc_named_params, key=lambda kv: kv[1].numel(), reverse=True)[:topk_tensors]
        for name, p in top_tensors:
            shape = tuple(p.shape)
            print(f"  {name:60s} {str(shape):>20s}  -> {p.numel() / 1e6:6.3f} M")

        # ------- 6) Top-K 最大“叶子层”聚合（按 nn.Module 叶子） -------
        # 叶子层 = 没有子模块的 nn.Module（通常是 Conv2d/Linear/BatchNorm 等）
        leaf_param_sum = {}
        leaf_class = {}

        # 先构建 name->module 映射
        name2module = dict(enc.named_modules())
        # 再遍历每个参数，找它归属的叶子层
        for pname, p in enc_named_params:
            # pname 类似 'fpn_blocks.0.xxx.weight'
            # 取掉最后的 '.weight' / '.bias' 得到叶子模块 name
            leaf_name = pname.rsplit('.', 1)[0]
            # 在极少数实现里可能还有一层参数名，需要兜底：逐级回退直到命中 module
            while leaf_name not in name2module and '.' in leaf_name:
                leaf_name = leaf_name.rsplit('.', 1)[0]
            m = name2module.get(leaf_name, None)
            if m is None:
                # 找不到就归到 unmatched 叶子（一般不会出现）
                leaf_name = "<unmatched_leaf>"
                leaf_param_sum[leaf_name] = leaf_param_sum.get(leaf_name, 0) + p.numel()
                leaf_class[leaf_name] = "Unknown"
            else:
                # 只把“叶子模块”（无子模块）计入
                if sum(1 for _ in m.children()) == 0:
                    leaf_param_sum[leaf_name] = leaf_param_sum.get(leaf_name, 0) + p.numel()
                    leaf_class[leaf_name] = m.__class__.__name__

        leaf_sorted = sorted(leaf_param_sum.items(), key=lambda x: x[1], reverse=True)[:topk_leaf]
        print("\n===== Encoder Top-{} Leaf Modules (aggregated) =====".format(topk_leaf))
        for lname, s in leaf_sorted:
            cls = leaf_class.get(lname, "Unknown")
            print(f"  {lname:60s} [{cls:12s}] -> {s / 1e6:6.3f} M")

        print("\n===============================================\n")
