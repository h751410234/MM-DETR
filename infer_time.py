#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双输入 RT-DETR 模型纯 forward 速度 & FLOPs 测试脚本

- 使用 YAMLConfig + cfg.model 构建模型
- 使用随机 tensor，避开 DataLoader / 后处理 / mAP 等开销
- 只测 model(samples0, samples1) 的时间
- 可选：
    * --batch-size / --img-size / --warmup / --iters / --use-amp
    * 打印参数量（总 & 可训练）
    * 用 thop 估计 FLOPs（/forward & /image）
    * torch.compile(model) 加速实际测速
"""

import os
import sys
import time
import argparse
import copy

import torch
import torch.backends.cudnn as cudnn

# 保证能找到 src/ 包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils


def parse_args():
    parser = argparse.ArgumentParser("RT-DETR style speed benchmark (dual input)", add_help=True)

    # 和原 main 一样的核心配置
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='configs/M3FD/rtdetrv2_r50vd_6x_coco.yml',
        help="path to yaml config"
    )
    parser.add_argument('-r', '--resume', type=str, help='optional: resume / load checkpoint (for weights)')
    parser.add_argument('-d', '--device', type=str, default='cuda:1', help='device, e.g. cuda or cuda:0')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='use AMP (autocast) during benchmark')
    parser.add_argument('--update', '-u', nargs='+', help='update yaml config (same as train)')

    # 环境打印（可以不关心）
    parser.add_argument('--print-method', type=str, default='builtin')
    parser.add_argument('--print-rank', type=int, default=0)

    # 速度测试相关参数
    parser.add_argument('--batch-size', '-b', default=1, type=int, help='batch size for speed test')
    parser.add_argument('--img-size', '-s', default=640, type=int, help='input resolution (H=W)')
    parser.add_argument('--warmup', default=10, type=int, help='number of warmup iterations')
    parser.add_argument('--iters', default=100, type=int, help='number of benchmarking iterations')

    # 是否打印 FLOPs（默认打开，用 --no-flops 关掉也行）
    parser.add_argument('--print-flops', default=True, help='print FLOPs via thop if installed')

    args = parser.parse_args()
    return args


def build_cfg_and_model(args):
    """
    使用 YAMLConfig 的方式构建 cfg 和 model
    等价于训练脚本里的：
        update_dict = yaml_utils.parse_cli(args.update)
        cfg = YAMLConfig(args.config, **update_dict)
        model = cfg.model
    """
    # 解析 CLI 更新（和原训练脚本保持一致）
    update_dict = yaml_utils.parse_cli(args.update)

    # 只把跟 cfg 有关、且 YAMLConfig 会用到的字段塞进去，避免测速参数污染 cfg
    for k in ['device', 'seed', 'use_amp', 'output_dir', 'summary_dir', 'resume', 'tuning']:
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None:
                update_dict[k] = v

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    model = cfg.model  # 和 BaseSolver._setup 里的 self.model = cfg.model 一样

    # 如果你想加载 checkpoint（可选）
    if args.resume:
        print(f'[Speed] Load checkpoint from {args.resume}')
        state = torch.load(args.resume, map_location='cpu')
        # 兼容 train 的结构：优先用 ema.module，其次 model
        if 'ema' in state and 'module' in state['ema']:
            ckpt = state['ema']['module']
        elif 'model' in state:
            ckpt = state['model']
        else:
            ckpt = state
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(
            f'[Speed] load_state_dict strict=False, '
            f'missing={len(missing)}, unexpected={len(unexpected)}'
        )

    return cfg, model


def main():
    args = parse_args()

    # 简单单卡测就行
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # -------- 1. 构建 cfg 与 model ----------
    cfg, model = build_cfg_and_model(args)

    # -------- 2. 设备 & CUDNN 优化 ----------
    if getattr(cfg, "device", None):
        device = torch.device(cfg.device)
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model.eval()

    # -------- 3. 构造随机双路输入（先在 device 上建好，供 FLOPs & 速度测试复用） ----------
    B = args.batch_size
    C = 3
    H = W = args.img_size

    dummy0 = torch.randn(B, C, H, W, device=device)
    dummy1 = torch.randn(B, C, H, W, device=device)

    print("\n=========== Speed Benchmark Config ===========")
    print(f"Config file    : {args.config}")
    print(f"Device         : {device}")
    print(f"Batch size     : {B}")
    print(f"Image size     : {H}x{W}")
    print(f"Warmup iters   : {args.warmup}")
    print(f"Benchmark iters: {args.iters}")
    print(f"Use AMP        : {args.use_amp}")
    print(f"Print FLOPs    : {args.print_flops}")
    print("==============================================\n")

    # -------- 4. 参数量统计（与 thop 无关，纯 Python 统计） ----------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("---------- Parameter Count ----------")
    print(f"[Params - total    ] {total_params / 1e6:.3f} M")
    print(f"[Params - trainable] {trainable_params / 1e6:.3f} M")
    print("-------------------------------------\n")

    # -------- 5. 可选：用 thop 估算 FLOPs （在未 compile 的模型上） ----------
    if args.print_flops:
        try:
            from thop import profile

            # 深拷贝一份模型出来专门给 thop 用，避免挂钩子污染主模型
            model_for_profile = copy.deepcopy(model)
            model_for_profile.to(device)
            model_for_profile.eval()

            # 如果之前已经 profile 过（比如在别的脚本里），清理掉遗留的属性
            for m in model_for_profile.modules():
                for attr in ("total_ops", "total_params"):
                    if hasattr(m, attr):
                        delattr(m, attr)

            with torch.no_grad():
                flops, params_thop = profile(
                    model_for_profile,
                    inputs=(dummy0, dummy1),
                    verbose=False
                )

            # 当前 batch 一次 forward 的 FLOPs
            flops_per_forward = flops
            flops_per_image   = flops / float(B)

            print("-------------- FLOPs (thop) --------------")
            print(f"[FLOPs / forward (batch={B})] {flops_per_forward / 1e9:.3f} GFLOPs")
            print(f"[FLOPs / image             ] {flops_per_image   / 1e9:.3f} GFLOPs")
            print(f"[Params (thop)             ] {params_thop      / 1e6:.3f} M")
            print("------------------------------------------\n")

            # 用完就丢掉，释放显存
            del model_for_profile
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] thop profile failed: {e}\n")

    # -------- 6. 速度测试用模型：先搬到 device，再尝试 compile ----------
    model.to(device)
    model.eval()

    #✅ compile 优化 (PyTorch >= 2.0)
    try:
        model = torch.compile(model, backend="inductor", mode="default")
        model = torch.compile(model)
        print("[compile] Model compiled successfully with torch.compile().")
    except Exception as e:
        print(f"[compile] Skip compile: {e}")

    # -------- 7. 预热 ----------
    warmup = args.warmup
    iters = args.iters
    use_amp = args.use_amp

    print("> Warmup ...")
    with torch.no_grad():
        for i in range(warmup):
            if use_amp:
                # 通用写法（torch 1.x/2.x 都兼容）
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model(dummy0, dummy1)
            else:
                _ = model(dummy0, dummy1)
    torch.cuda.synchronize()
    print("> Warmup done.\n")

    # -------- 8. 正式测速 ----------
    total_time = 0.0
    print("> Benchmarking ...")
    with torch.no_grad():
        for i in range(iters):
            torch.cuda.synchronize()
            t1 = time.time()

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model(dummy0, dummy1)
            else:
                _ = model(dummy0, dummy1)

            torch.cuda.synchronize()
            t2 = time.time()
            elapsed = t2 - t1
            total_time += elapsed

            print(f"[{i + 1}/{iters}] iter time: {elapsed * 1000:.3f} ms")
    print("> Benchmark done.\n")

    avg = total_time / max(1, iters)
    fps_img = B / avg

    print("================ Benchmark Result ================")
    print(f"Batch size           : {B}")
    print(f"Avg time per forward : {avg * 1000:.3f} ms")
    print(f"FPS (image-level)    : {fps_img:.2f} images/s")
    print("==================================================")

    dist_utils.cleanup()


if __name__ == "__main__":
    main()
