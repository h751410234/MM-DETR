"""
Multimodal (RGB + IR) inference & visualization script.

- 与 train_one_epoch_multimodal 保持一致的前向接口:
    outputs = model(samples0, samples1)    # 推理时不再传 targets
- 根据 pair_mode 从 RGB 图像目录自动找到 IR 图像
- 在 RGB 图像和 IR 图像上分别画检测框并保存
"""

import os
import sys
import argparse

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as tv_T

# 让脚本能找到 src/*
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import YAMLConfig

# ================== 颜色配置 ==================
# 基础颜色表（按索引定义一组足够多的互相区分的颜色）
colors = [
    (0.9020, 0.5412, 0.4902),  # 0 橙
    (0.3804, 0.5216, 0.4863),  # 1 深绿
    (0.56, 0.56, 0.56),        # 2 灰
    (0.8314, 0.5686, 0.8510),  # 3 紫
    (0.6392, 0.8078, 0.7765),  # 4 浅绿
    (1.0, 1.0, 0.0),           # 5 黄
    (0.5647, 0.6902, 0.8157),  # 6 浅蓝
    (0.3804, 0.2216, 0.8863),  # 7 红偏蓝
    (0.8, 0.4, 0.0),           # 8 深橙 / 棕橙
    (0.4, 0.2, 0.0),           # 9 棕
    (0.2, 0.6, 0.2),           # 10 鲜绿
    (0.3, 0.3, 0.3),           # 11 深灰
    (0.9, 0.8, 0.4),           # 12 金黄
    (0.2, 0.4, 0.8),           # 13 深蓝
]

INT_COLORS = [[int(i * 255) for i in c] for c in colors]

# 语义类别名字 -> 固定颜色
# 注意：key 一律用小写；后面 main 里用 name.lower() 去匹配
NAME_COLOR_TABLE = {
    # --- 多数据集公共类 ---
    "person":      INT_COLORS[0],   # 橙色
    "car":         INT_COLORS[1],   # 深绿
    "bicycle":     INT_COLORS[2],   # 灰
    "bus":         INT_COLORS[3],   # 紫
    "truck":       INT_COLORS[4],   # 浅绿

    # --- VEDAI 专有类 ---
    "pickup":      INT_COLORS[6],   # 浅蓝
    "camping":     INT_COLORS[10],  # 鲜绿
    "other":       INT_COLORS[11],  # 深灰
    "tractor":     INT_COLORS[9],   # 棕
    "boat":        INT_COLORS[8],   # 深橙 / 棕橙
    "van":         INT_COLORS[7],   # 红偏蓝

    # --- M3FD 特有 ---
    "lamp":        INT_COLORS[12],  # 金黄
    "motorcycle":  INT_COLORS[5],   # 黄（注意：M3FD 里是 "Motorcycle"，后面会统一 lower）

    # --- DV 特有 ---
    "freight_car": INT_COLORS[13],  # 深蓝
}


# ================== 可视化函数（xyxy像素坐标） ==================

def visualize_and_save_xyxy(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    save_path: str,
    label_list=None,
    score_thr: float = 0.3,
    color_map=None,
):
    """
    image: PIL.Image，已经是网络输入尺度 (例如 640x640)
    boxes: [N, 4], xyxy 像素坐标
    labels: [N]
    scores: [N]
    color_map: dict[int -> List[int]]，显式 类别id -> 颜色
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    boxes = boxes.cpu()
    labels = labels.cpu()
    scores = scores.cpu()

    keep = scores > score_thr
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    for box, cls_id, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        cls_id = int(cls_id)

        # ---- 同一类别固定同一颜色（优先用 color_map） ----
        if color_map is not None and cls_id in color_map:
            color = tuple(color_map[cls_id])
        else:
            color = tuple(INT_COLORS[cls_id % len(INT_COLORS)])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 如果需要在图上画文字，可以解开下面注释：
        # if label_list is not None and 0 <= cls_id < len(label_list):
        #     cls_name = label_list[cls_id]
        # else:
        #     cls_name = str(cls_id)
        # text = f"{cls_name} {score:.2f}"
        # tw, th = draw.textsize(text, font=font)
        # text_bg = [x1, max(0, y1 - th), x1 + tw, y1]
        # draw.rectangle(text_bg, fill=color)
        # draw.text((x1, max(0, y1 - th)), text, fill="black", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print(f"Saved: {save_path}")


# ================== 路径映射（和 CocoDetection_Multimodal 一致） ==================

def _default_mapper_vedai(root_rgb: str, file_rgb: str):
    root_ir = root_rgb.replace("images_co", "images_ir")
    ir_name = file_rgb.replace("_co", "_ir")
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)


def _default_mapper_flir(root_rgb: str, file_rgb: str):
    root_ir = root_rgb.replace("visible", "infrared")
    ir_name = file_rgb.replace(".jpg", ".jpeg")
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)


def _default_mapper_m3fd(root_rgb: str, file_rgb: str):
    root_ir = root_rgb.replace("images_vi", "images_ir")
    ir_name = file_rgb
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)


def _default_mapper_dv(root_rgb: str, file_rgb: str):
    root_ir = root_rgb.replace("testimg", "testimgr")
    ir_name = file_rgb
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)


def resolve_pair_paths(rgb_root: str, file_rgb: str, pair_mode: str):
    if pair_mode == "vedai":
        rgb_path, ir_path = _default_mapper_vedai(rgb_root, file_rgb)
    elif pair_mode == "flir":
        rgb_path, ir_path = _default_mapper_flir(rgb_root, file_rgb)
    elif pair_mode == "m3fd":
        rgb_path, ir_path = _default_mapper_m3fd(rgb_root, file_rgb)
    elif pair_mode == "dv":
        rgb_path, ir_path = _default_mapper_dv(rgb_root, file_rgb)
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"[infer_multimodal] 未找到可见光图像：{rgb_path}")
    if not os.path.isfile(ir_path):
        raise FileNotFoundError(f"[infer_multimodal] 未找到红外图像：{ir_path}")

    return rgb_path, ir_path


# ================== 模型加载（不合并通道） ==================

def load_model_from_yaml(config_path: str, ckpt_path: str, device: str = "cuda"):
    """
    用 YAMLConfig 加载模型，风格跟官方 train.py 一致。
    """
    cfg = YAMLConfig(config_path, resume=ckpt_path)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "ema" in ckpt:
            state = ckpt["ema"]["module"]
        else:
            state = ckpt["model"]
        cfg.model.load_state_dict(state)
    else:
        raise AttributeError("请通过 --resume 指定训练好的权重 (.pth)")

    model = cfg.model.to(device)
    model.eval()

    postprocessor = getattr(cfg, "postprocessor", None)

    return model, postprocessor, cfg


# ================== 前向：两路输入版 ==================

def forward_multimodal(model, rgb_tensor, ir_tensor):
    """
    与 train_one_epoch_multimodal 对齐：
        训练：outputs = model(samples0, samples1, targets=targets)
        推理：outputs = model(samples0, samples1)

    rgb_tensor: [B, 3, H, W]
    ir_tensor:  [B, 3, H, W]
    """
    outputs = model(rgb_tensor, ir_tensor)
    return outputs


# ================== 主推理流程 ==================

def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    model, postprocessor, cfg = load_model_from_yaml(args.config, args.resume, device)

    img_size = args.img_size
    t_resize = tv_T.Resize((img_size, img_size))
    t_to_tensor = tv_T.ToTensor()

    # 类别名：从命令行传进来，可能随数据集改变顺序
    label_list = None
    if args.class_names is not None:
        label_list = [s.strip() for s in args.class_names.split(",") if s.strip()]

    # ===== 类别 id -> 颜色映射（按“名字”对齐不同数据集） =====
    color_map = {}
    if label_list is not None:
        for cls_id, name in enumerate(label_list):
            key = name.lower()
            if key in NAME_COLOR_TABLE:
                color_map[cls_id] = NAME_COLOR_TABLE[key]
            else:
                # 没在表里的类别，用一个稳定的 fallback 颜色
                color_map[cls_id] = INT_COLORS[cls_id % len(INT_COLORS)]
    else:
        for cls_id in range(len(INT_COLORS)):
            color_map[cls_id] = INT_COLORS[cls_id]

    rgb_root = args.rgb_dir
    img_files = [
        f for f in os.listdir(rgb_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    img_files.sort()

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, file_rgb in enumerate(img_files):
            print(f"[{i+1}/{len(img_files)}] {file_rgb}")
            rgb_path, ir_path = resolve_pair_paths(rgb_root, file_rgb, args.pair_mode)

            # 加载两张图
            img_rgb = Image.open(rgb_path).convert("RGB")
            img_ir = Image.open(ir_path).convert("RGB")

            # 与训练一致：先 resize 再转 tensor
            img_rgb_resized = t_resize(img_rgb)
            img_ir_resized = t_resize(img_ir)

            rgb_tensor = t_to_tensor(img_rgb_resized).unsqueeze(0).to(device)  # [1,3,H,W]
            ir_tensor = t_to_tensor(img_ir_resized).unsqueeze(0).to(device)   # [1,3,H,W]

            # 前向：两路输入
            outputs = forward_multimodal(model, rgb_tensor, ir_tensor)

            if postprocessor is None:
                raise NotImplementedError("cfg.postprocessor 未定义，请根据模型输出自行 decode")

            h, w = rgb_tensor.shape[-2:]
            size = torch.tensor([[h, w]], device=device)

            # RTDETRPostProcessor 通常返回 list[dict] / tuple[dict,...]
            post_out = postprocessor(outputs, size)[0]

            labels = post_out["labels"]
            boxes = post_out["boxes"]
            scores = post_out["scores"]

            # 为了保证两张图都各自画一遍，先 copy 一下
            rgb_vis = img_rgb_resized.copy()
            ir_vis = img_ir_resized.copy()

            # 分别在 RGB 和 IR 图上可视化
            rgb_save_path = os.path.join(args.output_dir, "rgb", file_rgb)
            ir_save_path = os.path.join(args.output_dir, "ir", file_rgb)

            visualize_and_save_xyxy(
                rgb_vis,
                boxes,
                labels,
                scores,
                rgb_save_path,
                label_list=label_list,
                score_thr=args.score_thr,
                color_map=color_map,
            )

            visualize_and_save_xyxy(
                ir_vis,
                boxes,
                labels,
                scores,
                ir_save_path,
                label_list=label_list,
                score_thr=args.score_thr,
                color_map=color_map,
            )


# ================== CLI ==================

def get_args():
    parser = argparse.ArgumentParser("Multimodal detection inference & visualization")

    parser.add_argument(
        "-c", "--config",
        type=str,
        default='configs/VEDAI/rtdetrv2_r50vd_6x_coco.yml',
        help="YAML config 文件路径"
    )
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default='',
        help="训练好的 checkpoint 路径 (.pth)"
    )
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default='',
        help="可见光图像根目录（例如 DV 的 img）"
    )
    parser.add_argument(
        "--pair-mode",
        type=str,
        default="vedai",
        choices=["dv", "vedai", "flir", "m3fd"],
        help="多模态路径命名模式，需要和 CocoDetection_Multimodal 保持一致"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vedai",
        help="可视化结果输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="推理设备，如 'cuda:0' 或 'cpu'"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="统一 resize 到的输入尺寸"
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="可视化阈值"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default='car,pickup,camping,truck,other,tractor,boat,van',
        help="类别名字符串，如 'person,car,train'；留空则显示 id"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)


#类别排序,方便可视化
#vedai: car,pickup,camping,truck,other,tractor,boat,van
#flir:person,car,bicycle
#m3fd:person,car,bus,lamp,Motorcycle,truck
#dv:bus,car,freight_car,truck,van


