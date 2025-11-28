"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from faster_coco_eval.utils.pytorch import FasterCocoDetection
import torchvision

from PIL import Image 
from faster_coco_eval.core import mask as coco_mask

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register
from typing import Tuple, List, Any
import os

__all__ = ['CocoDetection','CocoDetection_Multimodal']

torchvision.disable_beta_transforms_warning()

@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
            # image, target = self.prepare(image, target, category2label=self.category2label)
        else:
            image, target = self.prepare(image, target)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        
        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s 

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}

#--------------多模态增添---------------

def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _open_as_rgb(path: str) -> Image.Image:
    # 即使是红外，后续增强通常按 RGB 处理，因此统一转 RGB
    return Image.open(path).convert("RGB")

def _default_mapper_vedai(root_rgb: str, file_rgb: str) -> Tuple[str, str]:
    """VEDAI 命名示例：
    rgb 根目录: .../images_co
    ir  根目录: .../images_ir
    文件：xxx_co.png -> 对应 xxx_ir.png
    """
    root_ir = root_rgb.replace("images_co", "images_ir")
    ir_name = file_rgb.replace("_co", "_ir")
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)

def _default_mapper_flir(root_rgb: str, file_rgb: str) -> Tuple[str, str]:
    """VEDAI 命名示例：
    rgb 根目录: .../images_co
    ir  根目录: .../images_ir
    文件：xxx_co.png -> 对应 xxx_ir.png
    """
    root_ir = root_rgb.replace("visible", "infrared")
    ir_name = file_rgb.replace(".jpg", ".jpeg")
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)

def _default_mapper_m3fd(root_rgb: str, file_rgb: str) -> Tuple[str, str]:
    """VEDAI 命名示例：
    rgb 根目录: .../images_co
    ir  根目录: .../images_ir
    文件：xxx_co.png -> 对应 xxx_ir.png
    """
    root_ir = root_rgb.replace("images_vi", "images_ir")
    ir_name = file_rgb
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)

def _default_mapper_dv(root_rgb: str, file_rgb: str) -> Tuple[str, str]:
    """VEDAI 命名示例：
    rgb 根目录: .../images_co
    ir  根目录: .../images_ir
    文件：xxx_co.png -> 对应 xxx_ir.png
    """
    root_ir = root_rgb.replace("img", "imgr")
    ir_name = file_rgb
    return os.path.join(root_rgb, file_rgb), os.path.join(root_ir, ir_name)


@register()
class CocoDetection_Multimodal(FasterCocoDetection, DetDataset):
    ### 多模态 COCO 数据集：返回 (img0, img1, target)
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False,pair_mode: str = "dv",):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.pair_mode = pair_mode

    def __getitem__(self, idx):
        img0, img1 , target = self.load_item(idx)
        if self._transforms is not None:
            (img0,img1), target, _ = self._transforms((img0,img1), target, self)  #多模态图像同步处理

        return img0,img1,target

    # ---------- 标签基础加载 ----------
    def _load_target(self, img_id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(img_id))

    # ---------- 图像路径映射 ----------
    def _resolve_pair_paths(self, img_id: int) -> Tuple[str, str]:
        file_rgb = self.coco.loadImgs(img_id)[0]["file_name"]
        root_rgb = self.root  # VisionDataset 将 img_folder 赋给 self.root
        if self.pair_mode == "vedai":
            rgb_path, ir_path = _default_mapper_vedai(root_rgb, file_rgb)

        if self.pair_mode == "flir":
            rgb_path, ir_path = _default_mapper_flir(root_rgb, file_rgb)

        if self.pair_mode == "m3fd":
            rgb_path, ir_path = _default_mapper_m3fd(root_rgb, file_rgb)

        if self.pair_mode == "dv":
            rgb_path, ir_path = _default_mapper_dv(root_rgb, file_rgb)


        if not os.path.isfile(rgb_path):
            raise FileNotFoundError(f"[CocoDetection_Multimodal] 未找到可见光图像：{rgb_path}")
        if not os.path.isfile(ir_path):
            raise FileNotFoundError(f"[CocoDetection_Multimodal] 未找到红外/第二模态图像：{ir_path}")
        return rgb_path, ir_path

    # ---------- 加载双路图像 ----------
    def _load_images_pair(self, img_id: int) -> Tuple[Image.Image, Image.Image]:
        rgb_path, ir_path = self._resolve_pair_paths(img_id)
        img0 = _open_rgb(rgb_path)
        img1 = _open_as_rgb(ir_path)
        return img0, img1


    def load_item(self, idx):
        image_id = self.ids[idx]
        img0, img1 = self._load_images_pair(image_id)

        target = self._load_target(image_id)
        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(img0, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(img0, target)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return img0, img1, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
            
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
