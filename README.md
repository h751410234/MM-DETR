# MM-DETR: An Efficient Multimodal Detection Transformer with Mamba-Driven Dual-Granularity Fusion and Frequency-Aware Modality Modeling

**Authors:** Jianhong Han, Yupei Wang, Yuan Zhang, and Liang Chen

This repository contains the official implementation of:

**MM-DETR: An Efficient Multimodal Detection Transformer with Mamba-Driven Dual-Granularity Fusion and Frequency-Aware Modality Modeling**

If this work benefits your research, please consider citing our paper.

---

![](/figs/Figure1.png)

---

## Acknowledgment
This implementation is built upon the excellent open-source project  
➡️ https://github.com/lyuwenyu/RT-DETR/

---

## Installation

See **requirements.txt** for environment setup.  
Our experimental environment:

- **OS:** Ubuntu 16.04  
- **Python:** 3.10.9  
- **CUDA:** 11.8  
- **PyTorch:** 2.0.1+

---

## Dataset Preparation

1. Download datasets from official sources.  
2. Convert all annotations to COCO-format JSON.  
3. Modify dataset paths in their corresponding configuration scripts. Example (VEDAI):  
   `configs/dataset/coco_detection_VEDAI.yml`

```yaml
# -- Training Configuration
train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection_Multimodal
    img_folder: images_co            # Visible (RGB) image directory
    ann_file: train_co.json          # COCO-format multimodal annotation file
    pair_mode: vedai                 # Dataset keyword; auto-matches the paired IR image for each RGB input
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFuncion_Multimodal

# -- Validation / Testing Configuration
val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection_Multimodal
    img_folder: images_co
    ann_file: test_co.json
    pair_mode: vedai
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFuncion_Multimodal
```

---

## Training / Evaluation / Inference

### Training  
Training scripts will be released after the paper is officially accepted.

### Inference & Visualization  

```bash
python inference.py
```

See: **infer_img.py**

### FLOPs / Model Size / Inference Time  

```bash
python infer_time.py
```

See: **infer_time.py**

---

## Pre-trained Models

| Task            | mAP50  | Config | Model |
|-----------------|--------|--------|--------|
| **VEDAI**       | 87.06% | configs/VEDAI/rtdetrv2_r50vd_6x_coco.yml | https://pan.baidu.com/s/1ut-KXgBUTMMUgto-CWV30A?pwd=gqeu |
| **M3FD**        | 73.39% | configs/M3FD/rtdetrv2_r50vd_6x_coco.yml | https://pan.baidu.com/s/1ut-KXgBUTMMUgto-CWV30A?pwd=gqeu |
| **FLIR**        | 83.59% | configs/FLIR/rtdetrv2_r50vd_6x_coco.yml | https://pan.baidu.com/s/1ut-KXgBUTMMUgto-CWV30A?pwd=gqeu |
| **DroneVehicle**| 82.31% | configs/DV/rtdetrv2_r50vd_6x_coco.yml | https://pan.baidu.com/s/1ut-KXgBUTMMUgto-CWV30A?pwd=gqeu |

---

## Reference

RT-DETR: https://github.com/lyuwenyu/RT-DETR/
