# -*- coding:utf-8 -*-

"""
    YOLOE 后处理

    适配单输入 ONNX 输出 output0: float32 [1, 300, 38]
    38 维向量分解：
        - 0~3:  bounding box (x1, y1, x2, y2)
        - 4:    confidence
        - 5:    nc_id（类别ID）
        - 6~37: mask coefficients (32 dim, 忽略)

    后处理步骤（无 NMS）：
        1. 解析 output0 -> boxes / conf / nc_id
        2. 按 conf_thres 过滤
        3. 将 box 坐标从 letterbox 尺寸(640x640) 缩放到原图尺寸
        4. 按 classes 过滤类别
"""

import numpy as np


def clip_coords(boxes, img_shape):
    """将框坐标裁剪到图像边界内"""
    boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def postprocess(img0, output0, conf_thres, classes=None):
    """
    YOLOE 后处理（无 NMS）

    :param img0: 原图 (numpy array)
    :param output0: [1, 300, 38] float32，模型原始输出
    :param conf_thres: 置信度阈值
    :param classes: 指定保留的类别列表，如 [0] 只保留 person
    :return: [N, 6] 的检测框，每行为 [x1, y1, x2, y2, conf, class_id]
    """
    # 去除 batch 维度
    if output0.ndim == 3 and output0.shape[0] == 1:
        output0 = output0[0]  # [300, 38]

    # 解析输出
    boxes = output0[:, 0:4]      # [300, 4]  xyxy
    conf = output0[:, 4]         # [300]     最终置信度
    nc_id = output0[:, 5].astype(np.int64)  # [300] 类别ID
    # mask_coeff = output0[:, 6:38]  # 不需要

    # 1. 按置信度过滤
    valid_mask = conf > conf_thres
    boxes = boxes[valid_mask]
    conf = conf[valid_mask]
    nc_id = nc_id[valid_mask]

    if boxes.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    # 如果 boxes 是归一化坐标(0~1)，先放大到 640x640
    if boxes.max() <= 1.5 and boxes.min() >= -0.5:
        boxes = boxes * 640.0

    # 2. 将坐标从 letterbox(640x640) 缩放到原图
    boxes = scale_coords((640, 640), boxes.copy(), img0.shape[:2])

    # 3. 组装结果 [x1, y1, x2, y2, conf, class_id]
    detections = np.concatenate([
        boxes,
        conf[:, None],
        nc_id[:, None].astype(np.float32)
    ], axis=1)

    # 4. 按 classes 过滤类别
    if classes is not None:
        mask = np.isin(detections[:, 5].astype(int), classes)
        detections = detections[mask]

    return detections
