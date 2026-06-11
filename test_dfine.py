#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-FINE 模型稳定性测试脚本

用法:
    python test_dfine.py --image debug/entrance_1.jpg --runs 5
    python test_dfine.py --image events/entrance_144659_OUT_1.jpg --runs 10 --save-vis

功能:
    1. 对同一张图片多次推理，检测输出是否一致
    2. 打印原始输出 tensor（labels/boxes/scores）的统计信息
    3. 对比后处理前的原始输出，定位不稳定来源
    4. 保存可视化结果供人工检查
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infer.dfine.infer import DfineDetector
from infer.dfine.preprocess import preprocess
from infer.dfine.postprocess import postprocess


def parse_args():
    parser = argparse.ArgumentParser(description="D-FINE 稳定性测试")
    parser.add_argument("--image", type=str, default="debug/entrance_1.jpg",
                        help="测试图片路径")
    parser.add_argument("--runs", type=int, default=5,
                        help="重复推理次数")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45,
                        help="NMS IoU 阈值")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="保留的类别，默认只检测 person")
    parser.add_argument("--plan", type=str,
                        default="models/trt_models/dfine_s_obj2coco_sim.plan",
                        help="TensorRT plan 文件路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--save-raw", action="store_true",
                        help="保存每次推理的原始输出 tensor 到 npy 文件")
    parser.add_argument("--save-vis", action="store_true",
                        help="保存每次推理的可视化结果图片")
    parser.add_argument("--vis-dir", type=str, default="debug/dfine_test",
                        help="可视化结果保存目录")
    return parser.parse_args()


def visualize(image, detections, save_path=None):
    """在图像上绘制检测框"""
    vis = image.copy()
    class_name_list = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    for x1, y1, x2, y2, conf, class_id in detections:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name_list[int(class_id)]}:{conf:.3f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
    return vis


def compare_raw_outputs(outputs_list):
    """对比多次推理的原始输出 tensor"""
    print("\n[Raw Output Comparison]")
    first = outputs_list[0]
    all_identical = True

    for key in first.keys():
        print(f"  Tensor: {key}")
        ref = first[key]
        print(f"    Shape: {ref.shape}, Dtype: {ref.dtype}")
        print(f"    First run stats: min={ref.min():.6f}, max={ref.max():.6f}, mean={ref.mean():.6f}")

        for i in range(1, len(outputs_list)):
            curr = outputs_list[i][key]
            if ref.shape != curr.shape:
                print(f"    RUN {i+1}: SHAPE MISMATCH! {ref.shape} vs {curr.shape}")
                all_identical = False
                continue
            if not np.allclose(ref, curr, rtol=1e-5, atol=1e-6):
                diff = np.abs(ref - curr)
                print(f"    RUN {i+1}: VALUES DIFFER! max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
                # 打印前 10 个不同位置的索引
                mismatch_idx = np.where(diff > 1e-5)
                print(f"      Mismatch indices (first 10): {list(zip(*mismatch_idx))[:10]}")
                all_identical = False
            else:
                print(f"    RUN {i+1}: identical")

    return all_identical


def compare_detections(dets_list):
    """对比多次推理的后处理结果"""
    print("\n[Post-processed Detection Comparison]")
    first = dets_list[0]
    print(f"  First run: {len(first)} detections")
    if len(first) > 0:
        print(f"    First 3 boxes:\n{first[:3]}")

    all_identical = True
    for i in range(1, len(dets_list)):
        curr = dets_list[i]
        print(f"  Run {i+1}: {len(curr)} detections")
        if len(first) != len(curr):
            print(f"    DETECTION COUNT MISMATCH: {len(first)} vs {len(curr)}")
            all_identical = False
            continue
        if len(first) == 0:
            print(f"    identical (empty)")
            continue
        if not np.allclose(first, curr, rtol=1e-5, atol=1e-4):
            diff = np.abs(first - curr)
            print(f"    BOXES DIFFER! max_diff={diff.max():.6f}")
            # 按行比较，找出差异最大的检测框
            row_diff = np.linalg.norm(diff[:, :4], axis=1)
            worst_idx = np.argmax(row_diff)
            print(f"    Worst mismatch at detection #{worst_idx}:")
            print(f"      First:  {first[worst_idx]}")
            print(f"      Run {i+1}:  {curr[worst_idx]}")
            all_identical = False
        else:
            print(f"    identical")

    return all_identical


def test_preprocess_stability(image, runs=10):
    """测试预处理是否稳定"""
    print("\n[Preprocess Stability Test]")
    ref = preprocess(image, 640, 640)
    for i in range(runs):
        curr = preprocess(image, 640, 640)
        if not np.allclose(ref, curr):
            print(f"  PREPROCESS UNSTABLE at run {i+1}!")
            return False
    print(f"  Preprocess is stable over {runs} runs.")
    return True


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        return

    # 读取图片
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: failed to read image: {args.image}")
        return
    print(f"Test image: {args.image}, shape: {image.shape}")

    # 测试预处理稳定性
    test_preprocess_stability(image, runs=args.runs)

    # 加载模型
    print(f"\nLoading TensorRT engine: {args.plan}")
    detector = DfineDetector(
        trt_plan=args.plan,
        gpu_id=args.gpu,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
    )
    print("Model loaded.")

    # 预热（可选）
    print("\nWarm-up inference...")
    _ = detector.inference(image, classes=args.classes)
    time.sleep(0.1)

    # 多次推理
    print(f"\nRunning inference {args.runs} times...")
    raw_outputs_list = []
    detections_list = []
    times = []

    # 为了获取原始输出，我们需要临时修改 inference_one 或直接调用底层 API
    # 这里我们先做标准推理，再做一次底层推理获取原始 tensor
    for i in range(args.runs):
        t0 = time.time()
        detections = detector.inference(image, classes=args.classes)
        t1 = time.time()
        times.append(t1 - t0)
        detections_list.append(detections.copy())
        print(f"  Run {i+1}: {len(detections)} detections, time={times[-1]*1000:.2f}ms")

    # 获取原始输出（通过手动调用 inference_one）
    print("\nFetching raw outputs for comparison...")
    input_data = preprocess(image, 640, 640)
    input_data = np.expand_dims(input_data, axis=0)
    orig_h, orig_w = image.shape[:2]
    orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)

    for i in range(args.runs):
        outputs = detector.inference_one(input_data, orig_size)
        raw_outputs_list.append({k: v.copy() for k, v in outputs.items()})
        if args.save_raw:
            for k, v in outputs.items():
                fname = f"debug/dfine_raw/run{i+1}_{k}.npy"
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                np.save(fname, v)

    # 对比原始输出
    raw_identical = compare_raw_outputs(raw_outputs_list)

    # 对比后处理结果
    post_identical = compare_detections(detections_list)

    # 保存可视化
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        for i, dets in enumerate(detections_list):
            path = os.path.join(args.vis_dir, f"run_{i+1:02d}.jpg")
            visualize(image, dets, save_path=path)
        print(f"\nVisualization saved to: {args.vis_dir}")

    # 总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Raw outputs identical:    {raw_identical}")
    print(f"Post-process identical:   {post_identical}")
    print(f"Avg inference time:       {np.mean(times)*1000:.2f} ms")
    print(f"Std inference time:       {np.std(times)*1000:.2f} ms")

    if not raw_identical:
        print("\n[DIAGNOSIS] 原始输出 tensor 就不一致 -> 问题出在 TensorRT 推理层")
        print("  可能原因:")
        print("    - TensorRT engine 使用了非确定性 kernel（FP16/INT8 尤其常见）")
        print("    - engine 文件损坏或每次被重建")
        print("    - GPU 显存/计算存在数据竞争（多个线程共享 detector）")
        print("    - 输入数据在拷贝到 device 时被污染")
    elif not post_identical:
        print("\n[DIAGNOSIS] 原始输出一致，但后处理结果不同 -> 问题出在后处理")
        print("  可能原因:")
        print("    - NMS 中 argsort 使用了不稳定的 quicksort（默认 kind='quicksort'）")
        print("    - 存在浮点数 tie-breaking 导致 NMS 保留的框不同")
        print("    - Python set 遍历顺序导致类别处理顺序不同（Python 3.7+ 应稳定）")
    else:
        print("\n[DIAGNOSIS] 所有结果完全一致，未检测到不稳定性")
        print("  如果实际运行时仍不稳定，请检查:")
        print("    - 是否多个摄像头线程共享了同一个 DfineDetector 实例")
        print("    - 输入图片是否确实完全相同（视频流解码差异）")

    detector.release()


if __name__ == "__main__":
    main()
