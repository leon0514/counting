# -*- coding: utf-8 -*-
"""
人流事件 Webhook 通知器 —— 独立运行入口

用法:
    python notifier/main.py --config config.yaml

功能:
    - 从配置文件加载摄像头和 webhook
    - 检测到人进入/离开时自动发送 HTTP 通知
    - 支持多摄像头并发（每个摄像头独立线程）
"""

import os
import sys
import time
import threading
import argparse
from queue import Queue, Empty

# 项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loguru import logger

# 配置日志输出（文件 + 控制台）
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="00:00",           # 每天零点自动轮转
    retention="7 days",         # 保留最近 7 天
    encoding="utf-8",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)

import cv2
from notifier.config import load_config, CameraConfig
from notifier.webhook import WebhookNotifier
from infer.detect.infer import YoloDetector
from infer.yoloe.infer import YoloeDetector
from infer.tracker.byte_tracker import BYTETracker
from stream.shm_capture import ShmCapture, DECODER_GPU_NVCUVID, DECODER_CPU_FFMPEG
from counting.counter import DoubleLineCounter
import numpy as np

TRACK_CLASSES = [0]  # person


class TrackArgs:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10


def _parse_line(s: str):
    parts = [int(float(x.strip())) for x in s.split(",")]
    assert len(parts) == 4
    return (parts[0], parts[1]), (parts[2], parts[3])


def _build_double_lines(cfg: CameraConfig):
    """从单条配置线自动向法向量两侧扩展为双线"""
    start, end = _parse_line(cfg.line)
    line_vec = np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32)
    # 法向量（指向 in 方向）
    normal = np.array([-line_vec[1], line_vec[0]], dtype=np.float32)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        raise ValueError(f"camera '{cfg.name}': 检测线长度不能为零")
    normal = normal / norm

    half = cfg.gate_width / 2.0
    offset = normal * half
    # 外侧线（out）：法向量反方向偏移
    out_start = tuple((np.array(start) - offset).astype(float))
    out_end   = tuple((np.array(end)   - offset).astype(float))
    # 内侧线（in）：法向量正方向偏移
    in_start  = tuple((np.array(start) + offset).astype(float))
    in_end    = tuple((np.array(end)   + offset).astype(float))

    logger.info(f"[{cfg.name}] 配置线: {tuple(map(int, start))}-{tuple(map(int, end))}, "
                f"法向量: ({normal[0]:.3f},{normal[1]:.3f}), "
                f"gate_width={cfg.gate_width}px, "
                f"外线: {tuple(map(int, out_start))}-{tuple(map(int, out_end))}, "
                f"内线: {tuple(map(int, in_start))}-{tuple(map(int, in_end))}")
    return (out_start, out_end), (in_start, in_end)


def _io_worker(q: Queue, notifier: WebhookNotifier):
    """后台 IO 线程：消费队列执行图片保存和 webhook 发送"""
    while True:
        task = q.get()
        if task is None:
            break
        try:
            task_type = task.get("type")
            if task_type == "save_image":
                cv2.imwrite(task["path"], task["image"])
            elif task_type == "notify":
                notifier.notify(**task["kwargs"])
        except Exception as e:
            logger.error(f"[_io_worker] 任务失败: {e}")


def _camera_worker(cfg: CameraConfig, notifier: WebhookNotifier, io_queue: Queue):
    """单个摄像头的后台检测线程"""
    logger.info(f"[{cfg.name}] 启动检测...")

    # 1. 初始化检测器
    try:
        if cfg.model_type.lower() == "yoloe":
            detector = YoloeDetector(
                trt_plan=cfg.detect_model,
                gpu_id=cfg.gpu_id,
                nms_thresh=0.45,
                conf_thresh=0.2,
            )
            logger.info(f"[{cfg.name}] 加载 YOLOE 检测器: {cfg.detect_model}")
        else:
            detector = YoloDetector(
                trt_plan=cfg.detect_model,
                gpu_id=cfg.gpu_id,
                nms_thresh=0.45,
                conf_thresh=0.2,
            )
            logger.info(f"[{cfg.name}] 加载 YOLO 检测器: {cfg.detect_model}")
    except Exception as e:
        logger.error(f"[{cfg.name}] 检测器初始化失败: {e}")
        return

    # 2. 初始化跟踪器
    track_args = TrackArgs(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    tracker = BYTETracker(track_args, frame_rate=30)

    # 3. 从单条配置线自动计算双线
    (line_out_start, line_out_end), (line_in_start, line_in_end) = _build_double_lines(cfg)
    line_counter = DoubleLineCounter(
        (line_out_start, line_out_end), (line_in_start, line_in_end), name=cfg.name
    )

    # 4. 打开视频流
    cap = ShmCapture(server=cfg.server)
    decoder_type = DECODER_GPU_NVCUVID if cfg.decoder == 1 else DECODER_CPU_FFMPEG
    if not cap.open(cfg.rtsp_url, decoder_type=decoder_type, interval_ms=cfg.interval_ms):
        logger.error(f"[{cfg.name}] 打开流失败: {cfg.rtsp_url}")
        detector.release()
        return

    # 事件目录
    event_dir = cfg.event_dir
    if event_dir:
        os.makedirs(event_dir, exist_ok=True)

    num_frames = 0
    try:
        while True:
            ret, frame = cap.read(blocking=True, timeout_ms=5000)
            if not ret or frame is None:
                if not cap.isOpened():
                    logger.warning(f"[{cfg.name}] 流断开，重连中...")
                    cap.stop()
                    time.sleep(1)
                    if not cap.open(cfg.rtsp_url, decoder_type=decoder_type, interval_ms=cfg.interval_ms):
                        continue
                    logger.info(f"[{cfg.name}] 重连成功")
                continue

            num_frames += 1

            # 检测 + 画图 + 跟踪 + 计数
            detect_res = detector.inference(frame, classes=TRACK_CLASSES)
            for x1, y1, x2, y2, conf, class_id in detect_res:
                label = f"{class_id}:{conf:.2f}"
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

            online_targets = tracker.update(detect_res)
            events = line_counter.update(online_targets)

            # 发送 webhook / 保存事件图片 —— 异步投递到 IO 队列，不阻塞追踪
            if events:
                in_c, out_c, total_c = line_counter.get_counts()
                current = max(0, in_c - out_c)

                for ev in events:
                    direction = ev["direction"]
                    tid = ev["track_id"]

                    # 如果 reverse，反转方向用于显示/通知
                    notify_direction = direction
                    if cfg.reverse:
                        notify_direction = "out" if direction == "in" else "in"

                    image_path = None
                    if event_dir:
                        ts = time.strftime("%H%M%S_%f")[:-3]
                        suffix = "IN" if direction == "in" else "OUT"
                        image_path = os.path.join(event_dir, f"{cfg.name}_{ts}_{suffix}_{tid}.jpg")
                        vis_frame = frame.copy()
                        line_counter.draw_on_image(vis_frame, tracks=online_targets)
                        io_queue.put({
                            "type": "save_image",
                            "path": image_path,
                            "image": vis_frame,
                        })

                    io_queue.put({
                        "type": "notify",
                        "kwargs": {
                            "camera_name": cfg.name,
                            "direction": notify_direction,
                            "track_id": tid,
                            "current_count": current,
                            "image_path": image_path,
                        }
                    })
                    logger.info(f"[EVENT] {cfg.name} {notify_direction} track_id={tid} current={current}")

            # 每 30 帧打印一次状态
            if num_frames % 30 == 0:
                in_c, out_c, total_c = line_counter.get_counts()
                logger.info(f"[{cfg.name}] frame={num_frames} in={in_c} out={out_c} current={max(0, in_c - out_c)}")

    except KeyboardInterrupt:
        logger.info(f"[{cfg.name}] 收到中断")
    finally:
        cap.stop()
        detector.release()
        logger.info(f"[{cfg.name}] 停止")


def main():
    parser = argparse.ArgumentParser("人流事件 Webhook 通知器")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    if not config.cameras:
        logger.error("配置文件中没有摄像头")
        return
    if not config.webhooks:
        logger.warning("配置文件中没有 webhook，事件将只打印不发送")

    notifier = WebhookNotifier(config)
    logger.info(f"加载 {len(config.cameras)} 个摄像头, {len(config.webhooks)} 个 webhook")

    # 2. 启动全局 IO 后台线程（所有摄像头共用）
    io_queue = Queue()
    io_thread = threading.Thread(target=_io_worker, args=(io_queue, notifier), daemon=False)
    io_thread.start()

    # 3. 启动摄像头线程
    threads = []
    for cam_cfg in config.cameras:
        t = threading.Thread(target=_camera_worker, args=(cam_cfg, notifier, io_queue), daemon=True)
        t.start()
        threads.append(t)

    # 4. 主线程等待
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断，等待摄像头停止...")
        # 通知 IO 线程退出并等待完成
        io_queue.put(None)
        io_thread.join(timeout=10)
        logger.info("已退出")


if __name__ == "__main__":
    main()
