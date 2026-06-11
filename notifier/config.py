# -*- coding: utf-8 -*-
"""
配置加载模块

支持 YAML 配置文件，定义摄像头参数和 webhook 通知规则。
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class WebhookConfig:
    """单个 webhook 配置"""
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    body_template: Dict = field(default_factory=dict)
    events: List[str] = field(default_factory=lambda: ["in", "out"])
    cooldown_seconds: float = 0.0  # 同一摄像头同一 webhook 的冷却时间


@dataclass
class CameraConfig:
    """单个摄像头配置"""
    name: str
    rtsp_url: str
    server: str = "172.16.20.193:50052"
    line: str = ""           # 检测线 x1,y1,x2,y2（以此为中心向两侧扩展成双线）
    gate_width: int = 60     # 双线间距（像素），原线向法向量两侧各扩展 gate_width/2
    detect_model: str = "./infer/detect/model.plan"
    model_type: str = "yolo"   # "yolo" 或 "yoloe"
    gpu_id: int = 0
    reverse: bool = False
    decoder: int = 1
    interval_ms: int = 0
    event_dir: Optional[str] = None


@dataclass
class AppConfig:
    """应用总配置"""
    cameras: List[CameraConfig] = field(default_factory=list)
    webhooks: List[WebhookConfig] = field(default_factory=list)


def load_config(path: str = "config.yaml") -> AppConfig:
    """从 YAML 文件加载配置"""
    if yaml is None:
        raise ImportError("请安装 PyYAML: pip install pyyaml")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cameras = []
    for c in data.get("cameras", []):
        cameras.append(CameraConfig(**c))

    webhooks = []
    for w in data.get("webhooks", []):
        webhooks.append(WebhookConfig(**w))

    return AppConfig(cameras=cameras, webhooks=webhooks)
