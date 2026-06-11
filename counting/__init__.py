# -*- coding: utf-8 -*-
"""
counting - 人流量统计模块

包含:
    - counter.py : LineCounter / PersonCounter (基于检测线的计数核心)
    - main.py    : 命令行入口 (SHM + YOLO11 + ByteTrack + 计数)
"""

from .counter import LineCounter, PersonCounter

__all__ = ["LineCounter", "PersonCounter"]
