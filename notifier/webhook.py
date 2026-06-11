# -*- coding: utf-8 -*-
"""
Webhook 通知发送器

支持多 webhook、事件过滤、冷却时间、消息模板格式化。
"""

import time
import json
from typing import Dict, Any

from loguru import logger
from notifier.config import WebhookConfig, AppConfig

try:
    import requests
except ImportError:
    requests = None


class WebhookNotifier:
    """多 webhook 事件通知器"""

    def __init__(self, config: AppConfig):
        self.webhooks = config.webhooks
        self._last_send: Dict[str, float] = {}  # (url, camera) -> timestamp

    def notify(self, camera_name: str, direction: str, track_id: int,
               current_count: int, image_path: str = None) -> None:
        """
        发送进出事件通知

        :param camera_name: 摄像头名称
        :param direction: "in" 或 "out"
        :param track_id: 跟踪 ID
        :param current_count: 当前在馆人数
        :param image_path: 事件图片路径（可选）
        """
        if requests is None:
            logger.error("[Webhook] 请安装 requests: pip install requests")
            return

        now = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        for wh in self.webhooks:
            # 事件过滤
            if direction not in wh.events:
                continue

            # 冷却时间检查
            key = f"{wh.url}:{camera_name}"
            if now - self._last_send.get(key, 0) < wh.cooldown_seconds:
                continue
            self._last_send[key] = now

            # 构造消息体
            payload = self._format(wh.body_template, {
                "camera": camera_name,
                "event": direction,
                "timestamp": timestamp,
            })
            print(f"[Webhook] 发送通知 {wh.url} - {payload}")  # 调试输出
            # 发送
            try:
                resp = requests.request(
                    method=wh.method.upper(),
                    url=wh.url,
                    headers=wh.headers,
                    json=payload if isinstance(payload, dict) else None,
                    data=json.dumps(payload) if isinstance(payload, dict) else None,
                    timeout=5,
                )
                if resp.status_code >= 400:
                    logger.warning(f"[Webhook] {wh.url} 返回 {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.error(f"[Webhook] 发送失败 {wh.url}: {e}")

    @staticmethod
    def _format(template: Any, data: Dict[str, Any]) -> Any:
        """递归格式化模板中的字符串占位符"""
        if isinstance(template, str):
            return template.format(**data)
        if isinstance(template, dict):
            return {k: WebhookNotifier._format(v, data) for k, v in template.items()}
        if isinstance(template, list):
            return [WebhookNotifier._format(v, data) for v in template]
        return template
