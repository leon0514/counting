# -*- coding: utf-8 -*-
"""
PersonCounter - 基于检测线的人流量统计模块

使用 ByteTrack 跟踪行人，通过判断行人中心轨迹与检测线的相交关系来统计穿越人数。

方向约定：
    检测线由 line_start -> line_end 定义，其法向量 normal = (-dy, dx)。
    默认 "进(in)" 方向为 normal 指向的一侧，"出(out)" 为相反方向。
    例如水平线 (0,0)->(100,0)，normal 指向下，向下穿越计为 in，向上为 out。
"""

import numpy as np
from collections import deque


class LineCounter:
    """单条检测线的人流量计数器"""

    def __init__(self, line_start, line_end, max_history=30, name="line1",
                 min_cross_depth=20.0, confirm_frames=2):
        """
        :param line_start: 检测线起点 (x, y)
        :param line_end:   检测线终点 (x, y)
        :param max_history: 单个轨迹保存的最大历史中心点数
        :param name:       检测线名称（用于日志/显示）
        :param min_cross_depth: 穿越深度阈值（像素），过线后需继续走出该距离才确认
        :param confirm_frames: 过线后需在目标侧停留的最少帧数
        """
        self.line = np.array([line_start, line_end], dtype=np.float32)
        self.name = name
        self.max_history = max_history
        self.min_cross_depth = min_cross_depth
        self.confirm_frames = confirm_frames

        # {track_id: deque([(cx, cy), ...])}
        self.track_history = {}
        # {track_id: "in" or "out"} — 已确认的计数
        self.counted = {}
        # {track_id: {"side": "in"/"out", "frames": n, "max_depth": d}} — 待确认的过线状态
        self.pending = {}

        self.in_count = 0
        self.out_count = 0

        # 计算法向量 normal = (-dy, dx)，用于判断穿越方向
        dx = self.line[1][0] - self.line[0][0]
        dy = self.line[1][1] - self.line[0][1]
        self.normal = np.array([-dy, dx], dtype=np.float32)
        norm_len = np.linalg.norm(self.normal)
        if norm_len > 0:
            self.normal /= norm_len
        else:
            self.normal = np.array([0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # 几何工具
    # ------------------------------------------------------------------
    @staticmethod
    def _cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _orient(self, a, b, c):
        return self._cross2d(b - a, c - a)

    def _segments_intersect(self, p1, p2, p3, p4):
        """判断两线段是否严格相交（不含端点共线重叠的情况）"""
        o1 = self._orient(p1, p2, p3)
        o2 = self._orient(p1, p2, p4)
        o3 = self._orient(p3, p4, p1)
        o4 = self._orient(p3, p4, p2)

        # 严格相交：方向符号相反（浮点用容差判断近似共线）
        eps = 1e-6
        if abs(o1) < eps or abs(o2) < eps or abs(o3) < eps or abs(o4) < eps:
            return False
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    def _cross_direction(self, p_prev, p_curr):
        """
        判断从 p_prev 到 p_curr 的移动相对于检测线的方向。
        返回值:  1  -> 与法向量同向 (计为 in)
                -1  -> 与法向量反向 (计为 out)
                 0  -> 无法判断
        """
        move_vec = p_curr - p_prev
        dot = np.dot(move_vec, self.normal)
        if abs(dot) < 1e-6:
            return 0
        return 1 if dot > 0 else -1

    def _signed_distance(self, point):
        """计算点到检测线的有符号距离（沿法向量方向为正）"""
        # 线段的起点到点的向量，投影到法向量上
        vec = point - self.line[0]
        return np.dot(vec, self.normal)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    def update(self, tracks):
        """
        根据当前帧的跟踪结果更新计数。
        采用"穿越深度确认"机制防止在线边徘徊误报。

        :param tracks: list[STrack]
        :return: list[dict] 本帧确认的事件列表，每个元素 {"track_id": int, "direction": "in"/"out"}
        """
        current_ids = set()
        events = []  # 本帧确认的事件

        for t in tracks:
            if not getattr(t, "is_activated", True):
                continue
            tid = int(t.track_id)
            current_ids.add(tid)

            x, y, w, h = t.tlwh
            center = np.array([x + w / 2.0, y + h], dtype=np.float32)

            hist = self.track_history.setdefault(tid, deque(maxlen=self.max_history))

            # --- 阶段1: 检测新过线事件 ---
            if len(hist) >= 1 and tid not in self.pending:
                prev = np.array(hist[-1], dtype=np.float32)
                if self._segments_intersect(prev, center, self.line[0], self.line[1]):
                    direction = self._cross_direction(prev, center)
                    if direction != 0:
                        side = "in" if direction == 1 else "out"
                        # 允许首次穿越或反向穿越（与当前所在侧相反的方向）
                        current_side = self.counted.get(tid)
                        if current_side is None or current_side != side:
                            cross_dist = self._signed_distance(center)
                            self.pending[tid] = {
                                "side": side,
                                "frames": 1,
                                "max_depth": abs(cross_dist),
                                "cross_dist": cross_dist
                            }

            # --- 阶段2: 更新 pending 状态 ---
            if tid in self.pending:
                p = self.pending[tid]
                curr_dist = self._signed_distance(center)
                if p["side"] == "in":
                    depth = curr_dist
                else:
                    depth = -curr_dist

                if depth > 0:
                    p["frames"] += 1
                    p["max_depth"] = max(p["max_depth"], depth)
                    if p["frames"] >= self.confirm_frames and p["max_depth"] >= self.min_cross_depth:
                        if p["side"] == "in":
                            self.in_count += 1
                        else:
                            self.out_count += 1
                        self.counted[tid] = p["side"]  # 更新当前所在侧
                        events.append({"track_id": tid, "direction": p["side"]})
                        del self.pending[tid]
                else:
                    del self.pending[tid]

            hist.append(center.tolist())

        # 清理已丢失目标的 pending 状态
        lost_pending = set(self.pending.keys()) - current_ids
        for tid in lost_pending:
            del self.pending[tid]

        return events

    def reset(self, soft: bool = False):
        """重置计数。

        :param soft: True=软重置（保留在场人数基准，不清空 counted，防止重复计数）
                    False=硬重置（完全清零，慎用）
        """
        if soft:
            # 软重置：保留当前在场人数作为新基准
            current = max(0, self.in_count - self.out_count)
            self.in_count = current
            self.out_count = 0
            # 不清空 counted/track_history，避免已入场人员重复计数
            self.pending.clear()
        else:
            # 硬重置：完全清零
            self.in_count = 0
            self.out_count = 0
            self.track_history.clear()
            self.counted.clear()
            self.pending.clear()

    def get_counts(self):
        """返回 (in_count, out_count, total_count)"""
        return self.in_count, self.out_count, self.in_count + self.out_count


class DoubleLineCounter:
    """双检测线人流量计数器（门式检测）

    两条线定义一个"门"区域：
        - line_out: 外侧线（靠近外部）
        - line_in: 内侧线（靠近内部）

    人必须从外侧穿过 line_out，再穿过 line_in 才算"进入"；
    反向（先 line_in 再 line_out）才算"出去"。
    两条线都穿过才触发事件，大幅减少单条线的误判。

    状态机:
        0: 初始/空闲
        1: 已穿过 line_out（从外向内）→ 等待穿过 line_in 确认进入
        2: 已穿过 line_in（从内向外）→ 等待穿过 line_out 确认出去
    """

    def __init__(self, line_out, line_in, name="gate1", max_history=30):
        """
        :param line_out: 外侧线坐标 ((x1,y1), (x2,y2))
        :param line_in:  内侧线坐标 ((x1,y1), (x2,y2))
        :param name: 检测门名称
        :param max_history: 轨迹历史最大长度
        """
        self.line_out = np.array(line_out, dtype=np.float32)
        self.line_in = np.array(line_in, dtype=np.float32)
        self.name = name
        self.max_history = max_history

        self.track_history = {}   # {tid: deque([(cx,cy),...])}
        self.track_state = {}     # {tid: 0/1/2}
        self.counted = {}         # {tid: "in"/"out"}

        self.in_count = 0
        self.out_count = 0

        # 统一法向量（以 line_in 为准）
        dx = self.line_in[1][0] - self.line_in[0][0]
        dy = self.line_in[1][1] - self.line_in[0][1]
        self.normal = np.array([-dy, dx], dtype=np.float32)
        nlen = np.linalg.norm(self.normal)
        if nlen > 0:
            self.normal /= nlen

        # 确保 line_out 法向量方向一致（若相反则自动翻转）
        out_dx = self.line_out[1][0] - self.line_out[0][0]
        out_dy = self.line_out[1][1] - self.line_out[0][1]
        out_normal = np.array([-out_dy, out_dx], dtype=np.float32)
        out_nlen = np.linalg.norm(out_normal)
        if out_nlen > 0:
            out_normal /= out_nlen
        if np.dot(self.normal, out_normal) < 0:
            # 方向相反，翻转 line_out 使其与 line_in 同向
            self.line_out = self.line_out[::-1].copy()

    # ------------------------------------------------------------------
    # 几何工具（与 LineCounter 复用）
    # ------------------------------------------------------------------
    @staticmethod
    def _cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _orient(self, a, b, c):
        return self._cross2d(b - a, c - a)

    def _segments_intersect(self, p1, p2, p3, p4):
        """判断两线段是否严格相交"""
        o1 = self._orient(p1, p2, p3)
        o2 = self._orient(p1, p2, p4)
        o3 = self._orient(p3, p4, p1)
        o4 = self._orient(p3, p4, p2)
        eps = 1e-6
        if abs(o1) < eps or abs(o2) < eps or abs(o3) < eps or abs(o4) < eps:
            return False
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    def _cross_direction(self, p_prev, p_curr):
        """移动方向与统一法向量的关系: 1=同向(in), -1=反向(out), 0=无法判断"""
        move_vec = p_curr - p_prev
        dot = np.dot(move_vec, self.normal)
        if abs(dot) < 1e-6:
            return 0
        return 1 if dot > 0 else -1

    def _line_cross(self, p_prev, p_curr, line):
        """判断线段 p_prev->p_curr 是否与 line 相交，返回方向"""
        if self._segments_intersect(p_prev, p_curr, line[0], line[1]):
            return self._cross_direction(p_prev, p_curr)
        return 0

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    def update(self, tracks):
        """
        根据跟踪结果更新计数。
        :param tracks: list[STrack]
        :return: list[dict] 本帧确认的事件
        """
        current_ids = set()
        events = []

        for t in tracks:
            if not getattr(t, "is_activated", True):
                continue
            tid = int(t.track_id)
            current_ids.add(tid)

            x, y, w, h = t.tlwh
            center = np.array([x + w / 2.0, y + h], dtype=np.float32)

            hist = self.track_history.setdefault(tid, deque(maxlen=self.max_history))

            if len(hist) >= 1:
                prev = np.array(hist[-1], dtype=np.float32)
                state = self.track_state.get(tid, 0)

                cross_out = self._line_cross(prev, center, self.line_out)
                cross_in = self._line_cross(prev, center, self.line_in)

                # 穿过 line_out（向内）→ 进入等待态
                if cross_out > 0 and state == 0:
                    self.track_state[tid] = 1

                # 穿过 line_in（向内）→ 确认进入
                if cross_in > 0 and state == 1:
                    self.in_count += 1
                    self.counted[tid] = "in"
                    events.append({"track_id": tid, "direction": "in"})
                    self.track_state[tid] = 0

                # 穿过 line_in（向外）→ 出去等待态
                if cross_in < 0 and state == 0:
                    self.track_state[tid] = 2

                # 穿过 line_out（向外）→ 确认出去
                if cross_out < 0 and state == 2:
                    self.out_count += 1
                    self.counted[tid] = "out"
                    events.append({"track_id": tid, "direction": "out"})
                    self.track_state[tid] = 0

                # 穿回取消：已进入等待态但反向穿回 line_out
                if cross_out < 0 and state == 1:
                    self.track_state[tid] = 0

                # 穿回取消：已出去等待态但反向穿回 line_in
                if cross_in > 0 and state == 2:
                    self.track_state[tid] = 0

            hist.append(center.tolist())

        # 清理丢失目标的残留状态
        lost = set(self.track_state.keys()) - current_ids
        for tid in lost:
            del self.track_state[tid]

        return events

    def reset(self, soft: bool = False):
        if soft:
            current = max(0, self.in_count - self.out_count)
            self.in_count = current
            self.out_count = 0
        else:
            self.in_count = 0
            self.out_count = 0
            self.track_history.clear()
            self.counted.clear()
            self.track_state.clear()

    def get_counts(self):
        return self.in_count, self.out_count, self.in_count + self.out_count

    def draw_on_image(self, image, tracks=None, font_scale=0.6, thickness=2):
        """在图像上绘制双检测线、方向指示和行人检测框（修改原图）。"""
        import cv2

        # 颜色
        COLOR_OUT = (0, 255, 0)       # 绿色 = 外线 (OUT)
        COLOR_IN  = (0, 0, 255)       # 红色 = 内线 (IN)
        COLOR_ARROW = (255, 0, 0)     # 蓝色 = in 方向箭头
        COLOR_BOX = (255, 255, 0)     # 青色 = 行人框

        # 画行人检测框和跟踪 ID
        if tracks:
            for t in tracks:
                if not getattr(t, "is_activated", True):
                    continue
                x, y, w, h = t.tlwh
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_BOX, 2, cv2.LINE_AA)
                tid = int(t.track_id)
                label = f"ID:{tid}"
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 1, cv2.LINE_AA)

        # 画外线
        p1 = tuple(self.line_out[0].astype(int))
        p2 = tuple(self.line_out[1].astype(int))
        cv2.line(image, p1, p2, COLOR_OUT, thickness, cv2.LINE_AA)
        cv2.putText(image, "OUT", (p1[0] + 5, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_OUT, thickness, cv2.LINE_AA)

        # 画内线
        q1 = tuple(self.line_in[0].astype(int))
        q2 = tuple(self.line_in[1].astype(int))
        cv2.line(image, q1, q2, COLOR_IN, thickness, cv2.LINE_AA)
        cv2.putText(image, "IN", (q1[0] + 5, q1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_IN, thickness, cv2.LINE_AA)

        # 画法向量箭头（从两条线的中心点出发，指向 in 方向）
        center_line_start = (self.line_in[0] + self.line_out[0]) / 2.0
        center_line_end   = (self.line_in[1] + self.line_out[1]) / 2.0
        mid = ((center_line_start + center_line_end) / 2.0).astype(int)
        arrow_end = (mid + self.normal * 40).astype(int)
        cv2.arrowedLine(image, tuple(mid), tuple(arrow_end),
                        COLOR_ARROW, thickness, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(image, "in-dir", (arrow_end[0] + 5, arrow_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, COLOR_ARROW, 1, cv2.LINE_AA)

        return image


class PersonCounter:
    """多检测门人流量统计器（包装 DoubleLineCounter，支持同时监控多个门）。"""

    def __init__(self):
        self.counters = []

    def add_gate(self, line_out, line_in, name=None):
        name = name or f"gate{len(self.counters) + 1}"
        counter = DoubleLineCounter(line_out, line_in, name=name)
        self.counters.append(counter)
        return counter

    def update(self, tracks):
        results = {}
        for c in self.counters:
            results[c.name] = c.update(tracks)
        return results

    def get_counts(self):
        return {c.name: c.get_counts() for c in self.counters}

    def reset(self, soft: bool = True):
        for c in self.counters:
            c.reset(soft=soft)
