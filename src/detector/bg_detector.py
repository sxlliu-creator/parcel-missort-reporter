"""
帧差 + 背景减除检测引擎

核心思路：
  1. 背景建模（MOG2）+ 帧差法，检测画面中的运动区域
  2. 在 ROI 区域内做连通域分析，提取运动物体质心
  3. 跟踪质心轨迹，判断是否穿过 entry_line
  4. 从 ROI 外 → ROI 内 视为"包裹进入"，触发事件

适合无 GPU 的 CPU 环境，毫秒级推理。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .engine import DetectionEvent


@dataclass
class _TrackedObject:
    tid: int
    centroid: tuple
    prev_centroid: tuple
    bbox: tuple
    area: int
    first_seen: float
    last_seen: float
    crossed_events: set = field(default_factory=set)


class BackgroundDetector:
    """
    基于背景减除的包裹检测器

    用法：
        detector = BackgroundDetector(chute_configs, system_config, camera_id)
        for frame in frames:
            events = detector.process_frame(frame)
    """
    def __init__(self, chute_configs, system_config, camera_id="CAM-001"):
        self.chutes = chute_configs
        self.cfg = system_config.detection
        self.camera_id = camera_id

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=float(self.cfg.frame_diff_threshold),
            detectShadows=False,
        )

        self.prev_gray = None
        self.next_tid = 1
        self.tracks = {}
        self.lost_tracks = {}
        self.cooldown_sec = self.cfg.cooldown_seconds
        self.last_trigger_time = {}

        self._roi_polygons = {}
        for ch in self.chutes:
            pts = np.array(ch.roi, dtype=np.int32)
            self._roi_polygons[ch.id] = pts

    # ------------------------------------------------------------------ #

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        now = time.time()

        fg_mask = self._compute_fg_mask(frame, w, h)
        contours = self._extract_contours(fg_mask, w, h)
        detections = self._contours_to_detections(contours, w, h, now)
        self._update_tracks(detections, now)
        events = self._check_line_crossing(now)
        return events

    # ------------------------------------------------------------------ #

    def _compute_fg_mask(self, frame, w, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg_mog = self.bg_subtractor.apply(frame)
        _, fg_mog = cv2.threshold(fg_mog, 250, 255, cv2.THRESH_BINARY)

        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, fg_diff = cv2.threshold(
                diff, int(self.cfg.frame_diff_threshold), 255, cv2.THRESH_BINARY)
            fg_diff = cv2.morphologyEx(
                fg_diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_diff = cv2.dilate(fg_diff, np.ones((5, 5), np.uint8), iterations=1)
        else:
            fg_diff = np.zeros((h, w), dtype=np.uint8)

        self.prev_gray = gray
        fg_mask = cv2.bitwise_or(fg_mog, fg_diff)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        return fg_mask

    def _extract_contours(self, fg_mask, w, h):
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(200, w * h * 0.0001)
        max_area = w * h * 0.8
        return [c for c in contours
                if min_area < cv2.contourArea(c) < max_area]

    def _contours_to_detections(self, contours, w, h, now):
        dets = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx = x + cw // 2
            cy = y + ch // 2
            area = cv2.contourArea(cnt)
            dets.append({
                "centroid": (cx, cy),
                "bbox": (x, y, x + cw, y + ch),
                "area": area,
                "confidence": min(1.0, area / (w * h * 0.1)),
            })
        return dets

    def _update_tracks(self, detections, now):
        matched = set()
        new_tracks = {}

        for det in detections:
            cx, cy = det["centroid"]
            best_tid = None
            best_dist = 80

            for tid, tobj in self.tracks.items():
                dist = np.hypot(cx - tobj.centroid[0], cy - tobj.centroid[1])
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is None:
                for tid, tobj in self.lost_tracks.items():
                    dist = np.hypot(cx - tobj.centroid[0], cy - tobj.centroid[1])
                    if dist < best_dist * 1.5:
                        best_dist = dist
                        best_tid = tid

            if best_tid is not None:
                tobj = self.tracks.get(best_tid) or self.lost_tracks.pop(best_tid)
                tobj.prev_centroid = tobj.centroid
                tobj.centroid = (cx, cy)
                tobj.bbox = det["bbox"]
                tobj.area = det["area"]
                tobj.last_seen = now
                new_tracks[best_tid] = tobj
                matched.add(best_tid)
            else:
                new_tracks[self.next_tid] = _TrackedObject(
                    tid=self.next_tid,
                    centroid=(cx, cy),
                    prev_centroid=(cx, cy),
                    bbox=det["bbox"],
                    area=det["area"],
                    first_seen=now,
                    last_seen=now,
                )
                self.next_tid += 1

        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.lost_tracks[tid] = self.tracks.pop(tid)

        to_delete = [tid for tid, tobj in self.lost_tracks.items()
                     if now - tobj.last_seen > 0.5]
        for tid in to_delete:
            del self.lost_tracks[tid]

        self.tracks = new_tracks

    def _check_line_crossing(self, now):
        events = []
        for tid, tobj in self.tracks.items():
            if tobj.prev_centroid == tobj.centroid:
                continue

            for ch in self.chutes:
                chute_id = ch.id
                pts = self._roi_polygons[ch.id]
                entry = ch.entry_line

                in_now = cv2.pointPolygonTest(pts, tobj.centroid, False) >= 0
                in_prev = cv2.pointPolygonTest(pts, tobj.prev_centroid, False) >= 0

                crossed_in = False
                if entry and len(entry) == 2:
                    crossed_in = self._check_line_cross(
                        tobj.prev_centroid, tobj.centroid,
                        tuple(entry[0]), tuple(entry[1]))

                if (not in_prev and in_now) or crossed_in:
                    cool_key = f"{tid}_{chute_id}"
                    last_cool = self.last_trigger_time.get(chute_id, 0)
                    if cool_key in tobj.crossed_events:
                        continue
                    if now - last_cool < self.cooldown_sec:
                        continue

                    tobj.crossed_events.add(cool_key)
                    self.last_trigger_time[chute_id] = now

                    ts = time.strftime("%Y-%m-%dT%H:%M:%S.", time.localtime(now))
                    ts += f"{int(now*1000)%1000:03d}+08:00"

                    ev = DetectionEvent(
                        chute_id=chute_id,
                        chute_name=ch.name,
                        camera_id=self.camera_id,
                        timestamp=ts,
                        confidence=min(1.0, tobj.area / 50000),
                        track_id=tid,
                        bbox=tobj.bbox,
                    )
                    events.append(ev)
        return events

    @staticmethod
    def _check_line_cross(p_prev, p_now, l1, l2):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        d1 = cross(l1, l2, p_prev)
        d2 = cross(l1, l2, p_now)
        d3 = cross(p_prev, p_now, l1)
        d4 = cross(p_prev, p_now, l2)

        return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
                ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

    def release(self):
        self.tracks.clear()
        self.lost_tracks.clear()
        self.prev_gray = None
