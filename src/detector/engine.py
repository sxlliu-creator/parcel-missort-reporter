"""
检测引擎：帧差预检 + YOLO 推理 + ROI 区域进入判定

流程：
1. 帧差法快速判断画面是否变化
2. 有变化 → YOLO 目标检测 + 跟踪
3. 对每个目标判断是否进入了某个格口的 ROI 区域
4. 触发事件（含毫秒级时间戳）
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import time

from ..loader import CameraConfig, ChuteConfig, DetectionConfig


CST = timezone(timedelta(hours=8))


@dataclass
class DetectionEvent:
    """包裹进入格口事件"""
    chute_id: str
    chute_name: str
    camera_id: str
    timestamp: str  # ISO 8601 ms
    parcel_id: Optional[str] = None  # 后续条码识别时使用
    confidence: float = 0.0
    track_id: int = -1
    bbox: Optional[Tuple] = None  # (x1, y1, x2, y2)
    # 预留：包裹形态特征
    parcel_shape: Optional[Dict] = field(default_factory=dict)


class FrameDiffChecker:
    """帧差法预检器：快速判断画面是否发生变化"""

    def __init__(self, threshold: int = 25, change_ratio: float = 0.02):
        self.threshold = threshold
        self.change_ratio = change_ratio
        self._prev_frame: Optional[np.ndarray] = None

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        判断当前帧相对于上一帧是否有明显变化
        返回 True 表示有变化，需要进一步检测
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_frame is None:
            self._prev_frame = gray
            return True  # 首帧默认触发

        diff = cv2.absdiff(self._prev_frame, gray)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        ratio = changed_pixels / total_pixels

        self._prev_frame = gray
        return ratio >= self.change_ratio

    def reset(self):
        self._prev_frame = None


class RegionEntryDetector:
    """
    ROI 区域进入检测器
    判断目标（包裹）质心是否从外部穿过格口的 entry_line
    """

    @staticmethod
    def _point_side(px: float, py: float,
                    x1: float, y1: float, x2: float, y2: float) -> float:
        """叉积法判断点在线的哪一侧，正数=右侧/下方，负数=左侧/上方"""
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    @staticmethod
    def _inside_roi(px: float, py: float, roi: List[List[int]]) -> bool:
        """射线法判断点是否在多边形 ROI 内"""
        n = len(roi)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = roi[i]
            xj, yj = roi[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def __init__(self, chutes: List[ChuteConfig], cooldown_seconds: float = 3.0):
        """
        chutes: 格口配置列表（含 ROI 和 entry_line）
        cooldown_seconds: 同一格口同一 track_id 的冷却时间
        """
        self.chutes = chutes
        self.cooldown = cooldown_seconds
        # track_id → (chute_id, last_trigger_time) 防重复
        self._last_trigger: Dict[int, Tuple[str, float]] = {}
        # track_id → [(chute_id, side)] 记录历史位置
        self._track_history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        # 清理旧 track 记录
        self._last_cleanup = time.time()

    def check_entry(self, centroid: Tuple[float, float],
                    track_id: int, frame_time: float) -> Optional[DetectionEvent]:
        """
        检查目标是否进入了某个格口的判定区域
        centroid: (cx, cy) 目标质心坐标
        track_id: 跟踪 ID
        frame_time: 当前帧时间戳（秒）
        """
        cx, cy = centroid

        for chute in self.chutes:
            chute_id = chute.id
            line = chute.entry_line
            x1, y1 = line[0]
            x2, y2 = line[1]

            # 计算质心在判定线哪一侧
            side = self._point_side(cx, cy, x1, y1, x2, y2)

            # 记录历史位置
            history = self._track_history[track_id]
            if history:
                last_chute, last_side = history[-1]
                # 只有同一格口才判断跨线
                if last_chute == chute_id:
                    # 方向判定
                    if chute.direction == "enter":
                        # enter：从上侧穿到下侧（side 从负到正）
                        crossed = last_side <= 0 and side > 0
                    else:
                        # exit：从下侧穿到上侧
                        crossed = last_side >= 0 and side < 0

                    if crossed:
                        # 检查是否在 ROI 区域内
                        if self._inside_roi(cx, cy, chute.roi):
                            # 冷却检查
                            last_trigger = self._last_trigger.get(track_id)
                            if last_trigger:
                                last_ch, last_time = last_trigger
                                if last_ch == chute_id and \
                                   (frame_time - last_time) < self.cooldown:
                                    return None

                            self._last_trigger[track_id] = (chute_id, frame_time)

                            # 生成毫秒级时间戳
                            now = datetime.now(CST)
                            ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + \
                                 f"{now.microsecond // 1000:03d}+08:00"

                            return DetectionEvent(
                                chute_id=chute.id,
                                chute_name=chute.name,
                                camera_id="",  # 由外层填充
                                timestamp=ts,
                                track_id=track_id,
                                confidence=0.0,  # 由外层填充
                            )

            history.append((chute_id, side))
            # 只保留最近 30 条历史
            if len(history) > 30:
                history.pop(0)

        return None

    def cleanup(self, current_time: float):
        """定期清理过期的 track 记录"""
        if current_time - self._last_cleanup < 30:
            return
        self._last_cleanup = current_time
        stale_ids = [
            tid for tid, triggers in self._last_trigger.items()
            if current_time - triggers[1] > 60
        ]
        for tid in stale_ids:
            self._last_trigger.pop(tid, None)
            self._track_history.pop(tid, None)


class ChuteDetector:
    """
    格口检测器（单相机多格口）
    组合帧差预检 + YOLO 推理 + ROI 进入判定
    """

    def __init__(self, camera: CameraConfig, detection_cfg: DetectionConfig):
        self.camera = camera
        self.detection_cfg = detection_cfg
        self.frame_diff = FrameDiffChecker(
            threshold=detection_cfg.frame_diff_threshold,
            change_ratio=detection_cfg.frame_diff_ratio,
        )
        self.entry_detector = RegionEntryDetector(
            chutes=camera.chutes,
            cooldown_seconds=detection_cfg.cooldown_seconds,
        )
        self._frame_count = 0

        # YOLO 模型（延迟加载）
        self._model = None

    def _load_model(self):
        """延迟加载 YOLO 模型"""
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO(self.detection_cfg.yolo_model)

    def process_frame(self, frame: np.ndarray) -> List[DetectionEvent]:
        """
        处理一帧画面
        返回触发的检测事件列表
        """
        self._frame_count += 1
        events = []
        frame_time = time.time()

        # Step1: 帧差预检
        if not self.frame_diff.has_motion(frame):
            return events

        # Step2: 跳帧控制
        if self._frame_count % self.detection_cfg.inference_interval != 0:
            return events

        # Step3: YOLO 推理 + 跟踪
        self._load_model()
        results = self._model.track(
            frame,
            persist=True,
            conf=self.detection_cfg.yolo_conf,
            imgsz=self.detection_cfg.imgsz,
            device=self.detection_cfg.device,
            tracker=self.detection_cfg.tracker,
            verbose=False,
            classes=None,  # 检测所有类；后期可限为包裹类
        )

        if results[0].boxes is None:
            return events

        boxes = results[0].boxes
        if boxes.id is None:
            return events

        # Step4: 遍历每个检测到的目标
        for box, track_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Step5: ROI 进入判定
            event = self.entry_detector.check_entry(
                (cx, cy), track_id, frame_time
            )
            if event is not None:
                event.camera_id = self.camera.id
                event.confidence = float(conf)
                events.append(event)

        # 定期清理
        self.entry_detector.cleanup(frame_time)

        return events

    def release(self):
        """释放资源"""
        self.frame_diff.reset()
        self._model = None
