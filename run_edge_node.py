"""
边缘节点主程序

每个边缘节点运行此程序：
1. 加载相机配置
2. 启动所有相机采集线程
3. 每个相机独立做检测（帧差 + YOLO + ROI）
4. 检测事件写入缓冲队列
5. 后台线程上报到中央汇聚服务
"""
import sys
import time
import threading
from pathlib import Path

import cv2
from loguru import logger

from src.loader import load_config, AppConfig
from src.detector.engine import ChuteDetector, DetectionEvent
from src.event.buffer import EventBuffer
from src.reporter.adapter import HTTPReporter, EventDispatcher


def setup_logging(config: AppConfig):
    """配置日志"""
    log_cfg = config.system.logging
    log_dir = Path(log_cfg.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_cfg.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    logger.add(
        log_cfg.file,
        level=log_cfg.level,
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        encoding="utf-8",
    )


def run_camera(camera_cfg, detection_cfg, dispatcher: EventDispatcher,
               node_id: str, show_preview: bool = False):
    """
    运行单个相机的检测循环
    """
    logger.info(f"启动相机 {camera_cfg.id} ({camera_cfg.name}), "
                f"覆盖 {len(camera_cfg.chutes)} 个格口")

    # 解析 source
    source = camera_cfg.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"无法打开相机 {camera_cfg.id}: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    cap.set(cv2.CAP_PROP_FPS, camera_cfg.fps)

    detector = ChuteDetector(camera_cfg, detection_cfg)
    frame_count = 0
    event_count = 0
    fps_timer = time.time()
    fps_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"相机 {camera_cfg.id} 读取失败")
                time.sleep(0.1)
                continue

            frame_count += 1
            fps_counter += 1

            # 处理帧
            events = detector.process_frame(frame)

            # 分发事件
            for event in events:
                dispatcher.push_event(event)
                event_count += 1
                logger.info(
                    f"事件: 格口={event.chute_name}({event.chute_id}) | "
                    f"时间={event.timestamp} | "
                    f"置信度={event.confidence:.2f}"
                )

            # 预览（调试用）
            if show_preview:
                # 绘制 ROI
                for chute in camera_cfg.chutes:
                    pts = np.array(chute.roi, np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(frame, chute.name, (chute.roi[0][0], chute.roi[0][1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    if len(chute.entry_line) == 2:
                        p1 = tuple(chute.entry_line[0])
                        p2 = tuple(chute.entry_line[1])
                        cv2.line(frame, p1, p2, (0, 0, 255), 2)

                cv2.imshow(f"Camera: {camera_cfg.id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 每秒打印 FPS
            elapsed = time.time() - fps_timer
            if elapsed >= 5.0:
                fps = fps_counter / elapsed
                pending = dispatcher.pending_count()
                logger.debug(
                    f"[{camera_cfg.id}] FPS: {fps:.1f} | "
                    f"检测帧: {frame_count} | 事件: {event_count} | "
                    f"缓冲: {pending}"
                )
                fps_timer = time.time()
                fps_counter = 0

    except KeyboardInterrupt:
        logger.info(f"相机 {camera_cfg.id} 收到中断信号")
    finally:
        detector.release()
        cap.release()
        if show_preview:
            cv2.destroyWindow(f"Camera: {camera_cfg.id}")
        logger.info(f"相机 {camera_cfg.id} 已停止")


import numpy as np  # noqa: E402 (preview 中用到)


def main():
    """边缘节点主入口"""
    config = load_config("config")
    setup_logging(config)

    node_id = config.system.node_id
    logger.info("=" * 50)
    logger.info(f"错分报告系统边缘节点启动: {node_id} ({config.system.node_name})")
    logger.info(f"加载 {len(config.cameras)} 个相机配置")
    logger.info("=" * 50)

    if not config.cameras:
        logger.error("没有配置任何相机，请先编辑 config/cameras.yaml")
        return

    # 初始化事件缓冲
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    buffer = EventBuffer(
        db_path=config.system.reporter.buffer_db,
        max_size=config.system.reporter.buffer_max_size,
    )
    logger.info(f"事件缓冲: {config.system.reporter.buffer_db} "
                f"(最大 {config.system.reporter.buffer_max_size} 条)")

    # 初始化上报器
    reporter = HTTPReporter(
        hub_url=config.system.reporter.hub_url,
        timeout=config.system.reporter.timeout,
        max_retries=config.system.reporter.max_retries,
    )

    # 初始化分发器
    dispatcher = EventDispatcher(
        buffer=buffer,
        reporter=reporter,
        config=config.system.reporter,
    )
    dispatcher.start()
    logger.info(f"上报目标: {config.system.reporter.hub_url}")

    # 启动所有相机线程
    threads = []
    for camera_cfg in config.cameras:
        t = threading.Thread(
            target=run_camera,
            args=(camera_cfg, config.system.detection, dispatcher, node_id, True),
            daemon=True,
            name=f"cam-{camera_cfg.id}",
        )
        t.start()
        threads.append(t)

    logger.info(f"所有相机已启动 ({len(threads)} 个线程)")

    # 主线程等待
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")

    # 清理
    dispatcher.stop()
    reporter.close()
    logger.info("边缘节点已停止")


if __name__ == "__main__":
    main()
