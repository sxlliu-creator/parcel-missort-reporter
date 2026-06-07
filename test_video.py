"""
视频文件检测测试脚本

用法：
    python test_video.py path/to/video.mp4

功能：
    1. 用帧差 + YOLO 检测视频中的物体
    2. 在画面上绘制 ROI 区域和 entry_line
    3. 输出检测到的"包裹进入"事件到终端
    4. 实时预览带标注的画面（按 q 退出）

适合在没有摄像头的情况下验证检测算法效果。
"""
import sys
import time
import argparse

import cv2
import numpy as np

# 复用项目内模块
from src.loader import load_config, ChuteConfig
from src.detector.engine import ChuteDetector, DetectionEvent


def draw_roi(frame: "np.ndarray", chutes: list) -> "np.ndarray":
    """在画面上绘制所有格口的 ROI 和 entry_line"""
    for chute in chutes:
        roi_pts = np.array(chute.roi, dtype=np.int32)
        entry = chute.entry_line

        # 绘制 ROI 区域（绿色半透明填充）
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_pts], (0, 200, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [roi_pts], True, (0, 255, 0), 2)

        # 标注格口名称
        cx = int(roi_pts[:, 0].mean())
        cy = int(roi_pts[:, 1].mean())
        cv2.putText(frame, chute.name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制 entry_line（红色）
        if len(entry) == 2:
            p1 = tuple(entry[0])
            p2 = tuple(entry[1])
            cv2.line(frame, p1, p2, (0, 0, 255), 3)
            # 画箭头表示方向
            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)
            cv2.arrowedLine(frame, (mx, my - 15), (mx, my + 15),
                           (0, 0, 255), 2, tipLength=0.3)

    return frame


def draw_detections(frame: "np.ndarray", detections: list) -> "np.ndarray":
    """绘制检测框和 track_id"""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        tid = det["track_id"]
        conf = det["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"ID:{tid} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="视频检测测试")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--config", default="config", help="配置目录")
    parser.add_argument("--camera-id", default="CAM-001",
                        help="使用的相机配置 ID（在 cameras.yaml 中定义）")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    camera_cfg = None
    for cam in config.cameras:
        if cam.id == args.camera_id:
            camera_cfg = cam
            break

    if camera_cfg is None:
        print(f"[错误] 找不到相机配置: {args.camera_id}")
        print(f"可用相机: {[c.id for c in config.cameras]}")
        sys.exit(1)

    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {args.video}")
    print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {fps}, 总帧数: {total_frames}")
    print(f"格口数: {len(camera_cfg.chutes)}")
    print("-" * 50)

    # 初始化检测器
    detector = ChuteDetector(camera_cfg, config.system.detection)

    frame_idx = 0
    event_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 检测（复用 ChuteDetector.process_frame）
            events = detector.process_frame(frame)

            # 绘制 ROI 和检测结果
            vis = draw_roi(frame.copy(), camera_cfg.chutes)

            # 从检测器内部获取最新检测结果用于绘制
            # （process_frame 只返回事件，不直接返回检测框）
            # 这里从 YOLO 结果里取框
            if detector._model is not None:
                results = detector._model.track(
                    frame, persist=True,
                    conf=config.system.detection.yolo_conf,
                    imgsz=config.system.detection.imgsz,
                    device=config.system.detection.device,
                    verbose=False, classes=[0],
                )
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box, tid, conf in zip(
                        results[0].boxes.xyxy.cpu().numpy(),
                        results[0].boxes.id.cpu().numpy().astype(int),
                        results[0].boxes.conf.cpu().numpy(),
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(vis, f"ID:{tid}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 打印事件
            for ev in events:
                event_count += 1
                print(f"[事件] 帧#{frame_idx} | "
                      f"格口={ev.chute_name}({ev.chute_id}) | "
                      f"时间={ev.timestamp} | "
                      f"置信度={ev.confidence:.2f}")

            # 状态栏
            interval = time.time() - t0
            fps_cur = frame_idx / interval if interval > 0 else 0
            info = (f"Frame: {frame_idx}/{total_frames} | "
                    f"FPS: {fps_cur:.1f} | Events: {event_count}")
            cv2.putText(vis, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 预览
            if not args.no_preview:
                cv2.imshow("Video Test - 按 q 退出", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # 空格暂停
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        elapsed = time.time() - t0
        print("-" * 50)
        print(f"处理完成: {frame_idx} 帧, {event_count} 个事件")
        print(f"耗时: {elapsed:.1f}s, 平均 FPS: {frame_idx/elapsed:.1f}")


if __name__ == "__main__":
    main()
