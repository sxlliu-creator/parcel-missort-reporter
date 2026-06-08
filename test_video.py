"""
视频文件检测测试脚本（基于背景减除）

用法：
    python test_video.py path/to/video.mp4
    python test_video.py path/to/video.mp4 --debug
    python test_video.py path/to/video.mp4 --no-preview
    python test_video.py path/to/video.mp4 --output result.mp4

功能：
    1. 用背景减除 + 帧差法检测运动物体
    2. 在画面上绘制 ROI 区域和 entry_line
    3. 输出检测到的"包裹进入"事件到终端
    4. 实时预览带标注的画面（按 q 退出）
    5. 可选将标注后的视频保存到文件
"""
import sys
import time
import argparse

import cv2
import numpy as np

from src.loader import load_config, ChuteConfig
from src.detector.bg_detector import BackgroundDetector
from src.detector.engine import DetectionEvent


def draw_roi(frame: np.ndarray, chutes: list) -> np.ndarray:
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

        # 绘制 entry_line（红色）+ 方向箭头
        if len(entry) == 2:
            p1 = tuple(entry[0])
            p2 = tuple(entry[1])
            cv2.line(frame, p1, p2, (0, 0, 255), 3)
            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)
            cv2.arrowedLine(frame, (mx, my - 15), (mx, my + 15),
                           (0, 0, 255), 2, tipLength=0.3)

    return frame


def draw_tracks(frame: np.ndarray, tracks: dict) -> np.ndarray:
    """绘制跟踪物体的质心和轨迹"""
    for tid, tobj in tracks.items():
        cx, cy = tobj.centroid
        prev_cx, prev_cy = tobj.prev_centroid

        # 画质心
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 255), -1)

        # 画轨迹线（从上一帧到当前帧）
        if (prev_cx, prev_cy) != (cx, cy):
            cv2.line(frame, (int(prev_cx), int(prev_cy)),
                     (int(cx), int(cy)), (0, 255, 255), 2)

        # 标注 track_id
        cv2.putText(frame, f"ID:{tid}", (int(cx) + 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 画包围盒
        x1, y1, x2, y2 = map(int, tobj.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    return frame


def draw_fg_mask(frame: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """将前景掩码缩放到画面右上角显示"""
    h, w = frame.shape[:2]
    small = cv2.resize(fg_mask, (w // 4, h // 4))
    small_color = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    frame[10:10 + h//4, w - w//4 - 10:w - 10] = small_color
    cv2.rectangle(frame, (w - w//4 - 10, 10), (w - 10, 10 + h//4),
                  (255, 255, 255), 1)
    cv2.putText(frame, "FG Mask", (w - w//4 - 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="视频检测测试（背景减除版）")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--config", default="config", help="配置目录")
    parser.add_argument("--camera-id", default="CAM-001",
                        help="使用的相机配置 ID")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    parser.add_argument("--debug", action="store_true",
                        help="打印所有跟踪信息（调试用）")
    parser.add_argument("--output", default=None,
                        help="保存标注视频到指定路径（如 result.mp4）")
    args = parser.parse_args()

    config = load_config(args.config)
    camera_cfg = None
    for cam in config.cameras:
        if cam.id == args.camera_id:
            camera_cfg = cam
            break

    if camera_cfg is None:
        print(f"[错误] 找不到相机配置: {args.camera_id}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {args.video}")
    print(f"分辨率: {w}x{h}")
    print(f"FPS: {fps:.1f}, 总帧数: {total_frames}")
    print(f"格口数: {len(camera_cfg.chutes)}")
    if args.debug:
        print("[调试模式] 将打印所有跟踪信息")
    if args.output:
        print(f"[输出] 标注视频将保存到: {args.output}")
    print("-" * 50)

    # 使用背景减除检测器
    detector = BackgroundDetector(
        camera_cfg.chutes, config.system, camera_id=camera_cfg.id)

    # 初始化视频写入器（优先用 XVID/AVI 兼容性最好，若路径指定 .mp4 则自动改 .avi）
    writer = None
    if args.output:
        import os
        out_path = args.output
        # 自动改后缀为 .avi 保证可播放
        base, ext = os.path.splitext(out_path)
        if ext.lower() != ".avi":
            out_path = base + ".avi"
            print(f"[提示] 输出格式改为 AVI（兼容性更好）: {out_path}")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[警告] 无法创建输出视频文件，将不保存")
            writer = None
        else:
            args.output = out_path  # 更新路径

    frame_idx = 0
    event_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 检测
            events = detector.process_frame(frame)

            # 绘制
            vis = draw_roi(frame.copy(), camera_cfg.chutes)
            vis = draw_tracks(vis, detector.tracks)

            # 在右上角显示前景掩码（用于调试）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            fg_mog = detector.bg_subtractor.apply(frame)
            _, fg_mog = cv2.threshold(fg_mog, 250, 255, cv2.THRESH_BINARY)
            if detector.prev_gray is not None:
                diff = cv2.absdiff(gray, detector.prev_gray)
                thresh = detector.cfg.frame_diff_threshold
                _, fg_diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                fg_diff = cv2.morphologyEx(fg_diff, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.bitwise_or(fg_mog, fg_diff)
            else:
                fg_mask = fg_mog
            vis = draw_fg_mask(vis, fg_mask)

            # 打印事件
            for ev in events:
                event_count += 1
                print(f"[事件] 帧#{frame_idx} | "
                      f"格口={ev.chute_name}({ev.chute_id}) | "
                      f"时间={ev.timestamp} | "
                      f"置信度={ev.confidence:.2f} | "
                      f"TrackID={ev.track_id}")

            # 调试：打印跟踪信息
            if args.debug and frame_idx % 10 == 0:
                for tid, tobj in detector.tracks.items():
                    cx, cy = tobj.centroid
                    in_rois = []
                    for ch in camera_cfg.chutes:
                        pts = np.array(ch.roi, dtype=np.int32)
                        if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                            in_rois.append(ch.name)
                    status = f"在{','.join(in_rois)}内" if in_rois else "在ROI外"
                    print(f"  [跟踪] ID:{tid} 质心=({cx},{cy}) 面积={tobj.area} {status}")

            # 状态栏
            interval = time.time() - t0
            fps_cur = frame_idx / interval if interval > 0 else 0
            info = (f"Frame: {frame_idx}/{total_frames} | "
                    f"FPS: {fps_cur:.1f} | Events: {event_count} | "
                    f"Tracks: {len(detector.tracks)}")
            cv2.putText(vis, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 保存帧到输出视频
            if writer is not None:
                writer.write(vis)

            # 预览
            if not args.no_preview:
                cv2.imshow("Video Test - 按 q 退出", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

            if args.no_preview and frame_idx % 100 == 0:
                print(f"进度: {frame_idx}/{total_frames} 帧, 事件: {event_count}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[输出] 标注视频已保存: {args.output}")
        cv2.destroyAllWindows()
        detector.release()
        elapsed = time.time() - t0
        print("-" * 50)
        print(f"处理完成: {frame_idx} 帧, {event_count} 个事件")
        print(f"耗时: {elapsed:.1f}s, 平均 FPS: {frame_idx/elapsed:.1f}")


if __name__ == "__main__":
    main()
