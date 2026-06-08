#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式 ROI 标定工具
用法: python roi_calibrate.py <video_path> [config/cameras.yaml]

在视频帧上点击标定每个格口的 ROI 多边形区域和触发线

按键说明:
  鼠标左键    - 添加 ROI 顶点
  鼠标右键    - 删除最后一个顶点
  c 键        - 完成当前 ROI，输入格口 ID
  l 键        - 绘制触发线（2个点）
  n 键        - 保存当前格口，开始下一个
  s 键        - 保存所有 ROI 并退出
  q 键        - 不保存退出
"""

import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

# -------- 全局状态 --------
points = []            # 当前 ROI 多边形顶点
trigger_points = []    # 当前触发线两点
rois = []             # 已完成的 ROI: {id, name, polygon, trigger_line}
drawing_trigger = False
cap = None
frame = None
clone = None
current_chute_id = "A01"
current_chute_name = "格口A01"

WINDOW_NAME = "ROI Calibration - Click=add point, c=complete, l=trigger, n=next, s=save"


def draw_overlay(img):
    """在图像上绘制已完成的 ROI 和当前正在绘制的点"""
    overlay = img.copy()

    # 已完成的 ROI（半透明绿色填充 + 边界）
    for roi in rois:
        poly = np.array(roi["polygon"], np.int32)
        cv2.fillPoly(overlay, [poly], (0, 200, 0))
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        # 标签
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        label = f"{roi['id']}"
        # 黑色描边
        cv2.putText(overlay, label, (cx - 20, cy + 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(overlay, label, (cx - 20, cy + 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 触发线（红色箭头）
        if roi.get("trigger_line") and len(roi["trigger_line"]) == 2:
            p1 = tuple(map(int, roi["trigger_line"][0]))
            p2 = tuple(map(int, roi["trigger_line"][1]))
            cv2.arrowedLine(overlay, p1, p2, (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(overlay, "entry", (p1[0] + 5, p1[1] - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 当前正在绘制的 ROI 顶点
    for pt in points:
        cv2.circle(overlay, tuple(map(int, pt)), 4, (0, 255, 255), -1)
    if len(points) > 1:
        pts_arr = np.array(points, np.int32)
        cv2.polylines(overlay, [pts_arr], False, (0, 255, 255), 2, cv2.LINE_AA)

    # 当前触发线
    for pt in trigger_points:
        cv2.circle(overlay, tuple(map(int, pt)), 5, (0, 0, 255), -1)
    if len(trigger_points) == 2:
        cv2.arrowedLine(overlay,
                         tuple(map(int, trigger_points[0])),
                         tuple(map(int, trigger_points[1])),
                         (0, 0, 255), 2, tipLength=0.3)

    # 状态文字（黑色描边 + 白色文字）
    info = []
    info.append(f"ROI points: {len(points)}  |  Trigger: {len(trigger_points)}")
    if current_chute_id:
        info.append(f"Current: {current_chute_id} ({current_chute_name})")
    info.append(f"Saved ROIs: {len(rois)}")
    info.append("Keys: [click]=add | [c]=done ROI | [l]=trigger | [n]=next | [s]=save")
    for j, txt in enumerate(info):
        cv2.putText(overlay, txt, (10, 25 + j * 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(overlay, txt, (10, 25 + j * 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 半透明叠加
    alpha = 0.3
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def mouse_callback(event, x, y, flags, param):
    global points, trigger_points, drawing_trigger
    if drawing_trigger:
        if event == cv2.EVENT_LBUTTONDOWN and len(trigger_points) < 2:
            trigger_points.append([x, y])
            print(f"  触发线点 {len(trigger_points)}: ({x}, {y})")
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"  ROI 顶点 {len(points)}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            removed = points.pop()
            print(f"  删除最后一个顶点: {removed}")


def get_chute_info():
    """让用户输入格口 ID 和名称（控制台交互）"""
    global current_chute_id, current_chute_name
    print("\n请输入格口信息 (Enter 跳过使用默认值):")
    chute_id = input(f"  格口 ID (默认 {current_chute_id}): ").strip()
    if chute_id:
        current_chute_id = chute_id
    chute_name = input(f"  格口名称 (默认 {current_chute_name}): ").strip()
    if chute_name:
        current_chute_name = chute_name
    print(f"  当前格口: {current_chute_id} ({current_chute_name})")


def save_rois(rois, config_path):
    """将标定好的 ROI 保存到 cameras.yaml"""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"cameras": []}

    # 使用第一个相机（或创建）
    if not cfg.get("cameras"):
        cfg["cameras"] = [{"id": "cam-001", "name": "主相机", "source": 0, "chutes": []}]
    cam = cfg["cameras"][0]

    # 转换 rois -> chutes 配置
    chutes = []
    for r in rois:
        chute = {
            "id": r["id"],
            "name": r["name"],
            "roi": r["polygon"],
        }
        if r.get("trigger_line"):
            chute["entry_line"] = r["trigger_line"]
        chutes.append(chute)

    cam["chutes"] = chutes

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ 已保存 {len(rois)} 个格口 ROI 到 {config_path}")
    print("\n生成的配置:")
    for r in rois:
        tl = "有" if r.get('trigger_line') else "无"
        print(f"  - {r['id']} ({r['name']}): ROI={len(r['polygon'])}个点  trigger={tl}")


def main():
    global cap, frame, clone, points, trigger_points, drawing_trigger
    global current_chute_id, current_chute_name, rois

    if len(sys.argv) < 2:
        print("用法: python roi_calibrate.py <video_path> [config/cameras.yaml]")
        sys.exit(1)

    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/cameras.yaml"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        sys.exit(1)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n视频尺寸: {w}x{h}, 帧率: {fps}")
    print("\n===== 操作说明 =====")
    print("  左键点击    : 添加 ROI 多边形顶点")
    print("  右键点击    : 删除最后一个顶点")
    print("  按 'c' 键   : 完成当前 ROI，然后输入格口信息")
    print("  按 'l' 键   : 进入触发线绘制模式（点2个点）")
    print("  按 'n' 键   : 保存当前格口，开始下一个")
    print("  按 's' 键   : 保存所有 ROI 并退出")
    print("  按 'q' 键   : 不保存退出")
    print("====================\n")

    clone = frame.copy()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    # 缩放窗口适应屏幕
    scale = min(1.0, 1400 / w, 820 / h)
    cv2.resizeWindow(WINDOW_NAME, int(w * scale), int(h * scale))

    get_chute_info()

    while True:
        disp = draw_overlay(clone.copy())
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("退出，不保存。")
            break

        elif key == ord('c'):
            # 完成当前 ROI
            if len(points) < 3:
                print(f"  ROI 至少需要 3 个顶点！当前: {len(points)}")
                continue
            print(f"  ROI 完成，共 {len(points)} 个顶点")
            print("  （按 'n' 保存此格口，或按 'l' 先画触发线）")

        elif key == ord('l'):
            drawing_trigger = True
            trigger_points = []
            print("  触发线模式: 请在画面中点击 2 个点（起点->终点）")

        elif key == ord('n'):
            # 保存当前格口 ROI
            if len(points) < 3:
                print("  当前 ROI 顶点不足 3 个，无法保存")
                continue
            roi_data = {
                "id": current_chute_id,
                "name": current_chute_name,
                "polygon": [list(map(int, p)) for p in points],
                "trigger_line": [list(map(int, p)) for p in trigger_points] if len(trigger_points) == 2 else None
            }
            rois.append(roi_data)
            print(f"  ✅ 已保存格口 {current_chute_id}，共 {len(rois)} 个格口")
            # 重置
            points = []
            trigger_points = []
            drawing_trigger = False
            get_chute_info()

        elif key == ord('s'):
            # 如果还有未保存的 ROI，先保存
            if len(points) >= 3:
                roi_data = {
                    "id": current_chute_id,
                    "name": current_chute_name,
                    "polygon": [list(map(int, p)) for p in points],
                    "trigger_line": [list(map(int, p)) for p in trigger_points] if len(trigger_points) == 2 else None
                }
                rois.append(roi_data)
                print(f"  已保存格口 {current_chute_id}")
            save_rois(rois, config_path)
            break

        elif key == 13:  # Enter 键，完成触发线
            if drawing_trigger and len(trigger_points) == 2:
                drawing_trigger = False
                print("  触发线绘制完成")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
