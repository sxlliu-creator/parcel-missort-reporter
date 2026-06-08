#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式 ROI 标定工具
用法: python roi_calibrate.py <video_path> [config/cameras.yaml]

操作说明:
  左键点击    : 添加 ROI 多边形顶点
  右键点击    : 删除最后一个顶点
  c 键        : 完成当前 ROI（至少 3 个点）
  l 键        : 画触发线（点 2 个点后自动完成）
  n 键        : 保存当前格口，开始下一个（需在控制台输入格口 ID）
  s 键        : 保存所有 ROI 并退出
  q 键        : 不保存退出
  空格键      : 视频跳帧（换一帧参考）
"""

import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

# ---------- 全局状态 ----------
g_points = []           # 当前 ROI 顶点（绘制中）
g_trigger = []          # 当前触发线（最多 2 点）
g_rois = []             # 已保存的格口列表
g_mode = "roi"          # "roi" | "trigger"
g_chute_id = "A01"
g_chute_name = "格口A01"
g_need_input = False    # 是否需要在控制台输入格口信息

WINDOW = "ROI Calibration - Click=add point, c=complete, l=trigger, n=next, s=save (未响应)"


def draw_frame(base):
    """在 base 上绘制所有标注，返回新图（不修改 base）"""
    img = base.copy()
    overlay = img.copy()

    # 已保存的 ROI
    for roi in g_rois:
        poly = np.array(roi["polygon"], np.int32)
        cv2.fillPoly(overlay, [poly], (0, 180, 0))

    # 半透明叠加（正确方向：img 是背景，overlay 是前景）
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    for roi in g_rois:
        poly = np.array(roi["polygon"], np.int32)
        cv2.polylines(img, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            cv2.putText(img, roi["id"], (cx - 20 + dx, cy + 5 + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, roi["id"], (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if roi.get("trigger_line") and len(roi["trigger_line"]) == 2:
            p1 = tuple(roi["trigger_line"][0])
            p2 = tuple(roi["trigger_line"][1])
            cv2.arrowedLine(img, p1, p2, (0, 0, 255), 2, tipLength=0.35)

    # 当前绘制中的 ROI（青色）
    for pt in g_points:
        cv2.circle(img, tuple(pt), 5, (0, 255, 255), -1)
    if len(g_points) >= 2:
        pts = np.array(g_points, np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
    if len(g_points) >= 3:
        pts = np.array(g_points, np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 200), 1, cv2.LINE_AA)

    # 当前触发线（红色）
    for pt in g_trigger:
        cv2.circle(img, tuple(pt), 6, (0, 0, 255), -1)
    if len(g_trigger) == 2:
        cv2.arrowedLine(img, tuple(g_trigger[0]), tuple(g_trigger[1]),
                        (0, 0, 255), 2, tipLength=0.35)

    # 状态栏（顶部）
    mode_txt = "模式: 触发线（点2个点）" if g_mode == "trigger" else "模式: ROI多边形"
    lines = [
        f"{mode_txt}  |  当前格口: {g_chute_id}  |  已保存: {len(g_rois)} 个",
        f"ROI顶点: {len(g_points)}  触发线点: {len(g_trigger)}",
        "操作: [左键]加点 [右键]删点 [c]完成ROI [l]触发线 [n]下一格口 [s]保存 [空格]换帧",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 200), 1)

    return img


def on_mouse(event, x, y, flags, param):
    global g_points, g_trigger, g_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if g_mode == "trigger":
            if len(g_trigger) < 2:
                g_trigger.append([x, y])
                print(f"  触发线点 {len(g_trigger)}: ({x}, {y})")
                if len(g_trigger) == 2:
                    g_mode = "roi"
                    print("  触发线完成，已切换回 ROI 模式")
        else:
            g_points.append([x, y])
            print(f"  ROI 顶点 {len(g_points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if g_mode == "roi" and g_points:
            removed = g_points.pop()
            print(f"  删除顶点: {removed}")


def ask_chute_info():
    """在控制台获取格口信息（非阻塞：在主循环外调用）"""
    global g_chute_id, g_chute_name
    print(f"\n--- 请在此窗口输入格口信息 ---")
    val = input(f"  格口 ID (Enter 使用 [{g_chute_id}]): ").strip()
    if val:
        g_chute_id = val
    val = input(f"  格口名称 (Enter 使用 [{g_chute_name}]): ").strip()
    if val:
        g_chute_name = val
    print(f"  OK: {g_chute_id} / {g_chute_name}")


def save_all(config_path):
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    if not cfg.get("cameras"):
        cfg["cameras"] = [{"id": "CAM-001", "name": "主相机", "source": 0, "chutes": []}]

    chutes = []
    for r in g_rois:
        c = {"id": r["id"], "name": r["name"], "roi": r["polygon"]}
        if r.get("trigger_line"):
            c["entry_line"] = r["trigger_line"]
        chutes.append(c)

    cfg["cameras"][0]["chutes"] = chutes
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n[OK] 已保存 {len(g_rois)} 个格口到 {path}")
    for r in g_rois:
        tl = "有触发线" if r.get("trigger_line") else "无触发线"
        print(f"  {r['id']} ({r['name']}): {len(r['polygon'])} 点  {tl}")


def main():
    global g_points, g_trigger, g_mode, g_rois, g_chute_id, g_chute_name

    if len(sys.argv) < 2:
        print("用法: python roi_calibrate.py <video_path> [config/cameras.yaml]")
        sys.exit(1)

    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/cameras.yaml"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {w}x{h}, 共 {total_frames} 帧")
    print("===== 操作说明 =====")
    print("  左键        : 添加 ROI 顶点")
    print("  右键        : 撤销最后一个顶点")
    print("  c           : 完成当前多边形")
    print("  l           : 切换到触发线模式（再点 2 个点）")
    print("  n           : 保存当前格口，开始下一个")
    print("  s           : 保存所有格口并退出")
    print("  空格        : 跳到视频中间帧换参考画面")
    print("  q           : 退出不保存")
    print("====================")
    print("\n请先在控制台输入第一个格口信息：")
    ask_chute_info()

    # 取第 5 帧作为初始参考画面（跳过全黑的第一帧）
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(5, total_frames - 1))
    ret, base_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, base_frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        sys.exit(1)

    # 根据屏幕大小缩放显示（不缩放内部坐标）
    disp_scale = min(1.0, 1400 / w, 900 / h)
    disp_w = int(w * disp_scale)
    disp_h = int(h * disp_scale)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, disp_w, disp_h)
    cv2.setMouseCallback(WINDOW, on_mouse)

    frame_idx = 5

    while True:
        disp = draw_frame(base_frame)
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("退出，未保存。")
            break

        elif key == ord('c'):
            if len(g_points) < 3:
                print(f"  至少需要 3 个顶点，当前 {len(g_points)} 个")
            else:
                print(f"  ROI 完成（{len(g_points)} 个顶点）。按 'l' 画触发线，或 'n' 保存此格口。")

        elif key == ord('l'):
            g_mode = "trigger"
            g_trigger = []
            print("  触发线模式：请在格口入口处点击 2 个点（从外向内方向）")

        elif key == ord('n'):
            if len(g_points) < 3:
                print("  ROI 顶点不足 3 个，无法保存")
                continue
            roi_entry = {
                "id": g_chute_id,
                "name": g_chute_name,
                "polygon": [list(map(int, p)) for p in g_points],
                "trigger_line": [list(map(int, p)) for p in g_trigger] if len(g_trigger) == 2 else None,
            }
            g_rois.append(roi_entry)
            print(f"  [已保存] {g_chute_id}，共 {len(g_rois)} 个格口")
            g_points = []
            g_trigger = []
            g_mode = "roi"
            # 提示在控制台输入下一个格口信息
            print("\n请在控制台输入下一个格口信息（返回窗口前请先按 Enter 确认）：")
            ask_chute_info()

        elif key == ord('s'):
            # 如果当前还有未保存的 ROI，一并保存
            if len(g_points) >= 3:
                roi_entry = {
                    "id": g_chute_id,
                    "name": g_chute_name,
                    "polygon": [list(map(int, p)) for p in g_points],
                    "trigger_line": [list(map(int, p)) for p in g_trigger] if len(g_trigger) == 2 else None,
                }
                g_rois.append(roi_entry)
                print(f"  [已保存当前] {g_chute_id}")
            save_all(config_path)
            break

        elif key == 32:  # 空格：跳到另一帧
            frame_idx = (frame_idx + total_frames // 4) % total_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret2, f2 = cap.read()
            if ret2:
                base_frame = f2
                print(f"  跳到第 {frame_idx} 帧")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
