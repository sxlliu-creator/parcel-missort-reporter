#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI 标定工具 — 纯 OpenCV 窗口操作，无需控制台输入

用法: python roi_calibrate.py <video_path> [config/cameras.yaml]

操作说明（全部在视频窗口内操作）:
  - 左键点击  : 添加 ROI 多边形顶点
  - 右键点击  : 删除最后一个顶点
  - c         : 闭合当前多边形（至少3个顶点）
  - l         : 画触发线（再点击2个点：从外向内方向）
  - n         : 保存当前格口，自动编号到下一个
  - s         : 保存所有格口并退出
  - q/Esc     : 不保存，退出
  - 空格      : 跳到下一帧（换参考画面）
  - d         : 删除上一个保存的格口
"""
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

# 自动编号
_AUTO_INDEX = 1
def next_chute_id():
    global _AUTO_INDEX
    cid = f"A{_AUTO_INDEX:02d}"
    _AUTO_INDEX += 1
    return cid

# 全局状态
g_points = []       # 当前 ROI 顶点
g_trigger = []      # 当前触发线
g_rois = []         # 已保存 [{id, polygon, trigger}]
g_mode = "roi"      # "roi" | "trigger"
g_base = None       # 参考帧
g_msg = ""          # 底部提示（3 秒后清除）
g_msg_timer = 0

WINDOW = "ROI Calibrator"


def show_msg(text, duration=90):  # 90 frames ≈ 3s at 30fps
    global g_msg, g_msg_timer
    g_msg = text
    g_msg_timer = duration


def draw_overlay(base):
    """绘制所有标注"""
    img = base.copy()
    h, w = img.shape[:2]

    # --- 已保存的格口 ---
    overlay = img.copy()
    for r in g_rois:
        poly = np.array(r["polygon"], np.int32)
        cv2.fillPoly(overlay, [poly], (0, 180, 0))
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    for r in g_rois:
        poly = np.array(r["polygon"], np.int32)
        cv2.polylines(img, [poly], True, (0, 255, 0), 2)
        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
        # 文字描边
        cv2.putText(img, r["id"], (cx - 25, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, r["id"], (cx - 25, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # 触发线
        tl = r.get("trigger")
        if tl and len(tl) == 2:
            cv2.arrowedLine(img, tuple(tl[0]), tuple(tl[1]),
                            (0, 0, 255), 2, tipLength=0.3)

    # --- 当前绘制中的 ROI（青色）---
    for pt in g_points:
        cv2.circle(img, tuple(pt), 6, (255, 200, 0), -1)
    if len(g_points) >= 2:
        pts = np.array(g_points, np.int32)
        cv2.polylines(img, [pts], False, (255, 200, 0), 2)
    if len(g_points) >= 3:
        cv2.polylines(img, [np.array(g_points, np.int32)], True,
                      (0, 220, 220), 2)

    # --- 当前触发线（红色）---
    for pt in g_trigger:
        cv2.circle(img, tuple(pt), 6, (0, 0, 255), -1)
    if len(g_trigger) == 2:
        cv2.arrowedLine(img, tuple(g_trigger[0]), tuple(g_trigger[1]),
                        (0, 0, 255), 2, tipLength=0.3)

    # --- 顶部状态栏 ---
    panel = np.zeros((80, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    mode_str = "[触发线模式] 点2个点" if g_mode == "trigger" else "[ROI模式]"
    status_lines = [
        f"{mode_str}  下一个: A{_AUTO_INDEX:02d}  已保存: {len(g_rois)}",
        f"ROI顶点: {len(g_points)}  |  触发线: {len(g_trigger)}/2",
        "左键+点  c闭合  l触发线  n保存+下一格口  s全部保存  q退出  空格换帧  d撤销上一格口",
    ]
    for i, line in enumerate(status_lines):
        cv2.putText(panel, line, (10, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- 底部消息 ---
    if g_msg_timer > 0:
        cv2.putText(img, g_msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5)
        cv2.putText(img, g_msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 竖排拼接：状态栏 + 主画面
    return np.vstack([panel, img])


def mouse_cb(event, x, y, flags, param):
    global g_points, g_trigger, g_mode
    if event != cv2.EVENT_LBUTTONDOWN and event != cv2.EVENT_RBUTTONDOWN:
        return

    # 减去状态栏高度
    real_y = y - 80
    if real_y < 0:
        return

    if event == cv2.EVENT_RBUTTONDOWN:
        if g_mode == "roi" and g_points:
            rm = g_points.pop()
            show_msg(f"已删除顶点 ({rm[0]},{rm[1]})")
        return

    # 左键
    if g_mode == "trigger":
        if len(g_trigger) < 2:
            g_trigger.append([x, real_y])
            if len(g_trigger) == 2:
                g_mode = "roi"
                show_msg("触发线完成，已切回 ROI 模式。按 'n' 保存此格口")
            else:
                show_msg(f"触发线点1: ({x},{real_y}), 请再点1个")
    else:
        g_points.append([x, real_y])
        show_msg(f"ROI 第{len(g_points)}点: ({x},{real_y})")


def save_config(config_path):
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
        chute = {"id": r["id"], "name": "格口" + r["id"], "roi": r["polygon"]}
        if r.get("trigger"):
            chute["entry_line"] = r["trigger"]
        chutes.append(chute)

    cfg["cameras"][0]["chutes"] = chutes
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n[OK] 已保存 {len(g_rois)} 个格口到: {path}")
    for r in g_rois:
        tl = "有触发线" if r.get("trigger") else "无触发线"
        print(f"  {r['id']}: {len(r['polygon'])}点  {tl}")


def main():
    global g_points, g_trigger, g_mode, g_rois, g_base, g_msg, g_msg_timer

    if len(sys.argv) < 2:
        print("用法: python roi_calibrate.py <video_path> [config/cameras.yaml]")
        sys.exit(1)

    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/cameras.yaml"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {fw}x{fh}, {total_frames}帧")

    # 读参考帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, g_base = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, g_base = cap.read()
    if not ret:
        print("[错误] 无法读取视频帧")
        sys.exit(1)

    # 缩放显示
    scale = min(1.0, 1200 / fw, 750 / fh)
    disp_w = max(1, int(fw * scale))
    disp_h = max(1, int(fh * scale))

    print("===== 操作说明 =====")
    print("  左键        : 加顶点")
    print("  右键        : 删最后一个顶点")
    print("  c           : 闭合多边形")
    print("  l           : 画触发线（再点2个点）")
    print("  n           : 保存当前->下一格口（自动编号）")
    print("  s           : 全部保存并退出")
    print("  空格        : 换参考帧")
    print("  d           : 删除上一个格口")
    print("  q / Esc     : 退出不保存")
    print("====================\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, disp_w, disp_h + 80)  # +80 状态栏
    cv2.setMouseCallback(WINDOW, mouse_cb)

    frame_idx = 10
    show_msg("请在画面上点击标记 ROI 顶点。按 c 闭合。")

    while True:
        disp = draw_overlay(g_base)
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(30) & 0xFF

        # 消息计时
        if g_msg_timer > 0:
            g_msg_timer -= 1
            if g_msg_timer == 0:
                g_msg = ""

        if key in (ord('q'), 27):
            print("退出，未保存。")
            break

        elif key == ord('c'):
            if len(g_points) < 3:
                show_msg(f"至少需要3个顶点（当前{len(g_points)}）")
            else:
                show_msg(f"ROI 已闭合（{len(g_points)}点）。按 l 画触发线 或 n 保存")

        elif key == ord('l'):
            g_mode = "trigger"
            g_trigger = []
            show_msg("请在格口入口处点2个点（包裹进入方向: 外->内）")

        elif key == ord('n'):
            if len(g_points) < 3:
                show_msg(f"顶点不足({len(g_points)})，请至少添加3个点")
                continue
            cid = next_chute_id()
            g_rois.append({
                "id": cid,
                "polygon": [[int(x), int(y)] for x, y in g_points],
                "trigger": [[int(x), int(y)] for x, y in g_trigger] if len(g_trigger) == 2 else None,
            })
            g_points = []
            g_trigger = []
            g_mode = "roi"
            show_msg(f"已保存 {cid}（{len(g_rois)}个格口）。继续标定下一个。")

        elif key == ord('d'):
            if g_rois:
                removed = g_rois.pop()
                global _AUTO_INDEX
                _AUTO_INDEX = max(1, _AUTO_INDEX - 1)
                show_msg(f"已删除 {removed['id']}")
            else:
                show_msg("无已保存的格口可删除")

        elif key == ord('s'):
            if len(g_points) >= 3:
                cid = next_chute_id()
                g_rois.append({
                    "id": cid,
                    "polygon": [[int(x), int(y)] for x, y in g_points],
                    "trigger": [[int(x), int(y)] for x, y in g_trigger] if len(g_trigger) == 2 else None,
                })
                g_points = []
                g_trigger = []
                g_mode = "roi"
            save_config(config_path)
            break

        elif key == 32:  # 空格
            frame_idx = (frame_idx + total_frames // 5) % total_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret2, f2 = cap.read()
            if ret2:
                g_base = f2
                show_msg(f"第 {frame_idx} 帧")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
