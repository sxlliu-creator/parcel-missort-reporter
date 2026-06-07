"""
ROI 区域配置工具

交互式标定格口检测区域：
- 从摄像头取一帧静态画面
- 鼠标点击定义每个格口的 ROI 多边形和 entry_line
- 导出为 cameras.yaml 格式

操作说明：
- 鼠标左键：添加顶点
- 按 'n'：完成当前 ROI，进入下一个格口
- 按 's'：保存配置
- 按 'r'：重置当前多边形
- 按 'q'：退出
"""
import cv2
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import List, Tuple


class ROIEditor:
    """交互式 ROI 编辑器"""

    def __init__(self, camera_id: int, camera_name: str = "CAM-001"):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        # 读取一帧作为背景
        ret, self.bg_frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头画面")
        self.cap.release()

        self.frame = self.bg_frame.copy()
        self.h, self.w = self.frame.shape[:2]

        # ROI 数据
        self.chutes: List[dict] = []
        self.current_roi: List[Tuple[int, int]] = []
        self.current_entry_line: List[Tuple[int, int]] = []
        self.drawing_entry_line = False  # 当前在画 entry_line 还是 ROI
        self.current_name = ""

        print(f"摄像头分辨率: {self.w}x{self.h}")
        print("操作说明:")
        print("  左键点击: 添加顶点")
        print("  按 'n': 完成当前格口，输入格口名称")
        print("  按 's': 保存配置到 config/cameras.yaml")
        print("  按 'r': 重置当前多边形")
        print("  按 'q': 退出")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing_entry_line:
                self.current_roi.append((x, y))
                print(f"  ROI 顶点 {len(self.current_roi)}: ({x}, {y})")
            else:
                self.current_entry_line.append((x, y))
                print(f"  Entry line 点 {len(self.current_entry_line)}: ({x}, {y})")

    def run(self):
        cv2.namedWindow("ROI Editor")
        cv2.setMouseCallback("ROI Editor", self.mouse_callback)

        while True:
            display = self.frame.copy()

            # 绘制已有格口
            for chute in self.chutes:
                if "roi" in chute and len(chute["roi"]) >= 2:
                    pts = np.array(chute["roi"], np.int32)
                    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                    # 标注名称
                    if pts.size > 0:
                        cx = int(pts[:, 0].mean())
                        cy = int(pts[:, 1].mean())
                        cv2.putText(display, chute.get("name", chute["id"]),
                                   (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (0, 255, 0), 1)

                if "entry_line" in chute and len(chute["entry_line"]) == 2:
                    p1 = tuple(chute["entry_line"][0])
                    p2 = tuple(chute["entry_line"][1])
                    cv2.line(display, p1, p2, (0, 0, 255), 2)

            # 绘制当前 ROI
            if len(self.current_roi) >= 2:
                pts = np.array(self.current_roi, np.int32)
                cv2.polylines(display, [pts], False, (255, 255, 0), 2)
            for pt in self.current_roi:
                cv2.circle(display, pt, 4, (0, 255, 255), -1)

            # 绘制当前 entry_line
            if len(self.current_entry_line) == 2:
                cv2.line(display, self.current_entry_line[0],
                        self.current_entry_line[1], (0, 165, 255), 2)
            for pt in self.current_entry_line:
                cv2.circle(display, pt, 4, (0, 165, 255), -1)

            # 状态提示
            status = "ROI" if not self.drawing_entry_line else "Entry Line"
            cv2.putText(display, f"Mode: {status} | Chutes: {len(self.chutes)} | "
                       f"Points: {len(self.current_roi)}",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("ROI Editor", display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('n'):
                # 完成当前格口
                if len(self.current_roi) < 3:
                    print("  [错误] ROI 至少需要 3 个顶点")
                    continue
                if not self.drawing_entry_line:
                    # ROI 画完，切换到 entry_line 模式
                    self.drawing_entry_line = True
                    print("\n  现在请点击两个点定义 entry_line（判定线）")
                    print("  第一个点 → 第二个点，方向从第一点到第二点为 'enter'")
                elif len(self.current_entry_line) == 2:
                    # entry_line 也画完了，保存
                    name = input("  格口名称: ").strip()
                    if not name:
                        name = f"CHUTE-{len(self.chutes)+1:02d}"

                    self.chutes.append({
                        "id": f"CHUTE-{len(self.chutes)+1:02d}",
                        "name": name,
                        "roi": [list(p) for p in self.current_roi],
                        "entry_line": [list(p) for p in self.current_entry_line],
                        "direction": "enter",
                    })
                    print(f"  ✓ 已添加格口: {name}")
                    print(f"    ROI: {self.current_roi}")
                    print(f"    Entry Line: {self.current_entry_line}")
                    self.current_roi = []
                    self.current_entry_line = []
                    self.drawing_entry_line = False
                else:
                    print("  [错误] entry_line 需要 2 个点")

            elif key == ord('r'):
                self.current_roi = []
                self.current_entry_line = []
                self.drawing_entry_line = False
                print("  已重置")

            elif key == ord('s'):
                self.save_config()

        cv2.destroyAllWindows()

    def save_config(self):
        """保存为 cameras.yaml"""
        output = {
            "cameras": [{
                "id": self.camera_id,
                "name": self.camera_name,
                "source": self.camera_id,
                "width": self.w,
                "height": self.h,
                "fps": 25,
                "chutes": self.chutes,
            }]
        }

        config_dir = Path(__file__).parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        output_path = config_dir / "cameras.yaml"

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, allow_unicode=True, default_flow_style=False)

        print(f"\n✓ 配置已保存到: {output_path}")
        print(f"  共 {len(self.chutes)} 个格口")


def main():
    camera_id = 0
    camera_name = "CAM-001"
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            camera_id = sys.argv[1]
    if len(sys.argv) > 2:
        camera_name = sys.argv[2]

    print(f"打开摄像头 {camera_id}...")
    try:
        editor = ROIEditor(camera_id, camera_name)
        editor.run()
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
