# 交叉带分拣机错分报告系统

基于视觉检测的包裹格口进入事件上报系统。

## 架构

```
[USB/工业相机] ──→ [边缘节点] ──→ [中央汇聚 Hub] ──→ [WCS]
   (持续视频流)       (帧差+YOLO+ROI)      (接收+转发)      (最终接收方)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 标定格口 ROI（可选，有默认配置）

```bash
python src/config_editor/roi_editor.py 0 CAM-001
```

用鼠标点击画面标定每个格口的检测区域和进入判定线，按 `s` 保存。

### 3. 启动 WCS 模拟端点（测试用）

```bash
python run_wcs_mock.py --port=9900
```

### 4. 启动中央汇聚服务

```bash
python run_hub.py --port=8800 --wcs=http://localhost:9900/api/events
```

### 5. 启动边缘节点

```bash
python run_edge_node.py
```

访问 http://localhost:8800/docs 查看 API 文档。

## 消息格式

上报给 WCS 的事件格式（JSON）：

```json
{
  "events": [
    {
      "chute_id": "CHUTE-A01",
      "chute_name": "格口 A01",
      "camera_id": "CAM-001",
      "timestamp": "2026-06-07T16:53:43.218+08:00",
      "parcel_id": null,
      "confidence": 0.94,
      "track_id": 1
    }
  ],
  "node_id": "edge-node-01"
}
```

## 目录结构

```
parcel-missort-reporter/
├── config/                  # 配置文件
│   ├── cameras.yaml         # 相机 + 格口 ROI 配置
│   └── system.yaml          # 系统参数
├── src/
│   ├── detector/            # 检测引擎
│   │   └── engine.py       # 帧差 + YOLO + ROI 进入判定
│   ├── event/               # 事件缓冲
│   │   └── buffer.py       # SQLite 本地队列
│   ├── reporter/            # WCS 上报适配器
│   │   └── adapter.py      # HTTP/MQTT/TCP 可插拔
│   ├── hub/                 # 中央汇聚服务
│   │   └── server.py       # FastAPI 接收 + 转发
│   └── config_editor/      # ROI 标定工具
│       └── roi_editor.py
├── tests/
│   └── wcs_mock.py        # WCS 模拟端点
├── run_edge_node.py         # 边缘节点主程序
├── run_hub.py              # 汇聚服务启动脚本
├── run_wcs_mock.py         # WCS 模拟端点启动脚本
└── requirements.txt
```

## 注意事项

- CPU 推理使用 YOLOv8n，单核约处理 8-12 格口
- 200+ 格口建议部署 20+ 个边缘节点
- 默认 `classes=[0]` 只检测人形，正式运行需改为包裹类别
- 工业相机接入时修改 `cameras.yaml` 中的 `source` 为 RTSP URL
