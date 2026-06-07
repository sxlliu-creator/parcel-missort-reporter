"""
配置加载器：从 YAML 文件读取相机配置和系统参数
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ChuteConfig:
    """单个格口配置"""
    id: str
    name: str
    roi: List[List[int]]  # 检测区域多边形顶点
    entry_line: List[List[int]]  # 进入判定线 [(x1,y1), (x2,y2)]
    direction: str = "enter"  # enter / exit


@dataclass
class CameraConfig:
    """单个相机配置"""
    id: str
    name: str
    source: str  # 摄像头索引（数字）或 RTSP URL
    width: int
    height: int
    fps: int
    chutes: List[ChuteConfig] = field(default_factory=list)


@dataclass
class DetectionConfig:
    frame_diff_threshold: int = 25
    frame_diff_ratio: float = 0.02
    yolo_conf: float = 0.3
    yolo_model: str = "yolov8n.pt"
    inference_interval: int = 1
    tracker: str = "bytetrack.yaml"
    cooldown_seconds: float = 3.0
    imgsz: int = 640
    device: str = "cpu"


@dataclass
class ReporterConfig:
    hub_url: str = "http://localhost:8800"
    timeout: float = 5.0
    max_retries: int = 3
    buffer_max_size: int = 10000
    buffer_db: str = "./data/event_buffer.db"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./logs/edge_node.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


@dataclass
class SystemConfig:
    node_id: str = "edge-node-01"
    node_name: str = "边缘节点01"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reporter: ReporterConfig = field(default_factory=ReporterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class HubConfig:
    host: str = "0.0.0.0"
    port: int = 8800
    wcs_endpoint: str = "http://localhost:9900/api/events"


@dataclass
class AppConfig:
    cameras: List[CameraConfig] = field(default_factory=list)
    system: SystemConfig = field(default_factory=SystemConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def _parse_chute(data: Dict[str, Any]) -> ChuteConfig:
    return ChuteConfig(
        id=data["id"],
        name=data.get("name", data["id"]),
        roi=data["roi"],
        entry_line=data["entry_line"],
        direction=data.get("direction", "enter"),
    )


def _parse_camera(data: Dict[str, Any]) -> CameraConfig:
    chutes = [_parse_chute(c) for c in data.get("chutes", [])]
    return CameraConfig(
        id=data["id"],
        name=data.get("name", data["id"]),
        source=data["source"],
        width=data.get("width", 1280),
        height=data.get("height", 720),
        fps=data.get("fps", 25),
        chutes=chutes,
    )


def _parse_detection(data: Dict[str, Any]) -> DetectionConfig:
    return DetectionConfig(**{k: v for k, v in data.items() if k in DetectionConfig.__dataclass_fields__})


def _parse_reporter(data: Dict[str, Any]) -> ReporterConfig:
    return ReporterConfig(**{k: v for k, v in data.items() if k in ReporterConfig.__dataclass_fields__})


def _parse_logging(data: Dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(**{k: v for k, v in data.items() if k in LoggingConfig.__dataclass_fields__})


def _parse_system(data: Dict[str, Any]) -> SystemConfig:
    return SystemConfig(
        node_id=data.get("node_id", "edge-node-01"),
        node_name=data.get("node_name", "边缘节点01"),
        detection=_parse_detection(data.get("detection", {})),
        reporter=_parse_reporter(data.get("reporter", {})),
        logging=_parse_logging(data.get("logging", {})),
    )


def _parse_hub(data: Dict[str, Any]) -> HubConfig:
    return HubConfig(**{k: v for k, v in data.items() if k in HubConfig.__dataclass_fields__})


def load_config(config_dir: str = "config") -> AppConfig:
    """加载完整配置"""
    config_dir = Path(config_dir)

    # 加载相机配置
    cameras_path = config_dir / "cameras.yaml"
    cameras_raw = {}
    if cameras_path.exists():
        with open(cameras_path, "r", encoding="utf-8") as f:
            cameras_raw = yaml.safe_load(f) or {}

    cameras = [_parse_camera(c) for c in cameras_raw.get("cameras", [])]

    # 加载系统配置
    system_path = config_dir / "system.yaml"
    system_raw = {}
    if system_path.exists():
        with open(system_path, "r", encoding="utf-8") as f:
            system_raw = yaml.safe_load(f) or {}

    system = _parse_system(system_raw.get("system", {}))
    hub = _parse_hub(system_raw.get("hub", {}))

    return AppConfig(cameras=cameras, system=system, hub=hub)
