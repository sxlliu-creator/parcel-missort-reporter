"""
中央汇聚服务 (Hub)

职责：
1. 接收各边缘节点上报的包裹进入事件
2. 转发给 WCS（或 WCS 模拟端点）
3. 提供状态监控 API
"""
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import threading
import time

from loguru import logger

CST = timezone(timedelta(hours=8))


# -------- 数据模型 --------

class ParcelEvent(BaseModel):
    """单个包裹事件"""
    chute_id: str
    chute_name: str = ""
    camera_id: str
    timestamp: str
    parcel_id: Optional[str] = None
    confidence: float = 0.0
    track_id: int = -1
    parcel_shape: Optional[Dict] = None


class EventBatch(BaseModel):
    """批量事件"""
    node_id: str = "unknown"
    events: List[ParcelEvent]


# -------- 统计收集器 --------

class HubStats:
    """内存统计：按节点、格口记录事件计数"""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_events = 0
        self.node_stats: Dict[str, int] = defaultdict(int)
        self.chute_stats: Dict[str, int] = defaultdict(int)
        self.recent_events: List[dict] = []  # 最近 100 条
        self.start_time = time.time()

    def record(self, node_id: str, events: List[ParcelEvent]):
        with self._lock:
            self.total_events += len(events)
            self.node_stats[node_id] += len(events)
            for e in events:
                self.chute_stats[e.chute_id] += 1

            # 保留最近 100 条
            for e in events:
                self.recent_events.append({
                    "chute_id": e.chute_id,
                    "camera_id": e.camera_id,
                    "timestamp": e.timestamp,
                    "node_id": node_id,
                    "confidence": e.confidence,
                })
            if len(self.recent_events) > 100:
                self.recent_events = self.recent_events[-100:]

    def snapshot(self) -> dict:
        with self._lock:
            uptime = time.time() - self.start_time
            return {
                "uptime_seconds": round(uptime, 1),
                "total_events": self.total_events,
                "node_stats": dict(self.node_stats),
                "chute_stats": dict(self.chute_stats),
                "recent_events": list(reversed(self.recent_events[-20:])),
            }


# -------- FastAPI 应用 --------

stats = HubStats()

app = FastAPI(
    title="Parcel Missort Reporter Hub",
    description="交叉带分拣机错分报告系统 - 中央汇聚服务",
    version="0.1.0",
)

# WCS 转发客户端
wcs_client: Optional[httpx.AsyncClient] = None
wcs_endpoint: str = ""


@app.on_event("startup")
async def startup():
    global wcs_client
    wcs_client = httpx.AsyncClient(timeout=10.0)
    logger.info("Hub 服务已启动")


@app.on_event("shutdown")
async def shutdown():
    if wcs_client:
        await wcs_client.aclose()
    logger.info("Hub 服务已关闭")


# -------- API 端点 --------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "parcel-missort-hub"}


@app.post("/api/events")
async def receive_events(batch: EventBatch):
    """接收边缘节点上报的事件批次"""
    if not batch.events:
        return {"status": "ok", "received": 0}

    # 记录统计
    stats.record(batch.node_id, batch.events)

    logger.info(
        f"收到 {batch.node_id} 的 {len(batch.events)} 个事件"
    )

    # 转发给 WCS（如果配置了）
    forwarded = 0
    if wcs_endpoint and wcs_client:
        try:
            payload = [e.model_dump() for e in batch.events]
            resp = await wcs_client.post(
                wcs_endpoint,
                json={"events": payload, "node_id": batch.node_id},
            )
            if resp.status_code == 200:
                forwarded = len(batch.events)
            else:
                logger.warning(f"WCS 转发返回 {resp.status_code}")
        except Exception as e:
            logger.error(f"WCS 转发失败: {e}")

    return {
        "status": "ok",
        "received": len(batch.events),
        "forwarded": forwarded,
    }


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    return stats.snapshot()


@app.get("/api/stats/nodes")
async def get_node_stats():
    """各节点统计"""
    return {"node_stats": dict(stats.node_stats)}


@app.get("/api/stats/chutes")
async def get_chute_stats():
    """各格口统计"""
    return {"chute_stats": dict(stats.chute_stats)}


def set_wcs_endpoint(endpoint: str):
    """设置 WCS 转发地址"""
    global wcs_endpoint
    wcs_endpoint = endpoint
    logger.info(f"WCS 转发地址已设置: {endpoint}")


def create_app(wcs_url: str = "") -> FastAPI:
    """工厂函数：创建应用并配置 WCS 地址"""
    if wcs_url:
        set_wcs_endpoint(wcs_url)
    return app
