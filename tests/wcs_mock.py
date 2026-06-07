"""
WCS 模拟端点

用于测试时模拟 WCS 接收事件，可查看接收到的所有事件记录
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json

CST = timezone(timedelta(hours=8))

app = FastAPI(
    title="WCS Mock Endpoint",
    description="WCS 模拟端点 — 用于测试错分报告系统",
    version="0.1.0",
)

# 接收记录
received_events: List[dict] = []
event_count_by_node: Dict[str, int] = defaultdict(int)
event_count_by_chute: Dict[str, int] = defaultdict(int)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "wcs-mock"}


@app.post("/api/events")
async def receive_events(request: Request):
    """接收事件"""
    body = await request.json()
    events = body.get("events", [])
    node_id = body.get("node_id", "unknown")

    now = datetime.now(CST).isoformat()
    for e in events:
        e["received_at"] = now
        e["forward_node"] = node_id
        event_count_by_node[node_id] += 1
        event_count_by_chute[e.get("chute_id", "?")] += 1

    received_events.extend(events)
    # 只保留最近 10000 条
    if len(received_events) > 10000:
        del received_events[:-10000]

    print(f"\n[WCS Mock] 收到 {len(events)} 个事件 (来自 {node_id})")
    for e in events:
        print(f"  → 格口 {e.get('chute_id')} @ {e.get('timestamp')}")

    return {"status": "ok", "received": len(events)}


@app.get("/api/events")
async def list_events(limit: int = 20, offset: int = 0):
    """查看最近的事件"""
    total = len(received_events)
    items = received_events[-offset-limit:-offset] if offset else received_events[-limit:]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": list(reversed(items)),
    }


@app.get("/api/events/count")
async def count_events():
    """事件统计"""
    return {
        "total": len(received_events),
        "by_node": dict(event_count_by_node),
        "by_chute": dict(event_count_by_chute),
    }


@app.delete("/api/events")
async def clear_events():
    """清空事件记录"""
    global received_events
    received_events.clear()
    event_count_by_node.clear()
    event_count_by_chute.clear()
    return {"status": "ok", "message": "已清空"}
