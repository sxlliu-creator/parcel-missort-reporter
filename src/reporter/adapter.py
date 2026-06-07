"""
WCS 上报适配器

设计为可插拔架构：
- HTTPReporter: HTTP POST 上报
- 后续可扩展 MQTTReporter、TCPReporter 等

抽象基类定义统一接口，换协议不影响业务逻辑
"""
import httpx
import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import asdict

from loguru import logger

from ..detector.engine import DetectionEvent
from ..event.buffer import EventBuffer
from ..loader import ReporterConfig


class BaseReporter(ABC):
    """上报适配器抽象基类"""

    @abstractmethod
    def send(self, events: List[DetectionEvent]) -> bool:
        """发送事件列表，返回是否成功"""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """检查与 WCS 的连接是否正常"""
        ...


class HTTPReporter(BaseReporter):
    """HTTP POST 上报适配器"""

    def __init__(self, hub_url: str, timeout: float = 5.0, max_retries: int = 3):
        self.hub_url = hub_url.rstrip("/") + "/api/events"
        self.health_url = hub_url.rstrip("/") + "/api/health"
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def send(self, events: List[DetectionEvent]) -> bool:
        """发送事件到 WCS/Hub"""
        if not events:
            return True

        payload = [asdict(e) for e in events]

        for attempt in range(self.max_retries):
            try:
                resp = self._get_client().post(
                    self.hub_url,
                    json={"events": payload},
                )
                if resp.status_code == 200:
                    return True
                logger.warning(
                    f"上报返回非200: {resp.status_code}, "
                    f"第{attempt+1}次重试"
                )
            except Exception as e:
                logger.warning(f"上报失败: {e}, 第{attempt+1}次重试")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))

        return False

    def health_check(self) -> bool:
        try:
            resp = self._get_client().get(self.health_url)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


class EventDispatcher:
    """
    事件分发器
    组合 EventBuffer（本地缓冲） + BaseReporter（上报适配器）
    """

    def __init__(self, buffer: EventBuffer, reporter: BaseReporter,
                 config: ReporterConfig):
        self.buffer = buffer
        self.reporter = reporter
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._batch_size = 50

    def push_event(self, event: DetectionEvent):
        """接收检测事件，写入缓冲"""
        self.buffer.push(event)

    def start(self):
        """启动后台发送线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._thread.start()
        logger.info("事件分发器已启动")

    def stop(self):
        """停止发送线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("事件分发器已停止")

    def _sender_loop(self):
        """后台发送循环"""
        while self._running:
            try:
                # 从缓冲取事件
                events = self.buffer.pop(limit=self._batch_size)
                if not events:
                    time.sleep(0.1)
                    continue

                # 转换为 DetectionEvent 列表
                detection_events = [
                    DetectionEvent(**{k: v for k, v in e.items() if k != "db_id"})
                    for e in events
                ]

                # 发送
                success = self.reporter.send(detection_events)

                db_ids = [e["db_id"] for e in events]
                if success:
                    self.buffer.mark_sent(db_ids)
                else:
                    self.buffer.mark_failed(db_ids)
                    time.sleep(1.0)  # 失败后等一会再试

            except Exception as e:
                logger.error(f"发送循环异常: {e}")
                time.sleep(1.0)

    def pending_count(self) -> int:
        return self.buffer.pending_count()
