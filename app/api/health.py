"""健康检查接口（``GET /api/health``）。

对应《教师智能问答系统技术文档》6.3.1 节。

聚合状态判定（文档 6.3.1 表格）：

==================  ====================================  ==========  ==============
状态                 判定条件                               HTTP 状态码  用途
==================  ====================================  ==========  ==============
``healthy``         全部组件正常                             200         正常放行
``degraded``        LLM 或 Milvus 异常，MySQL/Redis 正常     200         告警但不摘流量
``unhealthy``       MySQL 或 Redis 异常                      503         负载均衡摘除实例
==================  ====================================  ==========  ==============

为什么 MySQL/Redis 属强依赖而 Milvus/LLM 不是：
    会话与历史是**必须**落库的，缺了这两个服务整个问答链路没有意义；
    而 Milvus/LLM 故障时系统仍可用降级策略提供有标注的通用答案（技术文档 3.9）。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_cache, get_db, get_llm, get_milvus
from app.models.schemas import HealthResponse
from config.settings import get_config
from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import MySQLClient
from infrastructure.llm.qwen_client import QwenClient
from infrastructure.vector_db.milvus_client import MilvusClient

router = APIRouter(tags=["health"])

# 服务版本号（与技术文档 6.3.1 响应中的 version 对应）
SERVICE_VERSION = "1.0.0"


async def _timed(coro) -> tuple[str, int]:
    """执行探测并返回 ``(状态, 耗时毫秒)``。

    健康检查本身必须**永不抛异常**：探针挂了会掩盖真实状态。
    """
    started = time.perf_counter()
    try:
        status = await coro
    except Exception:  # noqa: BLE001
        status = "disconnected"
    return status, int((time.perf_counter() - started) * 1000)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="系统健康检查",
    description="聚合探测 MySQL / Redis / Milvus / LLM 四个组件，返回整体状态。",
)
async def health_check(
    response: Response,
    db: MySQLClient = Depends(get_db),
    cache: RedisClient = Depends(get_cache),
    milvus: MilvusClient = Depends(get_milvus),
    llm: QwenClient = Depends(get_llm),
) -> HealthResponse:
    """探测各依赖组件并聚合出整体健康状态。"""
    cfg = get_config()

    # 四个探测并发执行：串行探测在依赖超时的场景下会累加延迟，
    # 可能触发探针自身的超时而被误判为不健康。
    # Milvus 客户端是同步库，用 to_thread 移出事件循环。
    results = await asyncio.gather(
        _timed(db.health_check()),
        _timed(cache.health_check()),
        _timed(asyncio.to_thread(milvus.health_check)),
        _timed(llm.health_check()),
        return_exceptions=True,
    )

    def unpack(item, fallback: str) -> tuple[str, int]:
        """gather 若返回异常对象，按不可用处理。"""
        if isinstance(item, BaseException):
            return fallback, -1
        return item  # type: ignore[return-value]

    mysql_state, mysql_cost = unpack(results[0], "disconnected")
    redis_state, redis_cost = unpack(results[1], "disconnected")
    milvus_state, milvus_cost = unpack(results[2], "disconnected")
    llm_state, llm_cost = unpack(results[3], "unavailable")

    components = {
        "mysql": {
            "status": "up" if mysql_state == "connected" else "down",
            "latency_ms": mysql_cost,
        },
        "redis": {
            "status": "up" if redis_state == "connected" else "down",
            "latency_ms": redis_cost,
        },
        "milvus": {
            "status": "up" if milvus_state == "connected" else "down",
            "latency_ms": milvus_cost,
            "detail": {
                "collection": cfg.get("milvus", "collection_name"),
                "entities": milvus.num_entities if milvus_state == "connected" else 0,
            },
        },
        "llm": {
            "status": "up" if llm_state == "available" else "down",
            "latency_ms": llm_cost,
            "detail": {"model": llm.model},
        },
    }

    # ---- 聚合判定 ----
    core_up = (
        components["mysql"]["status"] == "up" and components["redis"]["status"] == "up"
    )
    all_up = core_up and all(
        components[name]["status"] == "up" for name in ("milvus", "llm")
    )

    if all_up:
        status = "healthy"
    elif core_up:
        status = "degraded"
    else:
        status = "unhealthy"
        # 强依赖不可用时返回 503，让负载均衡摘除该实例
        response.status_code = 503

    return HealthResponse(
        status=status,
        timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        version=SERVICE_VERSION,
        components=components,
    )
