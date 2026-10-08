"""历史接口：查询与清空。

对应《教师智能问答系统技术文档》：
- 6.3.5 节：``GET /api/history``（**游标分页**）
- 6.3.6 节：``DELETE /api/history``

为什么用游标（cursor）而不是 OFFSET：
    ``LIMIT 20 OFFSET 10000`` 需要扫描并丢弃前 10000 行，数据量增长后
    性能急剧劣化。游标分页（``record_id < cursor``）走 ``idx_session_time``
    索引范围扫描，10 万级记录仍保持毫秒级，满足 PRD「历史查询 < 500ms」。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_db, get_session_service
from app.models.schemas import ClearHistoryResponse, HistoryResponse
from app.services.session_service import SessionService
from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    MySQLClient,
)
from utils.exceptions import ErrorCode, SmartVoyageError
from utils.logger import get_logger
from app.api.deps import get_cache

logger = get_logger(__name__)

router = APIRouter(tags=["history"])


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="查询对话历史",
    description="按时间倒序返回会话历史，使用游标分页以支持滚动加载。",
)
async def get_history(
    session_id: str = Query(description="会话 ID（UUIDv4）"),
    limit: int = Query(
        default=DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description=f"每页条数，1~{MAX_PAGE_LIMIT}",
    ),
    cursor: int | None = Query(
        default=None,
        description="游标：上一页返回的 next_cursor；首页不传",
    ),
    db: MySQLClient = Depends(get_db),
    session_service: SessionService = Depends(get_session_service),
) -> HistoryResponse:
    """查询历史（接口 6.3.5）。

    会话不存在时返回 ``SESSION_NOT_FOUND``，避免用任意 ID 探测他人会话。
    """
    await session_service.ensure_exists(session_id)

    result = await db.get_history(session_id=session_id, cursor=cursor, limit=limit)
    logger.info(
        "历史查询：session_id=%s, cursor=%s, 返回 %s 条, has_more=%s",
        session_id, cursor, len(result["records"]), result["has_more"],
    )
    return HistoryResponse(**result)


@router.delete(
    "/history",
    response_model=ClearHistoryResponse,
    summary="清空对话历史",
    description="清空指定会话的全部问答记录，接口幂等。",
)
async def clear_history(
    session_id: str = Query(description="会话 ID（UUIDv4）"),
    db: MySQLClient = Depends(get_db),
    cache: RedisClient = Depends(get_cache),
    session_service: SessionService = Depends(get_session_service),
) -> ClearHistoryResponse:
    """清空历史（接口 6.3.6）。

    幂等设计：重复调用仍返回成功，避免前端重试时看到报错。
    """
    await session_service.ensure_exists(session_id)

    try:
        cleared = await db.clear_history(session_id)
        # Redis 里的会话快照可能残留旧轮次信息，一并清掉保持与库一致
        await cache.delete_session_meta(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("清空历史失败：session_id=%s, err=%s", session_id, exc)
        raise SmartVoyageError(
            ErrorCode.DB_ERROR, "清空历史失败，请稍后重试", detail=str(exc)
        ) from exc

    return ClearHistoryResponse(
        success=True,
        cleared_count=cleared,
        message="对话历史已清空",
    )
