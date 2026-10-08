"""学科列表接口（``GET /api/subjects``）。

对应《教师智能问答系统技术文档》6.3.3 节。

响应中的 ``doc_count`` 来自 Milvus 按 ``subject_code`` 的分组统计，
在 :class:`SubjectService` 中缓存 5 分钟（技术文档 2.2.3 的
``subject:{code}:stats`` 缓存策略），避免每次请求都全量扫描向量库。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_subject_service
from app.services.subject_service import SubjectService
from utils.exceptions import ErrorCode, SmartVoyageError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["subjects"])


@router.get(
    "/subjects",
    summary="学科列表",
    description="返回全部可用学科及各学科的知识块数量（doc_count）。",
)
async def list_subjects(
    subject_service: SubjectService = Depends(get_subject_service),
) -> dict:
    """学科列表（接口 6.3.3）。

    第一项固定为「全学科」（``code=null``，``is_default=true``），
    其 ``doc_count`` 为各学科之和。
    """
    try:
        return await subject_service.list_subjects()
    except Exception as exc:  # noqa: BLE001 - 统一转换为业务异常
        logger.error("学科列表查询失败：%s", exc)
        raise SmartVoyageError(
            ErrorCode.DB_ERROR, "学科列表查询失败，请稍后重试", detail=str(exc)
        ) from exc
