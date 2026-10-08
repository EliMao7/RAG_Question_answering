"""会话接口：创建会话。

对应《教师智能问答系统技术文档》6.3.2 节：``POST /api/session``。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_client_ip, get_session_service, get_subject_service
from app.models.schemas import SessionCreateRequest, SessionResponse
from app.services.session_service import SessionService
from app.services.subject_service import SubjectService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["session"])


@router.post(
    "/session",
    response_model=SessionResponse,
    summary="创建会话",
    description="生成 UUIDv4 会话标识，并返回欢迎语与可选学科列表。",
)
async def create_session(
    request: Request,
    payload: SessionCreateRequest | None = None,
    session_service: SessionService = Depends(get_session_service),
    subject_service: SubjectService = Depends(get_subject_service),
) -> SessionResponse:
    """创建新会话（接口 6.3.2）。

    请求体可省略；``subject_code`` 为可选，缺省或传 null 表示全学科。
    """
    raw_subject = payload.subject_code if payload else None
    subject_code = subject_service.validate(raw_subject)

    session = await session_service.create_session(
        subject_code=subject_code,
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    # 顺带返回学科列表：前端首屏即可渲染下拉菜单，省去一次额外请求
    subjects = await subject_service.list_subjects()
    options = [
        {"code": item["code"], "name": item["name"]}
        for item in subjects.get("subjects", [])
    ]

    logger.info("会话创建接口返回：session_id=%s", session["session_id"])
    return SessionResponse(
        session_id=session["session_id"],
        subject_code=session["subject_code"],
        created_at=session["created_at"],
        welcome_message=session["welcome_message"],
        subjects=options,
    )
