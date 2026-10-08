"""问答接口：即时问答与流式问答（SSE）。

对应《教师智能问答系统技术文档》：
- 6.3.4 节：``POST /api/chat`` 即时问答
- 6.4 节：``POST /api/chat/stream`` 流式问答（事件协议 start/delta/citation/done/error）

两条路径共用 :class:`ChatService.prepare` 的检索与 Prompt 组装结果，
API 层只负责协议差异（一次性 JSON vs SSE 事件流），避免业务逻辑重复。
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import (
    get_cache_service,
    get_chat_service,
    get_client_ip,
    get_session_service,
)
from app.models.schemas import ChatRequest, ChatResponse, Reference
from app.services.cache_service import CacheService
from app.services.chat_service import ChatService, PreparedChat
from app.services.session_service import SessionService
from utils.exceptions import ErrorCode, LLMError, SmartVoyageError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])

# SSE 响应头（技术文档 6.4）
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 关键：禁止 Nginx 缓冲响应，否则流式内容会被攒批下发，
    # 首字延迟从 <1s 劣化到「等全部生成完才一次性吐出」
    "X-Accel-Buffering": "no",
}


def sse_event(event_type: str, **data) -> str:
    """按 SSE 协议格式化一条事件。

    格式：``data: {json}\\n\\n``（技术文档 6.4 的事件流示例）。
    ``ensure_ascii=False`` 保证中文不被转义成 ``\\uXXXX``，减小传输体积。
    """
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def to_reference_models(prepared: PreparedChat) -> list[Reference]:
    """把内部溯源结构转换为接口响应模型。

    Pydantic 会丢弃模型中未声明的字段（如内部用的完整 ``content``），
    因此这里直接传 dict 即可，不会把父块全文泄露给前端。
    """
    return [Reference(**ref) for ref in prepared.references]


# ---------------------------------------------------------------------------
# 即时问答（接口 6.3.4）
# ---------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="即时问答",
    description="一次性返回完整答案，适用于简单、明确的问题。",
)
async def chat(
    request: Request,
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
    cache_service: CacheService = Depends(get_cache_service),
) -> ChatResponse:
    """即时问答（PRD 2.2.2.1）。"""
    await _guard(request, payload.session_id, session_service, cache_service)

    started = time.perf_counter()

    # 阶段一：校验 → 问候分流 → 缓存 → 检索 → 组装 Prompt
    prepared = await chat_service.prepare(
        question=payload.question,
        session_id=payload.session_id,
        subject_code=payload.subject_code,
        use_history=payload.use_history,
    )

    # 阶段二：生成（问候/缓存/兜底三种情况不会调用 LLM）
    result = await chat_service.generate(prepared)
    result["response_time_ms"] = int((time.perf_counter() - started) * 1000)

    # 阶段三：落库 + 写缓存
    record_id = await chat_service.save_result(prepared, payload.session_id, result)

    return ChatResponse(
        record_id=record_id,
        answer=result["answer"],
        references=to_reference_models(prepared),
        response_time_ms=result["response_time_ms"],
        is_from_cache=bool(result.get("is_from_cache")),
        finish_reason=result.get("finish_reason", "stop"),
    )


# ---------------------------------------------------------------------------
# 流式问答（接口 6.4）
# ---------------------------------------------------------------------------
@router.post(
    "/chat/stream",
    summary="流式问答（SSE）",
    description="逐步推送答案增量，实现打字机效果；事件类型见技术文档 6.4。",
)
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
    cache_service: CacheService = Depends(get_cache_service),
) -> StreamingResponse:
    """流式问答（PRD 2.2.2.2）。

    注意：**准备阶段（检索）在返回 StreamingResponse 之前完成**。
    这样做有两个好处：
    1. 参数错误可以正常返回 400，而不是先建立 SSE 流再在流里报错；
    2. 前端收到第一个事件时，首字延迟只包含 LLM 生成时间（满足 < 1s 指标）。
    """
    await _guard(request, payload.session_id, session_service, cache_service)

    prepared = await chat_service.prepare(
        question=payload.question,
        session_id=payload.session_id,
        subject_code=payload.subject_code,
        use_history=payload.use_history,
    )

    async def event_generator() -> AsyncIterator[str]:
        """按技术文档 6.4 的事件协议推送。

        关于客户端断开的处理（技术文档 3.7.3）：
        前端关闭连接时，Starlette 会对本生成器抛出 ``GeneratorExit``。
        此时**不能再 yield**（否则报 "async generator ignored GeneratorExit"），
        但仍可 await——因此把落库放在 ``except GeneratorExit`` 分支里完成，
        保证「用户中途关掉页面，已生成的部分仍然进历史记录」。
        """
        started = time.perf_counter()
        collected: list[str] = []
        finish_reason = "stop"

        async def persist(reason: str) -> int:
            """落库并返回耗时。流式接口不返回 token 用量（由上游决定），
            成本统计以非流式接口的记录为准。"""
            result = {
                "answer": "".join(collected),
                "is_from_cache": False,
                "model_name": chat_service.llm.model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "response_time_ms": int((time.perf_counter() - started) * 1000),
                "finish_reason": reason,
            }
            try:
                await chat_service.save_result(prepared, payload.session_id, result)
            except Exception as exc:  # noqa: BLE001 - 落库失败不影响已推送内容
                logger.error("流式问答落库失败：%s", exc)
            return result["response_time_ms"]

        try:
            yield sse_event("start", session_id=payload.session_id)

            # ---- 无需调用 LLM 的情况（问候 / 缓存命中 / 兜底话术）----
            local = chat_service.local_response(prepared)
            if local is not None:
                # 分片推送以复用同一套前端渲染逻辑（短文本影响可忽略）
                for piece in _chunk_text(local["answer"]):
                    collected.append(piece)
                    yield sse_event("delta", content=piece)

                if prepared.references:
                    yield sse_event("citation", references=_references_payload(prepared))

                cost_ms = await persist("stop")
                yield sse_event(
                    "done",
                    response_time_ms=cost_ms,
                    finish_reason="stop",
                    is_from_cache=local.get("is_from_cache", False),
                )
                return

            # ---- 正常生成：逐 token 推送 ----
            try:
                async for delta in chat_service.llm.stream_chat(prepared.messages):
                    collected.append(delta)
                    yield sse_event("delta", content=delta)

                # 生成完成后再推溯源：此时正文已渲染完，前端补上引用卡片
                yield sse_event("citation", references=_references_payload(prepared))

            except LLMError as exc:
                # 生成中途失败：保留已产出的部分，标记 finish_reason=error
                logger.error("流式生成失败：%s", exc)
                finish_reason = "error"
                yield sse_event("error", code=exc.error_code.code, message=exc.message)
            except Exception:  # noqa: BLE001 - 兜底，避免流悬挂不关闭
                logger.exception("流式生成未知异常")
                finish_reason = "error"
                yield sse_event(
                    "error",
                    code=ErrorCode.INTERNAL_ERROR.code,
                    message="生成失败，请重试",
                )

            # 上游报了错又没产出任何内容时，标记为 error 而非 stop
            if not collected:
                finish_reason = "error"

            cost_ms = await persist(finish_reason)
            yield sse_event(
                "done",
                response_time_ms=cost_ms,
                finish_reason=finish_reason,
                is_from_cache=False,
            )

        except GeneratorExit:
            # 客户端断开：不能再 yield，但可以 await 落库
            logger.info("客户端中断流式问答，保存已生成内容：session_id=%s", payload.session_id)
            await persist("interrupted")
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _references_payload(prepared: PreparedChat) -> list[dict]:
    """溯源信息的 SSE 载荷（剔除内部字段，与 6.4 示例一致）。"""
    return [
        {
            "index": ref["index"],
            "subject_code": ref.get("subject_code"),
            "subject_name": ref.get("subject_name"),
            "source": ref.get("source"),
            "page_no": ref.get("page_no"),
            "score": ref.get("score"),
            "summary": ref.get("summary"),
        }
        for ref in prepared.references
    ]


def _chunk_text(text: str, size: int = 12) -> list[str]:
    """把整段文本切成小片，用于非 LLM 路径的「伪流式」推送。"""
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


async def _guard(
    request: Request,
    session_id: str,
    session_service: SessionService,
    cache_service: CacheService,
) -> None:
    """问答接口的公共前置校验：会话存在性 + 限流。

    :raises SmartVoyageError: ``SESSION_NOT_FOUND`` / ``RATE_LIMITED``
    """
    await session_service.ensure_exists(session_id)

    allowed = await cache_service.check_rate_limit(
        client_ip=get_client_ip(request), session_id=session_id
    )
    if not allowed:
        raise SmartVoyageError(
            ErrorCode.RATE_LIMITED,
            "提问过于频繁，请稍后再试",
            detail=f"session_id={session_id}",
        )
