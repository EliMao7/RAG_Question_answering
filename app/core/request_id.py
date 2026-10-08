"""请求追踪中间件。

对应《教师智能问答系统技术文档》6.1 节「请求追踪」：

- 请求头 ``X-Request-Id`` 由客户端提供时沿用，缺省则服务端生成 UUID；
- 该 ID 写入 ``request.state``，供日志与错误响应使用；
- 响应头回传 ``X-Request-Id``，便于前端把用户反馈与后端日志关联起来。
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.logger import get_logger

logger = get_logger(__name__)

HEADER_NAME = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """为每个请求分配追踪 ID，并记录访问日志。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(HEADER_NAME) or str(uuid.uuid4())
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 未捕获异常在此记录，交由上层异常处理器返回统一错误体
            cost_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "请求异常 | request_id=%s | %s %s | %.1fms",
                request_id, request.method, request.url.path, cost_ms,
            )
            raise

        cost_ms = (time.perf_counter() - started) * 1000
        response.headers[HEADER_NAME] = request_id
        logger.info(
            "请求完成 | request_id=%s | %s %s | status=%s | %.1fms",
            request_id, request.method, request.url.path,
            response.status_code, cost_ms,
        )
        return response
