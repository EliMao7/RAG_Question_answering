"""依赖注入：从应用状态中取客户端与服务。

所有客户端都在应用启动时（``main.py`` 的 lifespan）一次性初始化并挂在
``app.state`` 上，请求处理阶段只做**读取**，避免每个请求重复建连。
"""
from __future__ import annotations

from fastapi import Request

from app.services.cache_service import CacheService
from app.services.chat_service import ChatService
from app.services.session_service import SessionService
from app.services.subject_service import SubjectService
from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import MySQLClient
from infrastructure.embedding.embedding_client import EmbeddingClient
from infrastructure.llm.qwen_client import QwenClient
from infrastructure.vector_db.milvus_client import MilvusClient


def get_db(request: Request) -> MySQLClient:
    return request.app.state.db


def get_cache(request: Request) -> RedisClient:
    return request.app.state.cache


def get_milvus(request: Request) -> MilvusClient:
    return request.app.state.milvus


def get_embedder(request: Request) -> EmbeddingClient:
    return request.app.state.embedder


def get_llm(request: Request) -> QwenClient:
    return request.app.state.llm


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_subject_service(request: Request) -> SubjectService:
    return request.app.state.subject_service


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_request_id(request: Request) -> str:
    """取本次请求的追踪 ID（由中间件写入，见 app/core/request_id.py）。"""
    return getattr(request.state, "request_id", "-")


def get_client_ip(request: Request) -> str | None:
    """获取客户端真实 IP。

    反向代理场景下 ``request.client.host`` 是代理的地址，需优先读
    ``X-Forwarded-For`` 的第一段（最左侧为原始客户端）。
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None
