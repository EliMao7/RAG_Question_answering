"""会话服务：创建、校验、活跃度维护。

对应《教师智能问答系统技术文档》：
- 6.3.2 节：``POST /api/session``
- 4.4 节：Redis 会话元数据缓存（``session:{id}`` Hash，TTL 24h）
- 5.3.1 节：会话创建时序

会话 ID 由**服务端**生成 UUIDv4：不可枚举，防止用户遍历他人会话（PRD 4.4.1）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import MySQLClient
from utils.exceptions import ErrorCode, SmartVoyageError
from utils.logger import get_logger

logger = get_logger(__name__)

# 进入页面时的欢迎语（PRD 4.2.3 引导机制）
WELCOME_MESSAGE = "你好！我是教师智能学习助手，有什么可以帮你？"


class SessionService:
    """会话生命周期管理。"""

    def __init__(self, db: MySQLClient, cache: RedisClient) -> None:
        self.db = db
        self.cache = cache

    # ------------------------------------------------------------------
    # 创建（接口 6.3.2）
    # ------------------------------------------------------------------
    async def create_session(
        self,
        subject_code: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict:
        """创建会话，同时写入 MySQL 与 Redis。"""
        session_id = str(uuid.uuid4())

        session = await self.db.create_session(
            session_id=session_id,
            subject_code=subject_code,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        # 缓存会话元数据，后续问答链路可直接命中，省一次 MySQL 查询
        await self.cache.save_session_meta(
            session_id,
            subject=subject_code or "all",
            created_at=session.get("created_at") or self._now(),
        )

        return {
            "session_id": session_id,
            "subject_code": subject_code,
            "created_at": session.get("created_at") or self._now(),
            "welcome_message": WELCOME_MESSAGE,
        }

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """校验 session_id 是否为合法 UUID。

        :raises SmartVoyageError: ``INVALID_SESSION_ID``（格式非法）
        """
        try:
            uuid.UUID(str(session_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise SmartVoyageError(
                ErrorCode.INVALID_SESSION_ID,
                "会话 ID 格式无效，请刷新页面重试",
                detail=f"session_id={session_id!r}",
            ) from exc
        return str(session_id)

    async def ensure_exists(self, session_id: str) -> dict:
        """确保会话存在。

        先查 Redis（快），未命中再查 MySQL（准）。Redis 不可用时直接落 MySQL，
        保证缓存故障不影响主流程（技术文档 3.9 降级矩阵）。

        :raises SmartVoyageError: ``SESSION_NOT_FOUND``
        """
        self.validate_session_id(session_id)

        meta = await self.cache.get_session_meta(session_id)
        if meta:
            return {"session_id": session_id, **meta}

        session = await self.db.get_session(session_id)
        if not session:
            raise SmartVoyageError(
                ErrorCode.SESSION_NOT_FOUND,
                "会话不存在或已过期，请刷新页面重试",
                detail=f"session_id={session_id}",
            )

        # 回填缓存（读时修复）
        await self.cache.save_session_meta(
            session_id,
            subject=session.get("subject_code") or "all",
            created_at=session.get("created_at") or self._now(),
        )
        return session

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    async def resolve_subject(
        self,
        session: dict,
        request_subject: str | None,
    ) -> str | None:
        """决定本次问答使用的学科。

        优先用请求显式指定的；未指定则沿用会话当前学科（用户切换过学科的场景）。
        """
        if request_subject:
            return request_subject
        stored = session.get("subject") or session.get("subject_code")
        if stored in (None, "", "all"):
            return None
        return stored

    async def refresh(self, session_id: str) -> None:
        """刷新会话活跃状态（缓存续期 + 库内计数），失败仅告警不阻断问答。"""
        try:
            await self.cache.save_session_meta(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话缓存续期失败：%s", exc)
        try:
            await self.db.bump_message_count(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话轮次累加失败：%s", exc)

    async def update_subject(self, session_id: str, subject_code: str | None) -> None:
        """更新会话学科（前端切换学科时调用），保证下次提问沿用该学科。"""
        try:
            await self.db.update_session_subject(session_id, subject_code)
            await self.cache.save_session_meta(session_id, subject=subject_code or "all")
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话学科更新失败：%s", exc)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
