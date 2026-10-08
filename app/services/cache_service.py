"""缓存服务：热点问答缓存与接口限流。

对应《教师智能问答系统技术文档》：
- 4.4 节：Redis 键设计（``hot_qa:{subject}:{sha1}``，TTL 1h）
- 8.2 节：热点问答缓存（命中则跳过检索与 LLM，零 API 调用）
- 9.4 节：基于 Redis 固定窗口的限流

在 :class:`RedisClient` 之上再包一层的原因：
客户端只管「读写 Redis」，而这里承载**业务语义**（缓存什么、缓存多久、
知识库更新后如何失效），便于在业务层单测与替换。
"""
from __future__ import annotations

from infrastructure.cache.redis_client import TTL_HOT_QA, RedisClient
from utils.logger import get_logger

logger = get_logger(__name__)

# 限流阈值（技术文档 9.4）
LIMIT_PER_IP = 30          # 单 IP：30 次/分钟
LIMIT_PER_SESSION = 20     # 单会话：20 次/分钟
WINDOW_SECONDS = 60


class CacheService:
    """热点问答缓存与限流。"""

    def __init__(self, cache: RedisClient) -> None:
        self.cache = cache

    # ------------------------------------------------------------------
    # 热点问答缓存
    # ------------------------------------------------------------------
    async def get_answer(self, question: str, subject_code: str | None) -> dict | None:
        """读取热点问答缓存。

        :return: ``{"answer": str, "references": [...]}``；未命中返回 None
        """
        return await self.cache.get_hot_answer(question, subject_code or "all")

    async def set_answer(
        self,
        question: str,
        answer: str,
        references: list[dict] | None = None,
        subject_code: str | None = None,
    ) -> bool:
        """写入热点问答缓存。

        ⚠️ 只在答案**完整生成后**调用。流式响应过程中不写缓存，
        否则会把半截答案缓存下来被后续请求复用。
        """
        return await self.cache.set_hot_answer(
            question=question,
            answer=answer,
            sources=[ref.get("summary", "")[:50] for ref in (references or [])],
            subject_code=subject_code or "all",
            ttl=TTL_HOT_QA,
        )

    async def clear_answers(self) -> int:
        """清空全部热点问答缓存（知识库重建后必须调用）。"""
        return await self.cache.clear_hot_qa()

    # ------------------------------------------------------------------
    # 限流（技术文档 9.4）
    # ------------------------------------------------------------------
    async def check_rate_limit(
        self,
        client_ip: str | None,
        session_id: str | None = None,
    ) -> bool:
        """固定窗口限流检查。

        :return: True 表示放行，False 表示超限
        缓存不可用时**放行**——限流是保护措施，不应因缓存故障误伤正常用户。
        """
        if client_ip:
            count = await self.cache.incr_rate(f"ip:{client_ip}", WINDOW_SECONDS)
            if count > LIMIT_PER_IP:
                logger.warning("触发 IP 限流：ip=%s, count=%s", client_ip, count)
                return False

        if session_id:
            count = await self.cache.incr_rate(f"session:{session_id}", WINDOW_SECONDS)
            if count > LIMIT_PER_SESSION:
                logger.warning("触发会话限流：session_id=%s, count=%s", session_id, count)
                return False

        return True

    async def health_check(self) -> str:
        """缓存健康检查透传。"""
        return await self.cache.health_check()
