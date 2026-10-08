"""Redis 缓存客户端。

对应技术设计文档：
- 2.2.3 节：缓存键设计（session / hot_qa / subject stats）
- 5.1.2 节：热点问答缓存（TTL 1 小时）
- 5.3.2 节：连接池 ``max_connections=50``
- 7.2 节：健康检查

键设计（与技术文档 2.2.3 保持一致）：
============================  ==========  ========  ==============================
键                            类型        TTL       用途
============================  ==========  ========  ==============================
``session:{session_id}``      Hash        24h       会话元数据（subject 等）
``hot_qa:{question_hash}``    String      1h        热点问答缓存（JSON）
``subject:{code}:stats``      Hash        5min      学科统计信息
``rate:{client_ip}:{窗口}``   String      见窗口     限流计数（技术文档 8.3）
============================  ==========  ========  ==============================

实现说明：
- 使用 ``redis.asyncio``（redis-py 内置的异步实现），与 FastAPI async 链路一致；
- 缓存键中的 ``question_hash`` 采用 **sha1 前 16 位**，避免问题文本过长导致 key 超限，
  同时规避把用户原始问题明文写进 Redis key（隐私考虑）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from config.settings import Config, get_config
from utils.exceptions import CacheError
from utils.logger import get_logger

logger = get_logger(__name__)

# TTL 常量（秒），与技术文档 2.2.3 的约定一致
TTL_SESSION = 24 * 60 * 60      # 会话元数据：24 小时
TTL_HOT_QA = 60 * 60            # 热点问答：1 小时
TTL_SUBJECT_STATS = 5 * 60      # 学科统计：5 分钟


class RedisClient:
    """Redis 异步客户端，封装缓存读写与会话元数据。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or get_config()
        self.host = cfg.get("redis", "host")
        self.port = cfg.get_int("redis", "port", 6379)
        self.password = cfg.get("redis", "password") or None
        self.db = cfg.get_int("redis", "db", 0)
        self.max_connections = cfg.get_int("redis", "max_connections", 50)

        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """创建连接池并验证连通性（幂等）。"""
        if self._client is not None:
            return
        try:
            self._pool = aioredis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=True,   # 直接返回 str，免去手动 decode
                socket_timeout=1.0,      # 缓存故障时应快速失败，不能拖垮主链路
                socket_connect_timeout=1.0,
                # 显式使用 RESP2 协议（redis-py 8 默认 RESP3）。
                # RESP3 握手会先发 HELLO 命令，而该命令是 Redis 6.0 才引入的；
                # 连 Redis 5 及更早版本（含 Windows 移植版 3.0.504）会直接报
                # "unknown command 'HELLO'"。指定 protocol=2 可兼容新旧服务端。
                protocol=2,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            await self._client.ping()
            logger.info("Redis 已连接：%s:%s (db=%s)", self.host, self.port, self.db)
        except Exception as exc:  # noqa: BLE001
            raise CacheError("缓存服务连接失败", detail=self._diagnose(exc)) from exc

    def _diagnose(self, exc: Exception) -> str:
        """把常见连接错误翻译成可执行的排查建议。"""
        text = str(exc)
        if "no password is set" in text:
            return (
                f"{text}\n  → 服务端未设置密码，但 config.ini 的 [redis].password 填了值。"
                f"请把该项留空（password = ），或给 Redis 设置密码：CONFIG SET requirepass <pwd>"
            )
        if "WRONGPASS" in text or "invalid password" in text:
            return (
                f"{text}\n  → 密码错误，请核对 config.ini 的 [redis].password"
            )
        if "HELLO" in text:
            return (
                f"{text}\n  → 服务端版本过旧（RESP3 不被支持）。"
                f"本客户端已强制 RESP2；若仍报错请升级 Redis 至 6.0+"
            )
        return text

    async def close(self) -> None:
        """关闭客户端与连接池。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis 连接已关闭")

    async def __aenter__(self) -> "RedisClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    @property
    def client(self) -> aioredis.Redis:
        """底层 redis 客户端，供需要执行高级命令时使用。"""
        if self._client is None:
            raise CacheError("Redis 尚未连接，请先调用 connect()")
        return self._client

    # ------------------------------------------------------------------
    # 通用缓存读写
    # ------------------------------------------------------------------
    async def get(self, key: str) -> str | None:
        """读取字符串值。"""
        return await self.client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """写入字符串值；``ttl`` 为过期秒数，None 表示不过期。"""
        try:
            if ttl:
                return bool(await self.client.setex(key, ttl, value))
            return bool(await self.client.set(key, value))
        except Exception as exc:  # noqa: BLE001 - 缓存写入失败不应中断业务
            logger.warning("Redis 写入失败：key=%s, err=%s", key, exc)
            return False

    async def delete(self, *keys: str) -> int:
        """删除若干键，返回实际删除数量。"""
        if not keys:
            return 0
        return int(await self.client.delete(*keys))

    async def exists(self, key: str) -> bool:
        """判断键是否存在。"""
        return bool(await self.client.exists(key))

    # ------------------------------------------------------------------
    # 会话元数据：session:{session_id}（Hash）
    # ------------------------------------------------------------------
    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}"

    async def _hset_many(self, key: str, mapping: dict[str, str], ttl: int) -> bool:
        """逐字段写入 Hash 并设置过期时间（单个 pipeline = 一次往返）。

        ⚠️ 为什么不直接用 ``hset(key, mapping=...)``：
        该写法会生成 ``HSET key f1 v1 f2 v2 ...`` 这种多字段命令，
        而**多字段 HSET 是 Redis 4.0 才引入的**。连 Redis 3.x（例如 Windows
        移植版 3.0.504）会报 ``wrong number of arguments for 'hset' command``。
        拆成逐字段写入后，Redis 3.x / 7.x 均可正常工作。
        """
        try:
            async with self.client.pipeline(transaction=False) as pipe:
                for field, value in mapping.items():
                    pipe.hset(key, field, value)
                pipe.expire(key, ttl)
                await pipe.execute()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hash 写入失败：key=%s, err=%s", key, exc)
            return False

    async def save_session_meta(self, session_id: str, **fields: Any) -> bool:
        """缓存会话元数据（Hash 结构）。

        :param fields: 例如 ``subject='ai'``、``created_at='2026-07-06T10:30:00'``
        """
        # 值统一转成字符串：Hash 只接受 str/int/float
        mapping = {k: ("" if v is None else str(v)) for k, v in fields.items()}
        if not mapping:
            return False
        return await self._hset_many(self._session_key(session_id), mapping, TTL_SESSION)

    async def get_session_meta(self, session_id: str) -> dict[str, str] | None:
        """读取会话元数据，缓存未命中返回 None。"""
        key = self._session_key(session_id)
        try:
            meta = await self.client.hgetall(key)
            if not meta:
                return None
            await self.client.expire(key, TTL_SESSION)  # 命中即续期（滑动过期）
            return meta
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话元数据读取失败：session_id=%s, err=%s", session_id, exc)
            return None

    async def delete_session_meta(self, session_id: str) -> int:
        """删除会话元数据（清空历史时同步调用，保证缓存与库一致）。"""
        return await self.delete(self._session_key(session_id))

    # ------------------------------------------------------------------
    # 热点问答缓存：hot_qa:{question_hash}（String / JSON）
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_question(question: str, subject_code: str = "all") -> str:
        """生成问题指纹。

        把学科一并纳入 hash：同一个问题在不同学科下检索范围不同、
        答案可能不同，若不区分会导致跨学科串答。
        """
        raw = f"{subject_code}::{question.strip()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _hot_qa_key(self, question: str, subject_code: str = "all") -> str:
        return f"hot_qa:{self._hash_question(question, subject_code)}"

    async def get_hot_answer(self, question: str, subject_code: str = "all") -> dict | None:
        """读取热点问答缓存（技术文档 5.1.2）。

        :return: ``{"answer": str, "sources": [...]}``，未命中返回 None
        """
        key = self._hot_qa_key(question, subject_code)
        try:
            cached = await self.client.get(key)
            if not cached:
                return None
            logger.info("热点问答缓存命中：subject=%s", subject_code)
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError) as exc:
            # 缓存内容损坏时直接删除，避免反复命中坏数据
            logger.warning("热点问答缓存内容损坏，已清除：key=%s, err=%s", key, exc)
            await self.delete(key)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("热点问答缓存读取失败：err=%s", exc)
            return None

    async def set_hot_answer(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        subject_code: str = "all",
        ttl: int = TTL_HOT_QA,
    ) -> bool:
        """写入热点问答缓存。

        注意：只在答案**完整生成后**调用。流式输出过程中不要写缓存，
        否则会把半截答案缓存下来复用。
        """
        payload = json.dumps(
            {"answer": answer, "sources": sources or []}, ensure_ascii=False
        )
        return await self.set(self._hot_qa_key(question, subject_code), payload, ttl)

    async def clear_hot_qa(self) -> int:
        """清空全部热点问答缓存。

        知识库重建后必须调用：否则旧答案会继续从缓存返回，
        用户看不到新知识库的内容（缓存与知识库不一致）。
        使用 SCAN 而非 KEYS，避免阻塞 Redis。
        """
        deleted = 0
        try:
            async for key in self.client.scan_iter(match="hot_qa:*", count=200):
                deleted += await self.delete(key)
            logger.info("热点问答缓存已清空，删除 %s 个键", deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("热点问答缓存清理失败：%s", exc)
        return deleted

    # ------------------------------------------------------------------
    # 学科统计：subject:{code}:stats（Hash）
    # ------------------------------------------------------------------
    async def save_subject_stats(self, subject_code: str, **stats: Any) -> bool:
        """缓存学科统计信息（技术文档 2.2.3，TTL 5 分钟）。"""
        mapping = {k: str(v) for k, v in stats.items()}
        if not mapping:
            return False
        return await self._hset_many(
            f"subject:{subject_code}:stats", mapping, TTL_SUBJECT_STATS
        )

    async def get_subject_stats(self, subject_code: str) -> dict[str, str] | None:
        """读取学科统计信息。"""
        try:
            stats = await self.client.hgetall(f"subject:{subject_code}:stats")
            return stats or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("学科统计读取失败：subject=%s, err=%s", subject_code, exc)
            return None

    # ------------------------------------------------------------------
    # 限流计数（技术文档 8.3）
    # ------------------------------------------------------------------
    async def incr_rate(self, key: str, window_seconds: int) -> int:
        """在固定时间窗内自增计数并返回当前值。

        首次自增时设置过期时间，形成「固定窗口限流」。
        调用方用返回值与阈值比较即可判断是否放行。
        """
        try:
            full_key = f"rate:{key}:{window_seconds}"
            current = await self.client.incr(full_key)
            if current == 1:
                await self.client.expire(full_key, window_seconds)
            return int(current)
        except Exception as exc:  # noqa: BLE001 - 限流失败时放行，避免误伤正常用户
            logger.warning("限流计数失败：key=%s, err=%s", key, exc)
            return 0

    # ------------------------------------------------------------------
    # 健康检查（技术文档 4.1.5 / 7.2）
    # ------------------------------------------------------------------
    async def health_check(self) -> str:
        """探测 Redis 可用性，返回 ``connected`` / ``disconnected``。"""
        try:
            await self.client.ping()
            return "connected"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 健康检查失败：%s", exc)
            return "disconnected"
