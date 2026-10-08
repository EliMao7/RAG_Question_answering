"""学科服务：白名单校验、学科列表与知识块统计。

对应《教师智能问答系统技术文档》：
- 5.5 节：学科过滤的实现细节
- 6.3.3 节：``GET /api/subjects``
- 4.2.1 节：``subject_code`` 与 config.ini 的 ``valid_sources`` 必须严格一致
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from config.settings import Config, get_config
from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import MySQLClient
from infrastructure.vector_db.milvus_client import MilvusClient
from utils.exceptions import ErrorCode, SmartVoyageError
from utils.logger import get_logger

logger = get_logger(__name__)

# 学科列表的缓存键与 TTL（技术文档 2.2.3 / 4.4）
SUBJECT_LIST_CACHE_KEY = "subject:list"
SUBJECT_LIST_TTL = 5 * 60


class SubjectService:
    """学科相关业务逻辑。"""

    def __init__(
        self,
        db: MySQLClient,
        cache: RedisClient,
        milvus: MilvusClient,
        config: Config | None = None,
    ) -> None:
        self.db = db
        self.cache = cache
        self.milvus = milvus
        cfg = config or get_config()
        # 启动时即加载白名单，用于所有入口的学科校验
        self.valid_sources: list[str] = cfg.get_json("app", "valid_sources") or []
        self._name_map: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self, subject_code: str | None) -> str | None:
        """校验学科代码。

        :return: 归一化后的学科代码；``None`` / ``all`` 统一返回 None（表示全学科）
        :raises SmartVoyageError: 学科不在白名单内（``INVALID_SUBJECT``）
        """
        if subject_code is None:
            return None
        code = subject_code.strip()
        if code in ("", "all"):
            return None
        if code not in self.valid_sources:
            raise SmartVoyageError(
                ErrorCode.INVALID_SUBJECT,
                f"不支持的学科：{code}",
                detail=f"valid_sources={self.valid_sources}",
            )
        return code

    # ------------------------------------------------------------------
    # 学科名称映射
    # ------------------------------------------------------------------
    async def get_name_map(self) -> dict[str, str]:
        """获取 ``{学科代码: 学科名称}`` 映射（进程内缓存，避免每次查库）。"""
        if self._name_map is None:
            try:
                self._name_map = await self.db.get_subject_map()
            except Exception as exc:  # noqa: BLE001 - 查库失败不应阻断问答
                logger.warning("学科名称查询失败，回退为代码本身：%s", exc)
                self._name_map = {}
        return self._name_map

    async def get_name(self, subject_code: str | None) -> str:
        """把学科代码翻译为中文名；未知或全学科时返回「全学科」。"""
        if not subject_code:
            return "全学科"
        name_map = await self.get_name_map()
        return name_map.get(subject_code, subject_code)

    # ------------------------------------------------------------------
    # 学科列表（接口 6.3.3）
    # ------------------------------------------------------------------
    async def list_subjects(self, use_cache: bool = True) -> dict:
        """返回学科列表，含各学科知识块数量。

        结构与技术文档 6.3.3 一致：第一个元素是全学科（``code=null``）。
        ``doc_count`` 来自 Milvus 按 ``subject_code`` 的分组统计，缓存 5 分钟。
        """
        if use_cache:
            cached = await self._get_cached_list()
            if cached:
                return cached

        subjects = await self.db.get_subjects()

        # 首项为「全学科」，满足前端默认选中逻辑
        items: list[dict] = [
            {"code": None, "name": "全学科", "is_default": True}
        ]

        # 知识块数量：Milvus 统计较慢，失败时降级为不带 doc_count
        try:
            distribution = await asyncio.to_thread(self.milvus.get_subject_distribution)
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识块数量统计失败：%s", exc)
            distribution = {}

        for row in subjects:
            items.append({
                "code": row["code"],
                "name": row["name"],
                "is_default": False,
                "doc_count": distribution.get(row["code"], 0),
            })

        # 全学科的 doc_count 为各学科之和（去重前的子块总数）
        items[0]["doc_count"] = sum(distribution.values())

        payload = {
            "subjects": items,
            "cached_at": self._now(),
        }
        await self._set_cached_list(payload)
        return payload

    async def _get_cached_list(self) -> dict | None:
        """读学科列表缓存。"""
        try:
            raw = await self.cache.get(SUBJECT_LIST_CACHE_KEY)
            if raw:
                import json

                return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("学科列表缓存读取失败：%s", exc)
        return None

    async def _set_cached_list(self, payload: dict) -> None:
        """写学科列表缓存。"""
        import json

        try:
            await self.cache.set(
                SUBJECT_LIST_CACHE_KEY,
                json.dumps(payload, ensure_ascii=False),
                SUBJECT_LIST_TTL,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("学科列表缓存写入失败：%s", exc)

    async def invalidate_cache(self) -> None:
        """清除学科列表缓存（管理员改动学科后调用）。"""
        await self.cache.delete(SUBJECT_LIST_CACHE_KEY)

    @staticmethod
    def _now() -> str:
        """当前时间（东八区，ISO 8601）。"""
        return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
