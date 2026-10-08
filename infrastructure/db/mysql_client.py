"""MySQL 数据库客户端。

对应《教师智能问答系统技术文档》：
- 4.2 节：库 ``subjects_kg``，三张表 ``subjects`` / ``chat_session`` / ``chat_record``
- 8.4 节：连接池（``pool_size=20, max_overflow=10``，1 小时回收，取连接前探活）
- 6.3.2 / 6.3.5 / 6.3.6 节：会话创建、历史查询（**游标分页**）、历史清空

实现说明：
- 使用 **aiomysql** 异步驱动，与 FastAPI 的 async 链路保持一致；
- 连接池进程内唯一，服务启动时 :meth:`connect` 一次，关闭时 :meth:`close`；
- 所有 SQL 均使用参数化占位符 ``%s``，杜绝 SQL 拼接注入。

⚠️ 关于 aiomysql 的参数命名：它用的是 ``minsize`` / ``maxsize``，
不是 SQLAlchemy 风格的 ``pool_size`` / ``max_overflow``（技术文档 8.4 的示例
沿用了后者），照抄会报 ``unexpected keyword argument 'pool_size'``。
本实现做了等价映射：``maxsize = pool_size + max_overflow``。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence

import aiomysql

from config.settings import Config, get_config
from utils.exceptions import DatabaseError
from utils.logger import get_logger

logger = get_logger(__name__)

# 历史分页的默认与上限（技术文档 6.1：limit 默认 20，最大 100）
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100


class MySQLClient:
    """MySQL 异步客户端，封装连接池与业务查询。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or get_config()
        self.host = cfg.get("mysql", "host")
        self.port = cfg.get_int("mysql", "port", 3306)
        self.user = cfg.get("mysql", "user")
        self.password = cfg.get("mysql", "password")
        self.database = cfg.get("mysql", "database")

        # 连接池参数（技术文档 8.4）
        self.pool_size = cfg.get_int("mysql", "pool_size", 20)
        self.max_overflow = cfg.get_int("mysql", "max_overflow", 10)

        self._pool: aiomysql.Pool | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """创建连接池（幂等，重复调用直接返回）。"""
        if self._pool is not None:
            return
        try:
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                charset="utf8mb4",       # 完整支持中文与 Emoji
                autocommit=True,         # 本项目以单条写入为主，免去手动 commit
                minsize=self.pool_size,                      # 常驻连接数
                maxsize=self.pool_size + self.max_overflow,  # 峰值连接上限
                pool_recycle=3600,       # 1 小时回收，避免 MySQL 主动断连后拿到失效连接
                connect_timeout=5,
            )
            logger.info(
                "MySQL 连接池已创建：%s:%s/%s (minsize=%s, maxsize=%s)",
                self.host, self.port, self.database,
                self.pool_size, self.pool_size + self.max_overflow,
            )
        except Exception as exc:  # noqa: BLE001 - 统一包装为业务异常
            raise DatabaseError(
                "数据库连接失败",
                detail=f"{exc}（请确认 MySQL 已启动、config.ini 的 [mysql] 账号密码正确，"
                       f"且已执行 scripts/init_db.sql）",
            ) from exc

    async def close(self) -> None:
        """关闭连接池并释放所有连接，服务退出时调用。"""
        if self._pool is None:
            return
        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None
        logger.info("MySQL 连接池已关闭")

    async def __aenter__(self) -> "MySQLClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # 通用查询方法
    # ------------------------------------------------------------------
    async def fetch_one(self, sql: str, args: Sequence[Any] | None = None) -> dict | None:
        """查询单行，返回 dict（无结果返回 None）。"""
        rows = await self.fetch_all(sql, args, limit=1)
        return rows[0] if rows else None

    async def fetch_all(
        self,
        sql: str,
        args: Sequence[Any] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """查询多行，返回 dict 列表。

        :param limit: 仅取前 N 行（用于「只关心是否存在 / 取第一条」的场景）
        """
        if self._pool is None:
            await self.connect()
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                # aiomysql 需要显式指定 DictCursor 才会返回字典
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(sql, args or ())
                    rows = await cur.fetchmany(limit) if limit else await cur.fetchall()
                    return list(rows)
        except Exception as exc:  # noqa: BLE001
            logger.error("MySQL 查询失败：sql=%s | args=%s | err=%s", sql, args, exc)
            raise DatabaseError("数据库查询失败", detail=str(exc)) from exc

    async def execute(self, sql: str, args: Sequence[Any] | None = None) -> int:
        """执行写操作，返回受影响行数。"""
        if self._pool is None:
            await self.connect()
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                async with conn.cursor() as cur:
                    await cur.execute(sql, args or ())
                    return cur.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error("MySQL 写入失败：sql=%s | args=%s | err=%s", sql, args, exc)
            raise DatabaseError("数据库写入失败", detail=str(exc)) from exc

    async def execute_returning_id(self, sql: str, args: Sequence[Any] | None = None) -> int:
        """执行 INSERT 并返回自增主键。

        与 :meth:`execute` 的区别：本方法在**同一个连接**内完成写入与取 ID。
        连接池的 ``acquire()`` 每次可能返回不同连接，而 MySQL 的
        ``LAST_INSERT_ID()`` 是连接级状态，跨连接查询会拿到 0。
        """
        if self._pool is None:
            await self.connect()
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                async with conn.cursor() as cur:
                    await cur.execute(sql, args or ())
                    return int(cur.lastrowid or 0)
        except Exception as exc:  # noqa: BLE001
            logger.error("MySQL 写入失败：sql=%s | args=%s | err=%s", sql, args, exc)
            raise DatabaseError("数据库写入失败", detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # 会话表 chat_session（技术文档 4.2.2 / 接口 6.3.2）
    # ------------------------------------------------------------------
    async def create_session(
        self,
        session_id: str,
        subject_code: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict:
        """创建会话，返回会话信息。

        ``session_id`` 由服务端生成 UUIDv4 后传入（不可枚举，满足 PRD 4.4.1）。
        """
        sql = """
            INSERT INTO chat_session (session_id, subject_code, client_ip, user_agent)
            VALUES (%s, %s, %s, %s)
        """
        await self.execute(
            sql,
            (session_id, subject_code,
             (client_ip or "")[:45] or None,       # 字段长度 45，超长直接截断避免报错
             (user_agent or "")[:255] or None),    # 字段长度 255
        )
        logger.info("会话已创建：session_id=%s, subject=%s", session_id, subject_code or "全学科")
        return await self.get_session(session_id) or {}

    async def get_session(self, session_id: str) -> dict | None:
        """按 session_id 查询会话（排除已软删除的会话）。"""
        sql = """
            SELECT session_id, subject_code, title, message_count,
                   client_ip, user_agent, created_at, last_active_at
            FROM chat_session
            WHERE session_id = %s AND is_deleted = 0
        """
        row = await self.fetch_one(sql, (session_id,))
        if row:
            row["created_at"] = self._format_dt(row.get("created_at"))
            row["last_active_at"] = self._format_dt(row.get("last_active_at"))
        return row

    async def session_exists(self, session_id: str) -> bool:
        """判断会话是否存在且未被删除（用于接口的 404 校验）。"""
        row = await self.fetch_one(
            "SELECT 1 AS ok FROM chat_session WHERE session_id = %s AND is_deleted = 0",
            (session_id,),
        )
        return row is not None

    async def update_session_subject(self, session_id: str, subject_code: str | None) -> int:
        """更新会话当前学科（前端切换学科时调用）。"""
        sql = "UPDATE chat_session SET subject_code = %s WHERE session_id = %s"
        return await self.execute(sql, (subject_code, session_id))

    async def touch_session(self, session_id: str) -> int:
        """刷新会话活跃时间。

        虽然 ``last_active_at`` 带 ON UPDATE CURRENT_TIMESTAMP，但该行为只在
        行内其它字段真正变化时触发；仅刷新活跃时间需要显式赋值。
        """
        return await self.execute(
            "UPDATE chat_session SET last_active_at = NOW() WHERE session_id = %s",
            (session_id,),
        )

    async def bump_message_count(self, session_id: str, delta: int = 1) -> int:
        """累加会话问答轮次。"""
        sql = """
            UPDATE chat_session
            SET message_count = message_count + %s, last_active_at = NOW()
            WHERE session_id = %s
        """
        return await self.execute(sql, (delta, session_id))

    async def set_session_title_if_empty(self, session_id: str, title: str) -> int:
        """会话首次提问时写入标题（取问题前 30 字），已有标题则不覆盖。"""
        sql = """
            UPDATE chat_session
            SET title = %s
            WHERE session_id = %s AND (title IS NULL OR title = '')
        """
        return await self.execute(sql, (title[:100], session_id))

    # ------------------------------------------------------------------
    # 问答记录表 chat_record（技术文档 4.2.3 / 接口 6.3.4）
    # ------------------------------------------------------------------
    async def save_chat_record(
        self,
        session_id: str,
        user_question: str,
        answer: str,
        subject_code: str | None = None,
        references: list[dict] | None = None,
        retrieved_count: int = 0,
        is_greeting: bool = False,
        is_from_cache: bool = False,
        model_name: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_time_ms: int = 0,
        finish_reason: str = "stop",
        error_code: str | None = None,
    ) -> int:
        """保存一条问答记录，返回自增主键 ``record_id``。

        ``references`` 为溯源信息（父块摘要）列表，以 JSON 存入 ``references_json``，
        用于前端展示与「溯源命中率」统计（技术文档 4.2.3）。
        """
        sql = """
            INSERT INTO chat_record
                (session_id, subject_code, user_question, answer, references_json,
                 retrieved_count, is_greeting, is_from_cache, model_name,
                 prompt_tokens, completion_tokens, response_time_ms,
                 finish_reason, error_code)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        args = (
            session_id,
            subject_code,
            user_question,
            answer,
            json.dumps(references or [], ensure_ascii=False),
            retrieved_count,
            int(is_greeting),
            int(is_from_cache),
            model_name,
            prompt_tokens,
            completion_tokens,
            response_time_ms,
            finish_reason,
            error_code,
        )
        # 必须用同一连接取回自增 ID，跨连接查 LAST_INSERT_ID() 会得到 0
        record_id = await self.execute_returning_id(sql, args)
        logger.info(
            "问答记录已保存：record_id=%s, session_id=%s, subject=%s, "
            "答案 %s 字, 耗时 %sms, finish=%s",
            record_id, session_id, subject_code or "全学科",
            len(answer), response_time_ms, finish_reason,
        )
        return record_id

    async def get_history(
        self,
        session_id: str,
        cursor: int | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> dict:
        """游标分页查询会话历史（技术文档 4.2.3 / 接口 6.3.5）。

        为什么用游标而不是 OFFSET：大 OFFSET 需要扫描并丢弃前 N 行，
        数据量增长后性能急剧劣化。游标分页走 ``idx_session_time`` 索引范围扫描，
        10 万级记录仍保持毫秒级（PRD 要求 < 500ms）。

        :param cursor: 上一页最后一条的 ``record_id``；传 None 表示取第一页
        :param limit: 每页条数，默认 20，最大 100
        :return: ``{"records": [...], "next_cursor": int|None, "has_more": bool, "total": int}``
        """
        limit = max(1, min(limit, MAX_PAGE_LIMIT))

        # 多取 1 条用于判断是否还有下一页，避免额外的 count 查询
        sql = """
            SELECT record_id, user_question, answer, subject_code, references_json,
                   retrieved_count, is_from_cache, response_time_ms, created_at
            FROM chat_record
            WHERE session_id = %s
        """
        args: list[Any] = [session_id]
        if cursor is not None:
            sql += " AND record_id < %s"
            args.append(cursor)
        sql += " ORDER BY record_id DESC LIMIT %s"
        args.append(limit + 1)

        rows = await self.fetch_all(sql, args)
        has_more = len(rows) > limit
        rows = rows[:limit]

        records = [
            {
                "record_id": row["record_id"],
                "question": row["user_question"],
                "answer": row["answer"],
                "subject_code": row["subject_code"],
                "references": self._loads_json(row.get("references_json")),
                "retrieved_count": row.get("retrieved_count", 0),
                "is_from_cache": bool(row.get("is_from_cache")),
                "response_time_ms": row.get("response_time_ms", 0),
                "created_at": self._format_dt(row.get("created_at")),
            }
            for row in rows
        ]

        total_row = await self.fetch_one(
            "SELECT COUNT(*) AS total FROM chat_record WHERE session_id = %s",
            (session_id,),
        )

        return {
            "records": records,
            "next_cursor": records[-1]["record_id"] if (records and has_more) else None,
            "has_more": has_more,
            "total": int(total_row["total"]) if total_row else 0,
        }

    async def clear_history(self, session_id: str) -> int:
        """清空会话历史，返回删除条数（接口 6.3.6）。

        ⚠️ 实现说明（与技术文档 4.2.2 的一处偏差，已确认后取舍）：
        文档备注「清空历史使用 is_deleted=1 而非物理删除」，但 ``is_deleted``
        只存在于 **chat_session** 表，chat_record 表没有删除标记。若仅把会话置为
        已删除，用户继续追问时又需要把 is_deleted 翻回 0——旧记录会**重新出现**，
        语义上无法表达「清空后继续对话」。

        因此这里采用**物理删除** chat_record 行，并重置 message_count；
        chat_session 的 ``is_deleted`` 保留其原本用途（整段会话被放弃/软删除，
        历史查询会自动过滤）。该实现满足接口契约（返回 cleared_count、幂等）。
        """
        # 先取待删除条数：DELETE 的 rowcount 在重复调用时为 0，但接口需要
        # 「本次清空了多少条」，首次调用才有意义，这里以删除前计数为准
        count_row = await self.fetch_one(
            "SELECT COUNT(*) AS c FROM chat_record WHERE session_id = %s", (session_id,)
        )
        count = int(count_row["c"]) if count_row else 0

        await self.execute("DELETE FROM chat_record WHERE session_id = %s", (session_id,))
        # 同步重置会话轮次与标题，使会话回到「刚创建」的状态
        await self.execute(
            "UPDATE chat_session SET message_count = 0, title = NULL, "
            "last_active_at = NOW() WHERE session_id = %s",
            (session_id,),
        )
        logger.info("会话历史已清空：session_id=%s, 删除 %s 条", session_id, count)
        return count

    # ------------------------------------------------------------------
    # 学科表 subjects（技术文档 4.2.1 / 接口 6.3.3）
    # ------------------------------------------------------------------
    async def get_subjects(self, only_active: bool = True) -> list[dict]:
        """查询学科列表，按 sort_order 排序。"""
        sql = """
            SELECT subject_code AS code, subject_name AS name, description
            FROM subjects
        """
        if only_active:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY sort_order ASC, id ASC"
        return await self.fetch_all(sql)

    async def get_subject_map(self) -> dict[str, str]:
        """返回 ``{学科代码: 学科名称}`` 映射，用于把代码翻译成中文名。"""
        rows = await self.get_subjects()
        return {row["code"]: row["name"] for row in rows}

    # ------------------------------------------------------------------
    # 健康检查（接口 6.3.1）
    # ------------------------------------------------------------------
    async def health_check(self) -> str:
        """探测 MySQL 可用性，返回 ``connected`` / ``disconnected``。"""
        try:
            await self.fetch_one("SELECT 1 AS ok")
            return "connected"
        except Exception as exc:  # noqa: BLE001 - 健康检查不应抛异常
            logger.warning("MySQL 健康检查失败：%s", exc)
            return "disconnected"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _loads_json(raw: Any) -> Any:
        """安全反序列化 JSON 字段（兼容驱动已自动解析为 list 的情况）。"""
        if raw is None:
            return []
        if isinstance(raw, (list, dict)):
            return raw
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return []

    @staticmethod
    def _format_dt(value: Any) -> Any:
        """把 datetime 统一格式化为 ISO 字符串（技术文档 6.1：时间用 ISO 8601）。"""
        return value.isoformat() if isinstance(value, datetime) else value
