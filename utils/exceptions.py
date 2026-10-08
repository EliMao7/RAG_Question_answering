"""统一异常与错误码。

对应技术设计文档第 4.2 节「错误码定义」。

设计约定：
- ``ErrorCode`` 是 HTTP 状态码 + 业务语义的组合，便于 API 层统一转换为响应；
- 所有自定义异常都继承 :class:`SmartVoyageError`，API 层只需捕获这一个基类；
- ``message`` 面向用户（可直接展示），日志中则记录 ``detail`` 便于排查。
"""
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """错误码枚举（值为 ``HTTP状态码:业务码`` 形式，便于排查与国际化扩展）。"""

    SUCCESS = "200:SUCCESS"

    # ---- 客户端错误 ----
    INVALID_PARAMS = "400:INVALID_PARAMS"              # 参数错误
    INVALID_QUESTION_LENGTH = "400:INVALID_QUESTION_LENGTH"  # 问题长度不在 1~2000
    INVALID_SUBJECT = "400:INVALID_SUBJECT"            # 学科不在白名单
    INVALID_SESSION_ID = "400:INVALID_SESSION_ID"      # 会话 ID 非 UUID 格式
    UNAUTHORIZED = "401:UNAUTHORIZED"                  # 未授权 / 会话无效
    SESSION_NOT_FOUND = "404:SESSION_NOT_FOUND"        # 会话不存在
    RATE_LIMITED = "429:RATE_LIMITED"                  # 触发本地限流
    LLM_RATE_LIMITED = "429:LLM_RATE_LIMITED"          # 触发上游（千问）限流

    # ---- 服务端错误 ----
    INTERNAL_ERROR = "500:INTERNAL_ERROR"              # 未分类异常
    DB_ERROR = "500:DB_ERROR"                          # MySQL 异常
    CACHE_ERROR = "500:CACHE_ERROR"                    # Redis 异常
    VECTOR_DB_ERROR = "500:VECTOR_DB_ERROR"            # Milvus 异常
    EMBEDDING_ERROR = "500:EMBEDDING_ERROR"            # 向量化失败

    # ---- 非错误但需要特殊处理 ----
    # 检索无结果：不调用 LLM 编造，直接返回兜底话术（技术文档 6.5）
    EMPTY_RETRIEVAL = "200:EMPTY_RETRIEVAL"

    # ---- 依赖不可用（可重试）----
    SERVICE_UNAVAILABLE = "503:SERVICE_UNAVAILABLE"    # 依赖服务不可用
    LLM_UNAVAILABLE = "503:LLM_UNAVAILABLE"            # 千问接口不可用
    VECTOR_DB_UNAVAILABLE = "503:VECTOR_DB_UNAVAILABLE"  # Milvus 不可用
    EMBEDDING_UNAVAILABLE = "503:EMBEDDING_UNAVAILABLE"  # 向量化服务不可用

    @property
    def http_status(self) -> int:
        """提取 HTTP 状态码部分。"""
        return int(self.value.split(":", 1)[0])

    @property
    def code(self) -> str:
        """提取业务码部分。"""
        return self.value.split(":", 1)[1]


class SmartVoyageError(Exception):
    """项目内所有业务异常的基类。

    :param error_code: 错误码
    :param message: 面向用户的提示（可直接展示给前端）
    :param detail: 面向开发者的细节（只进日志，不外传）
    """

    def __init__(
        self,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        message: str = "系统异常，请稍后重试",
        detail: str | None = None,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.detail = detail
        super().__init__(f"[{error_code.code}] {message}" + (f" | {detail}" if detail else ""))

    def to_dict(self, request_id: str | None = None, timestamp: str | None = None) -> dict:
        """转换为技术文档 6.1 定义的统一错误响应体。

        :return: ``{"code": ..., "message": ..., "request_id": ..., "timestamp": ...}``
                 （放在响应的 ``error`` 字段下，由 API 层的异常处理器包装）
        """
        from datetime import datetime, timezone, timedelta

        payload = {
            "code": self.error_code.code,
            "message": self.message,
            "request_id": request_id or "-",
            "timestamp": timestamp or datetime.now(
                timezone(timedelta(hours=8))   # 统一使用东八区（北京时间）
            ).isoformat(timespec="seconds"),
        }
        return payload


# ---------------------------------------------------------------------------
# 按依赖维度细分的异常：便于上层针对不同故障做差异化降级
# （例如 Milvus 挂了可以跳过检索用通用知识回答，MySQL 挂了则必须报错）
# ---------------------------------------------------------------------------
class DatabaseError(SmartVoyageError):
    """MySQL 操作异常。"""

    def __init__(self, message: str = "数据库异常，请稍后重试", detail: str | None = None) -> None:
        super().__init__(ErrorCode.DB_ERROR, message, detail)


class CacheError(SmartVoyageError):
    """Redis 操作异常。"""

    def __init__(self, message: str = "缓存服务异常", detail: str | None = None) -> None:
        super().__init__(ErrorCode.CACHE_ERROR, message, detail)


class VectorDBError(SmartVoyageError):
    """Milvus 操作异常。"""

    def __init__(
        self,
        message: str = "知识库检索异常",
        detail: str | None = None,
        unavailable: bool = False,
    ) -> None:
        code = ErrorCode.VECTOR_DB_UNAVAILABLE if unavailable else ErrorCode.VECTOR_DB_ERROR
        super().__init__(code, message, detail)


class EmbeddingError(SmartVoyageError):
    """向量化调用异常。"""

    def __init__(self, message: str = "文本向量化失败", detail: str | None = None) -> None:
        super().__init__(ErrorCode.EMBEDDING_ERROR, message, detail)


class LLMError(SmartVoyageError):
    """千问模型调用异常。"""

    def __init__(self, message: str = "模型服务暂时不可用，请稍后重试", detail: str | None = None) -> None:
        super().__init__(ErrorCode.LLM_UNAVAILABLE, message, detail)


# 注：配置异常（ConfigError）定义在 config/settings.py 中 —— 它属于配置层，
# 且在读取配置阶段就会抛出（此时本模块的异常体系尚未参与），故不在此重复定义，
# 避免出现两个同名但类型不同的 ConfigError 造成 except 捕获不到的问题。
