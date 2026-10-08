"""请求 / 响应数据模型（Pydantic）。

对应《教师智能问答系统技术文档》6.3 节的接口契约。

设计取舍：请求模型**刻意放宽类型约束**
    技术文档 6.3.4 规定了精确的错误码（``INVALID_QUESTION_LENGTH`` /
    ``INVALID_SUBJECT`` / ``INVALID_SESSION_ID``）。若把长度、UUID 格式等约束
    写在 Pydantic 里，FastAPI 会先返回 422 通用校验错误，拿不到文档要求的错误码。
    因此这里只保留必要类型，把业务校验放在服务层，逐条抛出精确错误码。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    """创建会话请求（接口 6.3.2）。字段均可选。"""

    subject_code: str | None = Field(
        default=None, description="学科代码；不传或传 null 表示全学科"
    )


class ChatRequest(BaseModel):
    """问答请求（接口 6.3.4 / 6.4）。"""

    session_id: str = Field(description="会话 ID，UUIDv4 格式")
    question: str = Field(description="用户问题，长度 1~2000 字符")
    subject_code: str | None = Field(
        default=None, description="学科代码；不传则沿用会话的学科"
    )
    use_history: bool = Field(default=True, description="是否带入多轮对话上下文")


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------


class Reference(BaseModel):
    """溯源信息（接口 6.3.4 的 references 元素）。"""

    index: int = Field(description="序号，从 1 开始")
    subject_code: str | None = Field(default=None, description="学科代码")
    subject_name: str = Field(description="学科名称")
    source: str | None = Field(default=None, description="来源文件名")
    page_no: int | None = Field(default=None, description="PDF 页码；-1 表示无")
    score: float = Field(description="相似度分数")
    summary: str = Field(description="参考资料内容摘要（100 字符）")


class SubjectItem(BaseModel):
    """学科列表元素（接口 6.3.3）。"""

    code: str | None = Field(description="学科代码；null 表示全学科")
    name: str = Field(description="学科名称")
    is_default: bool = Field(default=False, description="是否为默认选项")
    doc_count: int | None = Field(default=None, description="该学科的知识块数量")


class ChatResponse(BaseModel):
    """即时问答响应（接口 6.3.4）。"""

    record_id: int = Field(description="本次问答的记录 ID")
    answer: str = Field(description="助手答案")
    references: list[Reference] = Field(default_factory=list, description="溯源信息")
    response_time_ms: int = Field(description="端到端耗时（毫秒）")
    is_from_cache: bool = Field(default=False, description="是否命中热点缓存")
    finish_reason: str = Field(default="stop", description="stop / interrupted / error")


class SessionResponse(BaseModel):
    """创建会话响应（接口 6.3.2）。"""

    session_id: str
    subject_code: str | None = None
    created_at: str
    welcome_message: str
    subjects: list[dict[str, Any]]


class HistoryRecord(BaseModel):
    """历史记录元素（接口 6.3.5）。"""

    record_id: int
    question: str
    answer: str
    subject_code: str | None = None
    references: list[dict] = Field(default_factory=list)
    retrieved_count: int = 0
    is_from_cache: bool = False
    response_time_ms: int = 0
    created_at: str


class HistoryResponse(BaseModel):
    """历史查询响应（接口 6.3.5，游标分页）。"""

    records: list[HistoryRecord]
    next_cursor: int | None = Field(
        default=None, description="下一页游标；为 null 表示没有更多数据"
    )
    has_more: bool = False
    total: int = 0


class ClearHistoryResponse(BaseModel):
    """清空历史响应（接口 6.3.6）。"""

    success: bool
    cleared_count: int
    message: str


class ComponentStatus(BaseModel):
    """健康检查中的单个组件状态（接口 6.3.1）。"""

    status: str = Field(description="up / down")
    latency_ms: int | None = None
    detail: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """健康检查响应（接口 6.3.1）。"""

    status: str = Field(description="healthy / degraded / unhealthy")
    timestamp: str
    version: str
    components: dict[str, Any]


class ErrorDetail(BaseModel):
    """统一错误响应体（接口 6.1）。"""

    code: str
    message: str
    request_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    """统一错误响应外壳。"""

    error: ErrorDetail
