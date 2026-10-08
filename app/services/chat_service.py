"""问答服务：RAG 链路的编排与降级。

对应《教师智能问答系统技术文档》：
- 3.3 / 3.4 节：在线问答链路（预处理 → 向量化 → 检索 → 归并 → Prompt → 生成 → 落库）
- 3.8 节：问候语分流（跳过检索与 LLM）
- 3.9 节：**降级矩阵**（任一环节失败都不能让用户看到 500）
- 4.2.3 节：问答记录落库（含 token 用量、缓存命中、耗时）

链路编排集中在本模块，API 层只负责协议适配（HTTP/SSE）：
    这样同一套链路既能服务即时问答，也能服务流式问答，避免逻辑重复。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import Config, get_config
from core.rag.greeting import route_greeting
from core.rag.prompt import (
    build_fallback_answer,
    build_messages,
    build_messages_no_context,
    build_messages_with_history,
)
from infrastructure.db.mysql_client import MySQLClient
from infrastructure.embedding.embedding_client import EmbeddingClient
from infrastructure.llm.qwen_client import QwenClient
from infrastructure.vector_db.milvus_client import MilvusClient
from utils.exceptions import (
    EmbeddingError,
    ErrorCode,
    LLMError,
    SmartVoyageError,
    VectorDBError,
)
from utils.logger import get_logger

from app.services.cache_service import CacheService
from app.services.subject_service import SubjectService

logger = get_logger(__name__)

# 问题长度限制（技术文档 6.3.4，PRD 4.4.2）
MIN_QUESTION_LENGTH = 1
MAX_QUESTION_LENGTH = 2000

# 溯源摘要长度（技术文档 5.2.2「显示参考资料内容摘要（100字符）」）
SUMMARY_LENGTH = 100

# 带入 Prompt 的历史轮数上限（技术文档 3.6）
HISTORY_ROUNDS = 3


@dataclass
class PreparedChat:
    """一次问答的「检索与组装」结果。

    把「准备阶段」的产物独立出来，是为了让即时问答与流式问答共用同一份
    检索结果与 Prompt，避免两条路径各查一次向量库。
    """

    question: str
    subject_code: str | None
    subject_name: str
    is_from_cache: bool = False
    references: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    # 问候语分流：命中时直接返回 greeting_reply，不调用检索与 LLM
    is_greeting: bool = False
    greeting_reply: str | None = None
    # 降级标记：知识库不可用时置 True（技术文档 3.9）
    degraded: bool = False
    error_code: str | None = None
    retrieved_count: int = 0
    # 热点缓存命中的答案；非 None 表示无需调用 LLM
    cached_answer: str | None = None


class ChatService:
    """问答编排：检索 + 增强 + 生成 + 落库。"""

    def __init__(
        self,
        db: MySQLClient,
        cache_service: CacheService,
        subject_service: SubjectService,
        embedder: EmbeddingClient,
        milvus: MilvusClient,
        llm: QwenClient,
        config: Config | None = None,
    ) -> None:
        self.db = db
        self.cache_service = cache_service
        self.subject_service = subject_service
        self.embedder = embedder
        self.milvus = milvus
        self.llm = llm
        self.cfg = config or get_config()

    # ==================================================================
    # 阶段一：准备（校验 → 问候分流 → 缓存 → 检索 → 组装 Prompt）
    # ==================================================================
    async def prepare(
        self,
        question: str,
        session_id: str,
        subject_code: str | None = None,
        use_history: bool = True,
    ) -> PreparedChat:
        """完成检索与 Prompt 组装，返回可提交给 LLM 的上下文。

        :raises SmartVoyageError: ``INVALID_QUESTION_LENGTH`` / ``INVALID_SUBJECT``
        """
        question = self._validate_question(question)
        subject_code = self.subject_service.validate(subject_code)
        subject_name = await self.subject_service.get_name(subject_code)

        prepared = PreparedChat(
            question=question,
            subject_code=subject_code,
            subject_name=subject_name,
        )

        # ---- 1. 问候语分流：不检索、不调用 LLM（技术文档 3.8）----
        intent, reply = route_greeting(question)
        if reply:
            prepared.is_greeting = True
            prepared.greeting_reply = reply
            logger.info("问候语分流命中：intent=%s", intent)
            return prepared

        # ---- 2. 热点缓存：命中则跳过检索与生成（技术文档 8.2）----
        cached = await self.cache_service.get_answer(question, subject_code)
        if cached and cached.get("answer"):
            prepared.references = cached.get("references") or []
            prepared.cached_answer = cached["answer"]
            prepared.is_from_cache = True
            logger.info("热点缓存命中：subject=%s", subject_code or "全学科")
            return prepared

        # ---- 3. 检索（含降级处理，技术文档 3.9）----
        references = await self._retrieve(question, subject_code, subject_name, prepared)

        # ---- 4. 检索为空 ----
        if not references:
            prepared.references = []
            if prepared.degraded:
                # 降级路径：messages 已在 _retrieve 中构造成「无参考资料」Prompt，
                # 此处**不能清空**，否则会退化成兜底话术、丢失通用知识回答能力
                return prepared
            # 正常检索但确实没查到：走兜底话术，**不调用 LLM 编造**
            prepared.messages = []
            prepared.error_code = ErrorCode.EMPTY_RETRIEVAL.code
            return prepared

        # ---- 5. 组装 Prompt（可选带入多轮历史，技术文档 3.6）----
        prepared.references = references
        if use_history:
            history = await self._load_history(session_id)
            prepared.messages = build_messages_with_history(
                question, references, history, subject_name, HISTORY_ROUNDS
            )
        else:
            prepared.messages = build_messages(question, references, subject_name)

        return prepared

    async def _retrieve(
        self,
        question: str,
        subject_code: str | None,
        subject_name: str,
        prepared: PreparedChat,
    ) -> list[dict]:
        """执行「向量化 → Milvus 检索 → 构造溯源信息」，失败时降级。

        降级策略（技术文档 3.9）：
        - 向量化/Milvus 故障 → 不报 500，改为「用通用知识回答」并显著标注；
        - 检索有结果但为空 → 交由上层走兜底话术。
        """
        try:
            vector = await self.embedder.aencode(question)
            parents = await self.milvus.aretrieve(vector, subject_code=subject_code or "all")
        except (EmbeddingError, VectorDBError) as exc:
            # 记录故障但继续服务：用户拿到标注过的通用答案，好过一个 503
            logger.error("检索链路降级：%s", exc)
            prepared.degraded = True
            prepared.error_code = (
                ErrorCode.EMBEDDING_UNAVAILABLE.code
                if isinstance(exc, EmbeddingError)
                else ErrorCode.VECTOR_DB_UNAVAILABLE.code
            )
            # 降级路径使用「无参考资料」的 Prompt，要求模型声明答案未经课程资料核对
            prepared.messages = build_messages_no_context(question, subject_name)
            return []

        prepared.retrieved_count = sum(len(p.get("child_chunks", [])) for p in parents)
        return await self._build_references(parents)

    async def _build_references(self, parents: list[dict]) -> list[dict]:
        """把 Milvus 归并出的父块转成溯源信息。"""
        name_map = await self.subject_service.get_name_map()
        references: list[dict] = []

        for index, parent in enumerate(parents, start=1):
            content = parent.get("content") or ""
            code = parent.get("subject_code")
            page_no = parent.get("page_no")

            references.append({
                "index": index,
                "subject_code": code,
                "subject_name": name_map.get(code, code or "未知"),
                "source": parent.get("source"),
                # -1 表示无页码信息（如 DOCX），对外统一表现为 null
                "page_no": page_no if isinstance(page_no, int) and page_no > 0 else None,
                "score": round(float(parent.get("score", 0.0)), 4),
                "summary": self._summarize(content),
                # content 仅供内部注入 Prompt 使用；对外响应由 Pydantic 模型
                # 自动丢弃未声明字段，不会泄露到接口
                "content": content,
            })

        return references

    async def _load_history(self, session_id: str) -> list[dict]:
        """取最近若干轮历史（失败不阻断问答）。"""
        try:
            history = await self.db.get_history(session_id, limit=HISTORY_ROUNDS)
            return history.get("records", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("历史上下文加载失败，按无历史处理：%s", exc)
            return []

    # ==================================================================
    # 阶段二：生成
    # ==================================================================
    def local_response(self, prepared: PreparedChat) -> dict | None:
        """无需调用 LLM 就能作答的情况（问候 / 缓存 / 兜底）。

        把这三种情况集中在一处判断，保证即时问答与流式问答走同一套分支逻辑，
        不会出现「流式路径忘了处理缓存命中」这类不对称的 bug。

        :return: 响应 dict；返回 None 表示需要调用 LLM
        """
        # 问候语：固定回复，零 API 消耗（技术文档 3.8）
        if prepared.is_greeting:
            return self._local_result(prepared.greeting_reply or "", is_from_cache=False)

        # 热点缓存命中（技术文档 8.2）
        if prepared.cached_answer is not None:
            return self._local_result(prepared.cached_answer, is_from_cache=True)

        # 检索为空：兜底话术，不调用 LLM 编造（技术文档 3.9）
        if not prepared.messages:
            return self._local_result(build_fallback_answer(), is_from_cache=False)

        return None

    @staticmethod
    def _local_result(answer: str, is_from_cache: bool) -> dict:
        """构造「未调用 LLM」的响应体（token 用量恒为 0）。"""
        return {
            "answer": answer,
            "is_from_cache": is_from_cache,
            "model_name": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "stop",
        }

    async def generate(self, prepared: PreparedChat) -> dict:
        """非流式生成（即时回答模式，接口 6.3.4）。

        :return: ``{"answer", "is_from_cache", "model_name", "prompt_tokens",
                    "completion_tokens", "finish_reason"}``
        """
        # 问候 / 缓存 / 兜底三种情况无需调用 LLM
        local = self.local_response(prepared)
        if local is not None:
            return local

        response = await self.llm.chat(prepared.messages)
        return {
            "answer": response.content,
            "is_from_cache": False,
            "model_name": response.model or self.llm.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "finish_reason": response.finish_reason or "stop",
        }

    async def save_result(
        self,
        prepared: PreparedChat,
        session_id: str,
        result: dict,
    ) -> int:
        """把问答结果落库，并写入热点缓存。

        写入顺序（技术文档 4.6）：MySQL 优先，缓存其次。
        MySQL 失败时**答案仍会返回给用户**（用户体验优先于记录完整性），
        因此这里捕获异常后仅告警。
        """
        answer = result.get("answer", "") or ""
        finish_reason = result.get("finish_reason", "stop")

        # ---- 落库 ----
        record_id = 0
        try:
            record_id = await self.db.save_chat_record(
                session_id=session_id,
                user_question=prepared.question,
                answer=answer,
                subject_code=prepared.subject_code,
                references=prepared.references,
                retrieved_count=prepared.retrieved_count,
                is_greeting=prepared.is_greeting,
                is_from_cache=bool(result.get("is_from_cache")),
                model_name=result.get("model_name"),
                prompt_tokens=int(result.get("prompt_tokens", 0) or 0),
                completion_tokens=int(result.get("completion_tokens", 0) or 0),
                response_time_ms=int(result.get("response_time_ms", 0) or 0),
                finish_reason=finish_reason,
                error_code=prepared.error_code,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("问答记录落库失败（答案仍正常返回）：%s", exc)

        # ---- 会话状态维护：标题取首个问题前 30 字 ----
        try:
            await self.db.set_session_title_if_empty(session_id, prepared.question[:30])
            await self.db.touch_session(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话状态更新失败：%s", exc)

        # ---- 写热点缓存 ----
        # 只缓存「有知识依据的正常答案」：问候语、兜底话术、降级答案都不缓存，
        # 否则知识库恢复后用户仍会拿到旧的降级结果。
        if (
            not prepared.is_greeting
            and not prepared.degraded
            and prepared.references
            and finish_reason == "stop"
            and answer
        ):
            await self.cache_service.set_answer(
                prepared.question, answer, prepared.references, prepared.subject_code
            )

        return record_id

    # ==================================================================
    # 内部工具
    # ==================================================================
    @staticmethod
    def _validate_question(question: str) -> str:
        """校验并清洗问题文本。

        :raises SmartVoyageError: ``INVALID_QUESTION_LENGTH``
        """
        if not isinstance(question, str):
            raise SmartVoyageError(
                ErrorCode.INVALID_QUESTION_LENGTH, "请输入有效的问题内容"
            )
        # 压缩连续空白，避免用空白字符绕过长度校验
        cleaned = " ".join(question.split())
        if not (MIN_QUESTION_LENGTH <= len(cleaned) <= MAX_QUESTION_LENGTH):
            raise SmartVoyageError(
                ErrorCode.INVALID_QUESTION_LENGTH,
                f"问题长度需在 {MIN_QUESTION_LENGTH}~{MAX_QUESTION_LENGTH} 字符之间",
                detail=f"实际长度={len(cleaned)}",
            )
        return cleaned

    @staticmethod
    def _summarize(content: str) -> str:
        """生成参考资料摘要。

        父块正文以 ``【来源：xxx】`` 开头（chunker 写入的溯源标注），
        摘要里去掉这行再截取，避免摘要被来源标记占满。
        """
        text = content or ""
        if text.startswith("【来源："):
            _, _, text = text.partition("\n")
        text = " ".join(text.split())
        return text[:SUMMARY_LENGTH]
