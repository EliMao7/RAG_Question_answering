"""文本向量化（Embedding）客户端。

对应技术设计文档：
- 2.1.2 节：向量化处理，维度 1536，单条延迟 < 100ms，支持批量
- 3.1 节：问答流程第 3 步「查询向量化」
- 8.1.1 节：API Key 从环境变量读取

同时提供 **同步** 与 **异步** 两套接口，原因：
    Milvus 客户端（pymilvus）是同步库，其 ``retrieve()`` 内部需要同步拿到
    查询向量，因此必须有 :meth:`encode`；
    而在线问答链路是 async 的，批量建库时走 :meth:`aencode_batch` 才能并发提速。
    两者共用同一套请求/解析逻辑，避免行为不一致。

⚠️ 维度必须与 Milvus 集合一致（技术文档 2.2.2 定义 dim=1536）：
    向量化模型返回的维度由 ``[embedding].model`` + ``[embedding].dimensions``
    共同决定，且不同模型支持的维度范围不同。本客户端在首次调用后会**校验实际
    返回维度**，不符则立刻抛出明确错误，而不是等到写入 Milvus 时报晦涩的错误。
"""
from __future__ import annotations

import asyncio
from typing import Sequence

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Config, get_config
from utils.exceptions import EmbeddingError
from utils.logger import get_logger

logger = get_logger(__name__)

# 上游限流 / 服务端错误可重试
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EmbeddingClient:
    """文本向量化客户端（同步 + 异步）。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or get_config()
        self.model = cfg.get("embedding", "model")
        self.dimensions = cfg.get_int("embedding", "dimensions", 1536)
        self.batch_size = cfg.get_int("embedding", "batch_size", 25)
        # 复用 LLM 的 base_url：DashScope 兼容模式下 /embeddings 同源
        self.base_url = cfg.get("llm", "dashscope_base_url").rstrip("/")

        # httpx 要求：不给默认值时，四个超时参数必须全部显式指定，否则 ValueError
        self._timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

        # 维度自检只做一次，避免每次请求都判断
        self._dimension_checked = False

    # ------------------------------------------------------------------
    # 客户端与请求头
    # ------------------------------------------------------------------
    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {get_config().dashscope_api_key}",
            "Content-Type": "application/json",
        }

    @property
    def sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url, timeout=self._timeout, headers=self._headers
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers=self._headers,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            )
        return self._async_client

    def close(self) -> None:
        """关闭同步客户端。"""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """关闭异步客户端。"""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # ------------------------------------------------------------------
    # 同步接口（供同步的 Milvus 检索链路使用）
    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(EmbeddingError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def encode(self, text: str) -> list[float]:
        """单条文本向量化（技术文档 3.2.1 中 ``embedding_client.encode(query)``）。"""
        vectors = self._post_embeddings_sync([text])
        return vectors[0]

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化（同步），自动按 ``batch_size`` 分批。

        建库场景下批量调用可显著减少 HTTP 往返次数。
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            vectors.extend(self._post_embeddings_sync(batch))
            logger.debug("向量化进度：%s/%s", min(start + len(batch), len(texts)), len(texts))
        return vectors

    # ------------------------------------------------------------------
    # 异步接口（供在线问答链路与建库脚本使用）
    # ------------------------------------------------------------------
    async def aencode(self, text: str) -> list[float]:
        """单条文本向量化（异步）。"""
        vectors = await self._post_embeddings_async([text])
        return vectors[0]

    async def aencode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化（异步），分批串行、批内并发请求。

        这里刻意不把全部批次一次性并发：DashScope 对并发有配额限制，
        无限并发会触发 429。分批 + 有限并发是吞吐与稳定性的折中。
        """
        if not texts:
            return []

        batches = [
            list(texts[i:i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        results = await asyncio.gather(
            *(self._post_embeddings_async(batch) for batch in batches)
        )
        # gather 保证顺序与入参一致，直接展平即可
        return [vec for batch_result in results for vec in batch_result]

    # ------------------------------------------------------------------
    # HTTP 调用
    # ------------------------------------------------------------------
    def _build_payload(self, texts: Sequence[str]) -> dict:
        """组装请求体。

        ``dimensions`` 需显式传入：部分模型的默认维度与 Milvus 集合不一致，
        不显式指定会导致写入时维度不匹配。
        """
        return {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }

    @retry(
        retry=retry_if_exception_type(EmbeddingError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _post_embeddings_sync(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self.sync_client.post("/embeddings", json=self._build_payload(texts))
        except httpx.HTTPError as exc:
            raise EmbeddingError("向量化服务连接失败", detail=str(exc)) from exc

        if response.status_code != 200:
            raise self._build_error(response.status_code, response.text)
        return self._parse_response(response.json(), len(texts))

    @retry(
        retry=retry_if_exception_type(EmbeddingError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def _post_embeddings_async(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = await self.async_client.post(
                "/embeddings", json=self._build_payload(texts)
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError("向量化服务连接失败", detail=str(exc)) from exc

        if response.status_code != 200:
            raise self._build_error(response.status_code, response.text)
        return self._parse_response(response.json(), len(texts))

    def _parse_response(self, data: dict, expected_count: int) -> list[list[float]]:
        """解析响应并按 ``index`` 归位。

        上游不保证返回顺序与入参一致，必须按 ``index`` 排序后再使用，
        否则会出现「A 文本配了 B 向量」的静默错位（检索结果全乱）。
        """
        items = data.get("data") or []
        if not items:
            raise EmbeddingError("向量化返回为空", detail=str(data)[:300])

        items = sorted(items, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") or [] for item in items]

        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"向量化返回条数不符：期望 {expected_count}，实际 {len(vectors)}",
                detail=str(data)[:300],
            )

        self._check_dimension(vectors[0])
        return vectors

    def _check_dimension(self, vector: list[float]) -> None:
        """校验实际维度与配置是否一致（仅首次调用时执行）。"""
        if self._dimension_checked:
            return
        actual = len(vector)
        if actual != self.dimensions:
            raise EmbeddingError(
                f"向量维度不匹配：模型 {self.model} 返回 {actual} 维，"
                f"配置 [embedding].dimensions={self.dimensions}。\n"
                f"该维度必须与 Milvus 集合的向量维度（技术文档 2.2.2 定义 1536）一致。\n"
                f"请调整 [embedding].model 或 [embedding].dimensions 后重试。"
            )
        self._dimension_checked = True
        logger.info("向量化维度校验通过：%s 维，模型 %s", actual, self.model)

    @staticmethod
    def _build_error(status_code: int, body: str) -> EmbeddingError:
        """把上游错误翻译为业务异常；限流与 5xx 允许重试。"""
        detail = f"HTTP {status_code}: {body[:500]}"

        # 批量大小超限：各模型的 batch 上限不同（v1/v2/v3 为 25，v4 为 10），
        # 上游的报错文案不含「怎么改」，这里补上可执行的修复建议。
        if "batch size is invalid" in body:
            return EmbeddingError(
                "向量化批量超限：请调小 config.ini 的 [embedding].batch_size"
                "（text-embedding-v4 上限 10 条，v1/v2/v3 上限 25 条）",
                detail=detail,
            )

        if status_code == 401:
            return EmbeddingError("向量化鉴权失败，请检查 DASHSCOPE_API_KEY", detail=detail)
        if status_code == 429:
            return EmbeddingError("向量化请求过于频繁，请稍后重试", detail=detail)
        if status_code >= 500:
            return EmbeddingError("向量化服务暂时不可用", detail=detail)
        return EmbeddingError("向量化失败，请检查模型与维度配置", detail=detail)

def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """计算余弦相似度（离线评测 / 单测用）。

    说明：Milvus 侧使用 IP（内积）作为度量。当向量已做归一化时，
    内积等价于余弦相似度；若未归一化，二者会有差异，评测时需注意。
    """
    import math

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
