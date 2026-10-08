"""通义千问（Qwen）模型调用接口。

对应技术设计文档：
- 2.3 / 11.1 节：模型 ``qwen3.6-plus``，DashScope 兼容模式
  ``https://dashscope.aliyuncs.com/compatible-mode/v1``
- 3.4.1 节：流式输出（``async for chunk in llm_client.stream_chat(messages)``）
- 5.2.2 节：生成参数 ``temperature=0.7 / top_p=0.9 / max_tokens=2048``
- 8.1.1 节：API Key 从环境变量 ``DASHSCOPE_API_KEY`` 读取，不落配置文件

为什么走「兼容模式」而不是 DashScope 原生 SDK：
    兼容模式暴露的是 OpenAI 风格的 ``/chat/completions`` 接口，
    可以直接用 ``httpx`` 发请求，不引入额外 SDK 依赖，也便于日后
    切换到其他兼容 OpenAI 协议的服务（如本地 vLLM）而无需改动业务代码。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Config, get_config
from utils.exceptions import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)

# 上游限流 / 服务端错误才重试；参数错误(400)重试没有意义
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# SSE 流结束标记
SSE_DONE = "[DONE]"


@dataclass
class LLMResponse:
    """非流式调用的返回结果。"""

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """总 token 数，用于技术文档 7.1.2 的成本统计。"""
        return self.prompt_tokens + self.completion_tokens


class QwenClient:
    """通义千问客户端，提供流式与非流式两种调用方式。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or get_config()
        self.model = cfg.get("llm", "model")
        self.base_url = cfg.get("llm", "dashscope_base_url").rstrip("/")

        # 生成参数（技术文档 5.2.2）
        self.temperature = cfg.get_float("llm", "temperature", 0.7)
        self.top_p = cfg.get_float("llm", "top_p", 0.9)
        self.max_tokens = cfg.get_int("llm", "max_tokens", 2048)

        # 思考模式开关：三态（None = 不传该参数，交给服务端默认）
        #
        # 为什么需要这个开关（实测结论）：
        #   qwen3 系列是「思考模型」，默认会先输出一大段 reasoning_content
        #   再输出正文 content。实测同一条问题：
        #     默认(思考开启)      → 首个 content 分片在 ~7960ms
        #     enable_thinking=false → 首个 content 分片在 ~851ms
        #   对 SSE 打字机效果而言，用户看到的是「8 秒无任何输出」，
        #   直接违反技术文档「流式首字延迟 < 1s」的指标。
        #   因此问答场景默认关闭思考模式。
        raw_thinking = cfg.get("llm", "enable_thinking", None)
        self.enable_thinking = self._parse_optional_bool(raw_thinking)

        # 并发闸门：限制同时在途的请求数，避免瞬时打满上游配额触发 429
        # （技术文档要求支撑 100 并发，这里保守取 32，可按配额调整）
        self._semaphore = asyncio.Semaphore(cfg.get_int("llm", "max_concurrency", 32))

        # 连接超时 5s / 读取超时 60s：
        # 读取超时给足是因为流式生成期间每次 aiter_lines 都要等上游吐字
        self._timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)

        # 延迟到首次使用时创建，避免在未配置 API Key 的环境里 import 即报错
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def client(self) -> httpx.AsyncClient:
        """惰性创建 HTTP 客户端（连接池复用，省去每次 TLS 握手约 100~200ms）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                # 长连接池：并发场景下复用 TCP 连接，显著降低延迟
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    @property
    def _api_key(self) -> str:
        """每次取 Key 都走配置层，便于轮换密钥后无需重启进程。"""
        return get_config().dashscope_api_key

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("千问客户端已关闭")

    async def __aenter__(self) -> "QwenClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # 流式调用（技术文档 3.4.1，主用方式）
    # ------------------------------------------------------------------
    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式调用，逐段吐出生成内容。

        用法::

            async for chunk in client.stream_chat(messages):
                # chunk 为增量文本，可直接写入 SSE 响应
                yield f"data: {json.dumps({'content': chunk})}\\n\\n"

        :param messages: OpenAI 风格消息列表 ``[{"role": "system"|"user"|"assistant", "content": ...}]``
        """
        payload = self._build_payload(messages, stream=True,
                                      temperature=temperature, max_tokens=max_tokens)

        async with self._semaphore:
            try:
                async with self.client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    # 流式响应无法用 raise_for_status()：此时 body 还没读完，
                    # 直接读出来才能拿到上游的具体报错信息
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", errors="replace")
                        raise LLMError(
                            self._friendly_message(response.status_code),
                            detail=f"HTTP {response.status_code}: {detail[:500]}",
                        )

                    async for line in response.aiter_lines():
                        chunk = self._parse_sse_line(line)
                        if chunk:
                            yield chunk

            except LLMError:
                raise
            except httpx.TimeoutException as exc:
                raise LLMError("模型响应超时，请稍后重试", detail=f"timeout: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMError("模型服务连接失败", detail=str(exc)) from exc

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        """解析单行 SSE 数据，返回增量文本（无需产出时返回 None）。

        DashScope 兼容模式的流式数据形如::

            data: {"choices":[{"delta":{"content":"你好"}}]}
            data: [DONE]
        """
        if not line or not line.startswith("data:"):
            return None

        data = line[len("data:"):].strip()
        if not data or data == SSE_DONE:
            return None

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            # 个别心跳行不是合法 JSON，跳过即可，不应中断整个流
            return None

        choices = parsed.get("choices") or []
        if not choices:
            return None
        # delta.content 在首个/末个分片中可能缺失
        return choices[0].get("delta", {}).get("content") or None

    # ------------------------------------------------------------------
    # 非流式调用
    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(2),                 # 首次 + 重试 1 次
        wait=wait_exponential(multiplier=1, max=4),  # 退避 1s、2s
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """非流式调用，一次性返回完整答案（简单问题的「即时回答模式」）。

        仅对可重试错误（限流、5xx、超时）做最多 1 次退避重试；
        参数错误不重试，直接抛出。
        """
        payload = self._build_payload(messages, stream=False,
                                     temperature=temperature, max_tokens=max_tokens)

        async with self._semaphore:
            try:
                response = await self.client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                raise LLMError("模型响应超时，请稍后重试", detail=f"timeout: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMError("模型服务连接失败", detail=str(exc)) from exc

        if response.status_code != 200:
            detail = response.text[:500]
            message = self._friendly_message(response.status_code)
            if response.status_code in RETRYABLE_STATUS:
                # 抛 LLMError 会被 tenacity 捕获并重试
                raise LLMError(message, detail=f"HTTP {response.status_code}: {detail}")
            raise LLMError(message, detail=f"HTTP {response.status_code}: {detail}")

        data = response.json()
        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict) -> LLMResponse:
        """把 OpenAI 风格的响应体解析为 :class:`LLMResponse`。"""
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        usage = data.get("usage") or {}

        return LLMResponse(
            content=(choice.get("message") or {}).get("content", ""),
            model=data.get("model", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=choice.get("finish_reason", "") or "",
            raw=data,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        messages: list[dict],
        stream: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """组装请求体。"""
        if not messages:
            raise LLMError("消息列表不能为空", detail="messages is empty")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }

        # 仅在显式配置时才携带该参数：部分模型（非思考模型）不识别它，
        # 无脑传参可能导致 400。未配置时交给服务端默认行为。
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking

        return payload

    @staticmethod
    def _parse_optional_bool(raw: Any) -> bool | None:
        """把配置字符串解析为三态布尔值。

        ``""`` / ``None`` → None（不传参）；其余按常见真值词解析。
        """
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if text == "":
            return None
        return text in {"1", "true", "yes", "on"}

    @staticmethod
    def _friendly_message(status_code: int) -> str:
        """把上游状态码翻译成可直接展示给用户的提示（PRD 4.2.1）。"""
        if status_code == 401:
            return "模型服务鉴权失败，请检查 DASHSCOPE_API_KEY 配置"
        if status_code == 429:
            return "当前咨询人数较多，请稍后再试"
        if status_code >= 500:
            return "模型服务暂时不可用，请稍后重试"
        return "模型调用失败，请稍后重试"

    # ------------------------------------------------------------------
    # 健康检查（技术文档 4.1.5 / 7.2）
    # ------------------------------------------------------------------
    async def health_check(self) -> str:
        """探测模型服务可用性，返回 ``available`` / ``unavailable``。

        为避免健康探针本身消耗 token 配额，这里调用轻量的 ``/models``
        接口（若上游不支持则退化为「仅校验密钥是否存在」）。
        """
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                return "available"
            # 上游未实现 /models 时，只要不是鉴权失败即可认为可用
            return "unavailable" if response.status_code == 401 else "available"
        except Exception as exc:  # noqa: BLE001 - 健康检查不应抛异常
            logger.warning("千问服务健康检查失败：%s", exc)
            return "unavailable"
