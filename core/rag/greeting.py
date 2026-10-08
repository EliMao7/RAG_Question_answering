"""问候语识别与分流。

对应《教师智能问答系统技术文档》3.8 节。

日常问候（"你好"、"在吗"、"谢谢"）走检索链路既不经济也无必要：
一次问候会白耗一次嵌入调用 + 一次向量检索 + 一次 LLM 生成。

因此采用**规则优先 + 长度兜底**的前置分流：

- 长度兜底：超过 10 个字的句子几乎不可能是纯问候，直接放行到检索链路；
- 规则匹配：命中词表则返回固定回复，响应时间从 ~2s 降到 < 50ms，且不消耗 LLM 配额。

局限与演进：纯规则无法覆盖「你好，我想问下 Transformer」这类**问候+问题**的混合句。
当前策略刻意保守——只要句子里还有别的信息就不当问候处理，宁可多花一次 LLM 调用，
也不要把用户的真实问题当成问候敷衍掉。
"""
from __future__ import annotations

from core.rag.prompt import (
    GREETING_MAX_LENGTH,
    GREETING_PATTERNS,
    GREETING_REPLIES,
)


def route_greeting(question: str) -> tuple[str | None, str | None]:
    """判断是否为问候语，是则返回固定回复。

    :return: ``(意图, 回复)``；非问候语返回 ``(None, None)``

    用法::

        intent, reply = route_greeting("你好")
        if reply:
            return reply        # 走轻量路径，跳过检索与 LLM
    """
    text = question.strip().lower()

    # 长度兜底：长句子优先按真实问题处理
    if not text or len(text) > GREETING_MAX_LENGTH:
        return None, None

    for intent, words in GREETING_PATTERNS.items():
        for word in words:
            # 要求以问候词开头，且长度接近问候词本身，
            # 避免把「你好我想问」这类带问题的句子误判为纯问候
            if text.startswith(word) and len(text) <= len(word) + 3:
                return intent, GREETING_REPLIES[intent]

    return None, None


def is_greeting(question: str) -> bool:
    """便捷判断：是否属于问候语。"""
    intent, _ = route_greeting(question)
    return intent is not None
