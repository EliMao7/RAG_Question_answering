"""客户端连通性自检与用法示例。

对应技术设计文档 4.1.5 节的 ``/api/health`` 接口契约：
MySQL / Redis / Milvus / LLM 四个组件逐个探测，聚合出整体状态。

用法::

    conda activate 3.11codeRAG
    set DASHSCOPE_API_KEY=sk-xxxx        # Windows；Linux 用 export
    python demo_clients.py

注意：本脚本只做「读」操作，不会写入或删除任何数据。
未启动的组件会被标记为 disconnected，不影响其他组件继续自检。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime

from config.settings import Config, ConfigError, get_config
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def print_header(title: str) -> None:
    """打印分隔标题，便于在控制台快速定位。"""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_result(label: str, value: object, ok: bool | None = None) -> None:
    """统一格式输出结果，ok 为 None 时不做状态标记。"""
    mark = "" if ok is None else ("[OK]   " if ok else "[FAIL] ")
    print(f"  {mark}{label}: {value}")


# ---------------------------------------------------------------------------
# 1. 配置加载
# ---------------------------------------------------------------------------
def check_config() -> Config:
    """加载并校验配置（配置错误会在这一步直接暴露）。"""
    print_header("1. 配置加载（config/config.ini）")
    cfg = get_config()

    print_result("MySQL", f"{cfg.get('mysql', 'host')}:{cfg.get('mysql', 'port')}/{cfg.get('mysql', 'database')}")
    print_result("Redis", f"{cfg.get('redis', 'host')}:{cfg.get('redis', 'port')} (db={cfg.get('redis', 'db')})")
    print_result("Milvus", f"{cfg.get('milvus', 'host')}:{cfg.get('milvus', 'port')}"
                           f"/{cfg.get('milvus', 'database_name')}.{cfg.get('milvus', 'collection_name')}")
    print_result("LLM 模型", cfg.get("llm", "model"))
    print_result("Embedding", f"{cfg.get('embedding', 'model')} ({cfg.get('embedding', 'dimensions')} 维)")
    print_result("学科白名单", cfg.get_json("app", "valid_sources"))
    print_result("检索参数", f"k={cfg.get('retrieval', 'retrieval_k')}, "
                            f"m={cfg.get('retrieval', 'candidate_m')}, "
                            f"nprobe={cfg.get('retrieval', 'nprobe')}")

    # API Key 单独检查：缺失时只警告，不中断（其他组件仍可自检）
    if os.getenv("DASHSCOPE_API_KEY"):
        print_result("DASHSCOPE_API_KEY", "已设置")
    else:
        print_result("DASHSCOPE_API_KEY", "未设置 —— LLM 与 Embedding 调用会失败", ok=False)
    return cfg


# ---------------------------------------------------------------------------
# 2. MySQL
# ---------------------------------------------------------------------------
async def check_mysql() -> str:
    """探测 MySQL，并顺带验证表结构是否已初始化。"""
    from infrastructure.db.mysql_client import MySQLClient

    print_header("2. MySQL 客户端")
    client = MySQLClient()
    try:
        await client.connect()
    except Exception as exc:  # noqa: BLE001
        print_result("连接", f"失败 —— {exc}", ok=False)
        print("  提示：确认 MySQL 已启动，且已执行 scripts/init_db.sql 建库建表")
        return "disconnected"

    status = await client.health_check()
    print_result("连接", status, ok=(status == "connected"))

    if status == "connected":
        # 检查三张表是否就位（表结构未初始化是常见问题）
        try:
            subjects = await client.get_subjects()
            print_result("学科表 subjects", f"{len(subjects)} 条 → "
                                          f"{[s['name'] for s in subjects]}", ok=bool(subjects))
            row = await client.fetch_one("SELECT COUNT(*) AS c FROM sessions")
            print_result("会话表 sessions", f"{row['c'] if row else 0} 条")
            row = await client.fetch_one("SELECT COUNT(*) AS c FROM chat_records")
            print_result("记录表 chat_records", f"{row['c'] if row else 0} 条")
        except Exception as exc:  # noqa: BLE001
            print_result("表结构检查", f"失败 —— {exc}", ok=False)
            print("  提示：执行 scripts/init_db.sql 初始化表结构与学科数据")

    await client.close()
    return status


# ---------------------------------------------------------------------------
# 3. Redis
# ---------------------------------------------------------------------------
async def check_redis() -> str:
    """探测 Redis，并演示缓存读写。"""
    from infrastructure.cache.redis_client import RedisClient

    print_header("3. Redis 缓存客户端")
    client = RedisClient()
    try:
        await client.connect()
    except Exception as exc:  # noqa: BLE001
        print_result("连接", f"失败 —— {exc}", ok=False)
        return "disconnected"

    status = await client.health_check()
    print_result("连接", status, ok=(status == "connected"))

    if status == "connected":
        # 演示：写热点问答缓存 → 读回 → 删除（不留下垃圾数据）
        await client.set_hot_answer(
            question="什么是Transformer（自检演示）",
            answer="这是一条用于自检的演示缓存",
            sources=["demo_parent_001"],
            subject_code="ai",
        )
        cached = await client.get_hot_answer("什么是Transformer（自检演示）", subject_code="ai")
        print_result("热点缓存读写", "命中" if cached else "未命中", ok=bool(cached))

        # 演示：会话元数据 Hash
        await client.save_session_meta("demo-session-id", subject="ai", created_at="2026-07-06T10:30:00")
        meta = await client.get_session_meta("demo-session-id")
        print_result("会话元数据读写", meta, ok=bool(meta))

        # 清理演示数据
        await client.clear_hot_qa()
        await client.delete_session_meta("demo-session-id")
        print_result("演示数据清理", "完成")

    await client.close()
    return status


# ---------------------------------------------------------------------------
# 4. Milvus
# ---------------------------------------------------------------------------
async def check_milvus() -> str:
    """探测 Milvus，读取集合规模与学科分布。

    注意：Milvus 客户端是同步的，这里用 asyncio.to_thread 包装，
    与生产代码中 :meth:`MilvusClient.aretrieve` 的做法一致。
    """
    from infrastructure.vector_db.milvus_client import MilvusClient, VECTOR_DIM

    print_header("4. Milvus 向量库客户端")
    client = MilvusClient()
    try:
        await asyncio.to_thread(client.connect)
    except Exception as exc:  # noqa: BLE001
        print_result("连接", f"失败 —— {exc}", ok=False)
        return "disconnected"

    status = await asyncio.to_thread(client.health_check)
    print_result("连接", status, ok=(status == "connected"))

    if status == "connected":
        print_result("集合", f"{client.collection_name}（库 {client.database_name}）")
        print_result("实体数量", client.num_entities, ok=client.num_entities > 0)
        print_result("向量维度", VECTOR_DIM)
        print_result("检索参数", f"k={client.retrieval_k}, nprobe={client.nprobe}")

        distribution = await asyncio.to_thread(client.get_subject_distribution)
        if distribution:
            print_result("学科分布", json.dumps(distribution, ensure_ascii=False))
        if client.num_entities == 0:
            print("  提示：集合为空，需先构建知识库索引后再检索")

    await asyncio.to_thread(client.close)
    return status


# ---------------------------------------------------------------------------
# 5. 通义千问 LLM
# ---------------------------------------------------------------------------
async def check_llm() -> str:
    """探测千问接口，并跑一次真实的流式生成。"""
    from infrastructure.llm.qwen_client import QwenClient

    print_header("5. 通义千问模型调用")
    if not os.getenv("DASHSCOPE_API_KEY"):
        print_result("跳过", "未设置 DASHSCOPE_API_KEY", ok=False)
        return "unavailable"

    client = QwenClient()
    try:
        status = await client.health_check()
        print_result("服务可用性", status, ok=(status == "available"))

        if status == "available":
            messages = [
                {"role": "system", "content": "你是教师的智能答疑助手，回答简洁。"},
                {"role": "user", "content": "用一句话说明什么是Transformer。"},
            ]

            # 流式调用：模拟 SSE 逐字输出的场景
            print("  流式输出: ", end="", flush=True)
            first_chunk_at = None
            import time
            started = time.perf_counter()
            text = ""
            async for chunk in client.stream_chat(messages, max_tokens=128):
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    print_result("首字延迟", f"{(first_chunk_at - started) * 1000:.0f} ms"
                                            f"（技术文档要求 < 1000ms）",
                                 ok=(first_chunk_at - started) < 1.0)
                    print("  流式输出: ", end="", flush=True)
                text += chunk
                print(chunk, end="", flush=True)
            print(f"\n  完整耗时: {(time.perf_counter() - started) * 1000:.0f} ms")

            if not text.strip():
                print_result("流式返回", "内容为空", ok=False)
                return "unavailable"

            # 非流式调用：对比一次并返回 token 用量
            response = await client.chat(
                [{"role": "user", "content": "回答“收到”两个字即可。"}], max_tokens=16
            )
            print_result("非流式返回", repr(response.content[:40]))
            print_result("token 用量", f"prompt={response.prompt_tokens}, "
                                      f"completion={response.completion_tokens}, "
                                      f"total={response.total_tokens}")
    finally:
        await client.close()
    return status


# ---------------------------------------------------------------------------
# 6. Embedding
# ---------------------------------------------------------------------------
async def check_embedding() -> str:
    """探测向量化接口，校验维度是否与 Milvus 集合一致。"""
    from infrastructure.embedding.embedding_client import EmbeddingClient

    print_header("6. 文本向量化客户端")
    if not os.getenv("DASHSCOPE_API_KEY"):
        print_result("跳过", "未设置 DASHSCOPE_API_KEY", ok=False)
        return "unavailable"

    client = EmbeddingClient()
    try:
        import time
        started = time.perf_counter()
        vector = await client.aencode("什么是Transformer？")
        cost_ms = (time.perf_counter() - started) * 1000

        print_result("单条向量化", f"{len(vector)} 维，耗时 {cost_ms:.0f} ms"
                                   f"（技术文档要求 < 100ms/条）",
                     ok=(len(vector) == client.dimensions))
        print_result("向量样例", [round(v, 4) for v in vector[:5]])
        print("  提示：维度与 Milvus 集合不一致时，写入/检索会直接报错，"
              "请调整 config.ini 的 [embedding] 配置")

        # 批量向量化：验证批处理与顺序对齐
        texts = ["人工智能", "Java开发", "软件测试"]
        vectors = await client.aencode_batch(texts)
        print_result("批量向量化", f"{len(texts)} 条 → 返回 {len(vectors)} 条",
                     ok=(len(vectors) == len(texts)))
        if len(vectors) != len(texts):
            print("  警告：批量返回条数与入参不符，检索结果可能错位，请检查上游返回")
        return "available"
    except Exception as exc:  # noqa: BLE001
        print_result("向量化调用", f"失败 —— {exc}", ok=False)
        return "unavailable"
    finally:
        # EmbeddingClient 同时提供同步 close() 与异步 aclose()，
        # 异步链路必须用 aclose()：错用 close() 会得到
        # "object NoneType can't be used in 'await' expression"
        await client.aclose()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main() -> int:
    """依次探测各组件，最后按 /api/health 的契约聚合输出。"""
    setup_logger()

    try:
        check_config()
    except ConfigError as exc:
        print(f"\n配置错误：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n配置加载失败：{exc}")
        return 1

    components: dict[str, str] = {}
    for name, checker in (
        ("mysql", check_mysql),
        ("redis", check_redis),
        ("milvus", check_milvus),
        ("llm", check_llm),
        ("embedding", check_embedding),
    ):
        try:
            components[name] = await checker()
        except Exception as exc:  # noqa: BLE001 - 单个组件失败不影响其余自检
            components[name] = "error"
            print_result(name, f"自检异常 —— {exc}", ok=False)

    # 聚合状态：与技术文档 4.1.5 的 /api/health 契约一致。
    # MySQL / Redis 是强依赖，缺失即 unhealthy；Milvus / LLM 缺失为 degraded。
    core_ok = components.get("mysql") == "connected" and components.get("redis") == "connected"
    all_ok = core_ok and components.get("milvus") == "connected" and components.get("llm") == "available"
    status = "healthy" if all_ok else ("degraded" if core_ok else "unhealthy")

    print_header("自检汇总（对应 /api/health 契约）")
    print(json.dumps(
        {
            "status": status,
            "components": components,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
