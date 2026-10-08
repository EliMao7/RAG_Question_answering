"""后端接口端到端冒烟测试。

逐个调用《教师智能问答系统技术文档》第 6 章定义的全部接口，
包含正常路径与错误路径，用于快速验证后端是否可用。

前置条件：服务已启动
    uvicorn app.main:app --port 8000

用法::
    python scripts/smoke_test_api.py [--base-url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录并打印一条断言结果。"""
    (PASS if ok else FAIL).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f" — {detail}" if detail else ""))


async def test_health(client: httpx.AsyncClient) -> None:
    print("\n[1] GET /api/health")
    resp = await client.get("/api/health")
    body = resp.json()
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    check("聚合状态", body.get("status") in ("healthy", "degraded"),
          f"status={body.get('status')}")
    for comp in ("mysql", "redis", "milvus", "llm"):
        state = body.get("components", {}).get(comp, {}).get("status")
        check(f"组件 {comp}", state in ("up", "down"), f"{state}")
    print(f"       → {json.dumps(body.get('components'), ensure_ascii=False)}")


async def test_session(client: httpx.AsyncClient) -> str | None:
    print("\n[2] POST /api/session")
    resp = await client.post("/api/session", json={"subject_code": "ai"})
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code != 200:
        print(f"       → {resp.text[:200]}")
        return None

    body = resp.json()
    session_id = body.get("session_id", "")
    check("返回 session_id", bool(session_id), session_id)
    check("session_id 为 UUID", len(session_id) == 36, f"{len(session_id)} 字符")
    check("返回欢迎语", bool(body.get("welcome_message")))
    check("返回学科列表", isinstance(body.get("subjects"), list),
          f"{len(body.get('subjects') or [])} 项")
    return session_id


async def test_subjects(client: httpx.AsyncClient) -> None:
    print("\n[3] GET /api/subjects")
    resp = await client.get("/api/subjects")
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code != 200:
        return

    items = resp.json().get("subjects", [])
    check("首项为全学科", bool(items) and items[0].get("code") is None,
          f"code={items[0].get('code') if items else 'N/A'}")
    check("包含 5 个学科", len(items) == 6, f"共 {len(items)} 项（含全学科）")
    ai = next((i for i in items if i.get("code") == "ai"), {})
    check("ai 学科有知识块", (ai.get("doc_count") or 0) > 0, f"doc_count={ai.get('doc_count')}")


async def test_chat(client: httpx.AsyncClient, session_id: str) -> int | None:
    print("\n[4] POST /api/chat（即时问答，走完整 RAG 链路）")
    started = time.perf_counter()
    resp = await client.post("/api/chat", json={
        "session_id": session_id,
        "question": "什么是Transformer？",
        "subject_code": "ai",
    }, timeout=120)
    cost_ms = (time.perf_counter() - started) * 1000
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code != 200:
        print(f"       → {resp.text[:300]}")
        return None

    body = resp.json()
    answer = body.get("answer", "")
    refs = body.get("references", [])
    check("返回非空答案", bool(answer.strip()), f"{len(answer)} 字符")
    check("返回溯源信息", len(refs) > 0, f"{len(refs)} 条")
    if refs:
        top = refs[0]
        check("溯源含 source", bool(top.get("source")), str(top.get("source")))
        check("溯源含 score", isinstance(top.get("score"), (int, float)),
              f"score={top.get('score')}")
        check("溯源含 subject_name", bool(top.get("subject_name")),
              str(top.get("subject_name")))
    check("返回 record_id", (body.get("record_id") or 0) > 0, f"id={body.get('record_id')}")
    print(f"       → 耗时 {cost_ms:.0f}ms | {answer[:60]}…")
    return body.get("record_id")


async def test_greeting(client: httpx.AsyncClient, session_id: str) -> None:
    print("\n[5] POST /api/chat（问候语分流，应跳过检索与 LLM）")
    started = time.perf_counter()
    resp = await client.post("/api/chat", json={
        "session_id": session_id, "question": "你好",
    }, timeout=60)
    cost_ms = (time.perf_counter() - started) * 1000
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code != 200:
        return
    body = resp.json()
    check("返回问候回复", "你好" in body.get("answer", ""), body.get("answer", "")[:40])
    check("无溯源（未检索）", not body.get("references"))
    check("响应 < 500ms", cost_ms < 500, f"{cost_ms:.0f}ms")


async def test_stream(client: httpx.AsyncClient, session_id: str) -> None:
    print("\n[6] POST /api/chat/stream（SSE 流式）")
    events: list[dict] = []
    first_delta_ms: float | None = None
    started = time.perf_counter()

    try:
        async with client.stream("POST", "/api/chat/stream", json={
            "session_id": session_id,
            "question": "自注意力机制是怎么工作的？",
            "subject_code": "ai",
        }, timeout=120) as resp:
            check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
            check("Content-Type 为 SSE",
                  "text/event-stream" in resp.headers.get("content-type", ""),
                  resp.headers.get("content-type", ""))
            check("禁用代理缓冲",
                  resp.headers.get("x-accel-buffering") == "no",
                  str(resp.headers.get("x-accel-buffering")))

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                events.append(event)
                if event.get("type") == "delta" and first_delta_ms is None:
                    first_delta_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001
        check("SSE 流读取", False, f"{type(exc).__name__}: {exc}")
        return

    types = [e.get("type") for e in events]
    answer = "".join(e.get("content", "") for e in events if e.get("type") == "delta")

    check("首个事件为 start", types and types[0] == "start", str(types[:1]))
    check("收到多个 delta", types.count("delta") > 3, f"{types.count('delta')} 个")
    check("收到 citation", "citation" in types)
    check("最后事件为 done", types and types[-1] == "done", str(types[-1:]))
    check("无 error 事件", "error" not in types)
    check("拼接出完整答案", len(answer) > 20, f"{len(answer)} 字符")
    if first_delta_ms is not None:
        check("首字延迟 < 1000ms", first_delta_ms < 1000, f"{first_delta_ms:.0f}ms")
    print(f"       → 事件 {len(events)} 个 | {answer[:60]}…")


async def test_history(client: httpx.AsyncClient, session_id: str) -> None:
    print("\n[7] GET /api/history（游标分页）")
    resp = await client.get("/api/history", params={"session_id": session_id, "limit": 2})
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code != 200:
        print(f"       → {resp.text[:200]}")
        return

    body = resp.json()
    records = body.get("records", [])
    check("返回记录", len(records) > 0, f"{len(records)} 条")
    check("含 total 字段", "total" in body, f"total={body.get('total')}")
    check("含 has_more 字段", "has_more" in body, f"has_more={body.get('has_more')}")
    if records:
        first = records[0]
        check("记录含 question", bool(first.get("question")))
        check("记录含 answer", bool(first.get("answer")))
        check("倒序（最新在前）", records[0]["record_id"] >= records[-1]["record_id"])

    # 翻页：用 next_cursor 再取一页，验证游标有效
    if body.get("has_more") and body.get("next_cursor"):
        resp2 = await client.get("/api/history", params={
            "session_id": session_id, "limit": 2, "cursor": body["next_cursor"],
        })
        page2 = resp2.json().get("records", [])
        check("游标翻页成功", resp2.status_code == 200 and len(page2) > 0,
              f"第二页 {len(page2)} 条")
        if page2 and records:
            check("第二页无重复", page2[0]["record_id"] < records[-1]["record_id"],
                  f"{page2[0]['record_id']} < {records[-1]['record_id']}")


async def test_errors(client: httpx.AsyncClient, session_id: str) -> None:
    print("\n[8] 错误路径（技术文档 6.5 错误码表）")

    cases = [
        ("非法学科", "/api/chat",
         {"session_id": session_id, "question": "测试", "subject_code": "python"},
         400, "INVALID_SUBJECT"),
        ("空问题", "/api/chat",
         {"session_id": session_id, "question": "", "subject_code": "ai"},
         400, "INVALID_QUESTION_LENGTH"),
        ("超长问题", "/api/chat",
         {"session_id": session_id, "question": "问" * 2001, "subject_code": "ai"},
         400, "INVALID_QUESTION_LENGTH"),
        ("非法会话 ID", "/api/chat",
         {"session_id": "not-a-uuid", "question": "测试"},
         400, "INVALID_SESSION_ID"),
        ("会话不存在", "/api/chat",
         {"session_id": "00000000-0000-4000-8000-000000000000", "question": "测试"},
         404, "SESSION_NOT_FOUND"),
    ]

    for name, path, payload, expect_status, expect_code in cases:
        resp = await client.post(path, json=payload, timeout=60)
        body = resp.json()
        actual_code = (body.get("error") or {}).get("code")
        # 超长问题的校验可能先被 Pydantic 拦下，两者都算通过
        ok = resp.status_code == expect_status and actual_code in (expect_code, "INVALID_PARAMS")
        check(f"{name} → {expect_status} {expect_code}", ok,
              f"实际 status={resp.status_code}, code={actual_code}")
        if not ok:
            print(f"         {resp.text[:200]}")

    # 错误响应结构必须符合 6.1 节
    resp = await client.post("/api/chat", json={
        "session_id": session_id, "question": "测试", "subject_code": "python",
    }, timeout=60)
    error = resp.json().get("error", {})
    check("错误体含 code/message/request_id/timestamp",
          all(k in error for k in ("code", "message", "request_id", "timestamp")),
          f"字段={sorted(error.keys())}")


async def test_clear(client: httpx.AsyncClient, session_id: str) -> None:
    print("\n[9] DELETE /api/history（清空，幂等）")
    resp = await client.delete("/api/history", params={"session_id": session_id})
    check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
    body = resp.json()
    check("success 为 true", body.get("success") is True)
    check("返回 cleared_count", "cleared_count" in body, f"cleared={body.get('cleared_count')}")

    # 清空后应为空
    resp = await client.get("/api/history", params={"session_id": session_id})
    records = resp.json().get("records", [])
    check("清空后历史为空", len(records) == 0, f"{len(records)} 条")

    # 幂等：重复清空仍成功
    resp = await client.delete("/api/history", params={"session_id": session_id})
    check("重复清空幂等", resp.status_code == 200 and resp.json().get("success") is True,
          f"cleared={resp.json().get('cleared_count')}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="后端接口冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    print("=" * 70)
    print(f"后端接口冒烟测试 → {args.base_url}")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        try:
            await test_health(client)
        except Exception as exc:  # noqa: BLE001
            print(f"\n无法连接服务：{exc}")
            print("请先启动：uvicorn app.main:app --port 8000")
            return 1

        session_id = await test_session(client)
        await test_subjects(client)

        if session_id:
            await test_chat(client, session_id)
            await test_greeting(client, session_id)
            await test_stream(client, session_id)
            await test_history(client, session_id)
            await test_errors(client, session_id)
            await test_clear(client, session_id)

    print("\n" + "=" * 70)
    print(f"通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("\n失败项：")
        for item in FAIL:
            print(f"  - {item}")
    print("=" * 70)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
