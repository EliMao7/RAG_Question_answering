# 教师智能问答系统 — 后端

基于《教师智能问答系统技术文档》（`教师智能问答系统技术文档.md`）实现的后端，
覆盖 **数据存储 / RAG 检索 / 大模型生成 / HTTP 接口** 全链路。

环境：conda 环境 **`3.11codeRAG`**（Python 3.11）

技术栈：FastAPI + Milvus 2.3 + MySQL 8 + Redis + 通义千问（DashScope 兼容模式）

---

## 一、目录结构

```
教师智能问答系统/
├── app/                          # 应用层（对应技术文档 5.1 节的 app/）
│   ├── main.py                   # FastAPI 入口：lifespan、中间件、异常处理、路由注册
│   ├── api/                      # 接口层：只做协议适配（HTTP / SSE）
│   │   ├── deps.py               #   依赖注入（从 app.state 取客户端与服务）
│   │   ├── health.py             #   GET  /api/health
│   │   ├── session.py            #   POST /api/session
│   │   ├── subjects.py           #   GET  /api/subjects
│   │   ├── chat.py               #   POST /api/chat、POST /api/chat/stream
│   │   └── history.py            #   GET /api/history、DELETE /api/history
│   ├── services/                 # 业务层：编排与降级策略
│   │   ├── session_service.py    #   会话生命周期、存在性校验
│   │   ├── subject_service.py    #   学科白名单校验、doc_count 统计
│   │   ├── cache_service.py      #   热点问答缓存、接口限流
│   │   └── chat_service.py       #   RAG 链路编排（核心）
│   ├── models/schemas.py         # Pydantic 请求 / 响应模型
│   └── core/request_id.py        # 请求追踪中间件
├── core/rag/                     # RAG 引擎层（对应文档 5.1 节的 app/rag/）
│   ├── document_loader.py        #   文档解析（PDF / DOCX / MD / TXT）
│   ├── chunker.py                #   父子分块（1200 / 300 / 50）
│   ├── prompt.py                 #   Prompt 模板与上下文组装
│   └── greeting.py               #   问候语分流
├── infrastructure/               # 基础设施层（对应文档 5.1 节的 app/clients/）
│   ├── db/mysql_client.py        #   MySQL：会话、记录、学科、游标分页
│   ├── cache/redis_client.py     #   Redis：会话元数据、热点缓存、限流
│   ├── vector_db/milvus_client.py#   Milvus：向量检索、父块归并
│   ├── llm/qwen_client.py        #   通义千问：流式 / 非流式
│   └── embedding/embedding_client.py # 文本向量化：1536 维
├── config/
│   ├── config.ini                # 配置基线（环境变量可覆盖）
│   └── settings.py               # 配置加载器 + 启动校验
├── utils/
│   ├── logger.py                 # 日志（轮转、UTF-8、跨平台路径）
│   └── exceptions.py             # 统一异常与错误码
├── scripts/
│   ├── init_db.sql               # 建库建表 + 学科数据 + 授权
│   ├── init_milvus.py            # 建 Milvus 集合与索引
│   ├── build_index.py            # 知识库构建（文档 → 分块 → 向量化 → Milvus）
│   └── smoke_test_api.py         # 接口端到端冒烟测试
├── demo_clients.py               # 客户端连通性自检
├── check_env.py                  # 依赖环境自检
└── requirements.txt
```

> 与技术文档目录的对应关系：文档里的 `app/clients/` 在本仓库中名为 `infrastructure/`，
> `app/rag/` 名为 `core/rag/`（`core/rag` 已包含分块器，与文档合并为一处）。
> 分层依赖方向单向：`app/api → app/services → core/rag + infrastructure → utils`。

---

## 二、快速开始

```bash
conda activate 3.11codeRAG

# 1) 配置 API Key（不写入配置文件，技术文档 8.1.1）
set DASHSCOPE_API_KEY=sk-xxxx          # Windows
export DASHSCOPE_API_KEY=sk-xxxx       # Linux / macOS

# 2) 初始化数据存储
python -c "..."                        # 执行 scripts/init_db.sql（见下方说明）
python scripts/build_index.py --data-dir "<文档目录>" --subject ai --recreate --yes

# 3) 启动后端
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 4) 接口冒烟测试（另开一个终端）
python scripts/smoke_test_api.py
```

接口文档（Swagger）：http://localhost:8000/docs

---

## 三、接口清单

| # | 方法 | 路径 | 说明 | 文档 |
|---|------|------|------|------|
| 1 | GET | `/api/health` | 健康检查，聚合探测 4 个组件 | 6.3.1 |
| 2 | POST | `/api/session` | 创建会话（返回 UUID + 欢迎语 + 学科列表） | 6.3.2 |
| 3 | GET | `/api/subjects` | 学科列表（含各学科知识块数 doc_count） | 6.3.3 |
| 4 | POST | `/api/chat` | 即时问答，一次性返回完整答案 | 6.3.4 |
| 5 | POST | `/api/chat/stream` | 流式问答（SSE 打字机效果） | 6.4 |
| 6 | GET | `/api/history` | 历史查询（游标分页） | 6.3.5 |
| 7 | DELETE | `/api/history` | 清空历史（幂等） | 6.3.6 |

**SSE 事件协议**（`/api/chat/stream`）：`start` → `delta`*n → `citation` → `done`，
异常时为 `error`。详见技术文档 6.4。

**统一错误响应**：

```json
{"error": {"code": "INVALID_SUBJECT", "message": "...", "request_id": "...", "timestamp": "..."}}
```

错误码映射见技术文档 6.5；请求追踪 ID 通过 `X-Request-Id` 请求头/响应头贯穿。

---

## 三、各客户端用法

### 3.1 MySQL（异步）

```python
import asyncio
from infrastructure import MySQLClient

async def main():
    async with MySQLClient() as db:
        # 会话
        await db.create_session(session_id="550e8400-e29b-41d4-a716-446655440000", subject_code="ai")
        await db.touch_session(session_id)                      # 刷新活跃时间
        await db.update_session_subject(session_id, "java")     # 切换学科

        # 问答记录
        record_id = await db.save_chat_record(
            session_id=session_id,
            user_question="什么是Transformer？",
            ai_answer="Transformer 是一种……",
            subject_code="ai",
            source_chunks=["ai_001_parent", "ai_007_parent"],   # 溯源用
            response_time_ms=1850,
        )

        # 历史（分页，倒序）
        history = await db.get_history(session_id, page=1, page_size=20)
        print(history["total"], history["records"])

        await db.clear_history(session_id)
        print(await db.health_check())      # connected / disconnected

asyncio.run(main())
```

### 3.2 Redis（异步）

```python
from infrastructure import RedisClient

async with RedisClient() as cache:
    # 会话元数据（Hash，TTL 24h）
    await cache.save_session_meta("session-id", subject="ai", created_at="2026-07-06T10:30:00")
    meta = await cache.get_session_meta("session-id")     # 命中自动续期

    # 热点问答（String/JSON，TTL 1h）—— 只在答案完整生成后写入
    await cache.set_hot_answer("什么是Transformer？", "答案……", sources=["ai_001_parent"], subject_code="ai")
    cached = await cache.get_hot_answer("什么是Transformer？", subject_code="ai")

    # 知识库重建后必须清理，否则旧答案会一直从缓存返回
    await cache.clear_hot_qa()

    # 限流计数（固定窗口）
    count = await cache.incr_rate("ip:127.0.0.1", window_seconds=60)
```

### 3.3 Milvus（RAG 检索核心）

```python
from infrastructure import MilvusClient, EmbeddingClient

embedder = EmbeddingClient()
milvus = MilvusClient()

# 同步链路（pymilvus 为同步库）
vector = embedder.encode("什么是Transformer？")
parents = milvus.retrieve(vector, subject_code="ai")     # 子块召回 → 父块归并

# 异步链路（FastAPI 中必须用这个，否则会阻塞事件循环）
parents = await milvus.aretrieve(vector, subject_code="ai")

for p in parents:
    print(p["score"], p["subject_code"], p["content"][:50])
```

返回结构（按相似度倒序）：

```python
{
    "parent_chunk_id": "ai_001_parent",
    "content": "……命中子块原文……",
    "subject_code": "ai",
    "score": 0.87,
    "child_chunks": [{"chunk_id": "ai_001_child_003", "score": 0.87}],
}
```

### 3.4 通义千问

```python
from infrastructure import QwenClient

client = QwenClient()
messages = [
    {"role": "system", "content": "你是教师的智能答疑助手……"},
    {"role": "user", "content": "什么是Transformer？"},
]

# 流式（SSE 打字机效果，技术文档 3.4.1）
async for chunk in client.stream_chat(messages):
    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

# 非流式（即时回答模式）
resp = await client.chat(messages)
print(resp.content, resp.total_tokens)
await client.close()
```

### 3.5 文本向量化

```python
from infrastructure import EmbeddingClient

client = EmbeddingClient()
vec = await client.aencode("什么是Transformer？")           # 单条，1536 维
vecs = await client.aencode_batch(["人工智能", "Java"])     # 批量，自动分批 + 保序

# 同步版（供同步的 Milvus 链路使用）
vec = client.encode("什么是Transformer？")
await client.aclose()      # 异步链路用 aclose()，不是 close()
```

---

## 四、必须注意的坑（均已在本代码中处理）

| # | 问题 | 现象 | 处理方式 |
|---|------|------|----------|
| 1 | **aiomysql 参数名** | `connect() got an unexpected keyword argument 'pool_size'` | aiomysql 用 `minsize`/`maxsize`，不是 SQLAlchemy 的 `pool_size`/`max_overflow`；已做等价映射 |
| 2 | **Redis RESP3 握手** | `unknown command 'HELLO'` | redis-py 8 默认 RESP3，需 Redis 6+；已显式 `protocol=2` 兼容老版本 |
| 3 | **多字段 HSET** | `wrong number of arguments for 'hset' command` | 多字段 HSET 是 Redis 4.0+ 特性；已改为 pipeline 逐字段写入 |
| 4 | **思考模型首字延迟** | 流式输出 8 秒无任何内容 | qwen3 思考模型先输出 reasoning_content；已默认 `enable_thinking=false`，首字从 ~8s 降到 ~0.7s |
| 5 | **Milvus VARCHAR 长度单位** | `string exceeds max length` | `max_length` 单位是**字节**：中文 1 字符≈3 字节，300 字子块约 900 字节，故取 2000 |
| 6 | **连接池取自增 ID** | 新记录 ID 恒为 0 | `LAST_INSERT_ID()` 是连接级状态，必须与 INSERT 同连接；已改用 `cursor.lastrowid` |
| 7 | **httpx 超时参数** | `Timeout must either include a default, or set all four parameters` | 不设默认值时四个超时字段必须全部显式指定 |
| 8 | **Embedding 返回乱序** | 文本与向量错位，检索结果全乱 | 已按响应中的 `index` 强制排序后再使用 |
| 9 | **向量维度不一致** | 写入/检索报维度错误，信息晦涩 | 首次调用后校验实际维度，与 `[embedding].dimensions` 不符时给出明确提示 |
| 10 | **跨学科缓存串答** | 同一问题在不同学科返回相同答案 | 缓存 key 的哈希包含 `subject_code` |

---

## 五、环境适配说明（本机实测）

以下三项与 `config.ini` 原基线不一致，已在配置注释中标注：

| 项 | 原基线 | 本机实际 | 处理 |
|----|--------|----------|------|
| Redis 密码 | 原配置填了密码 | 本机服务端**未设置密码**，填了会报 `no password is set` | 由 `.env` 的 `EDU_REDIS_PASSWORD` 提供；无密码则留空 |
| Redis 版本 | 要求 7.0+ | 本机为 **3.0.504**（Windows 移植版） | 已通过 `protocol=2` + 逐字段 HSET 兼容；生产建议按文档用 `redis:7-alpine` |
| Embedding 模型 | 未指定 | 集合维度为 1536 | 默认 `text-embedding-v4`（支持 1536 维）；`text-embedding-v3` 最高仅 1024 维，请勿混用 |

> ⚠️ **换 Embedding 模型等于换向量空间**，旧向量与新查询向量不可比，必须全量重建 Milvus 索引。

---

## 六、已知设计缺口

**Milvus schema 未存父块原文。** 技术文档 2.2.2 定义的字段只有 `content`（子块原文），
没有父块内容字段。因此 `MilvusClient._map_to_parent_chunks` 归并出的「父块内容」
实际是该父块下**命中子块的原文**，而非完整的 1200 字父块，喂给 LLM 的上下文偏短。

两种修复方向：

1. schema 增加 `parent_content` 字段，写入子块时冗余存父块原文（改动小，推荐）；
2. 父块原文存 MySQL，检索后按 `parent_chunk_id` 回表取全文（存储更规范）。

`MilvusClient` 已预留 `output_fields` 参数，改造时调用方无需变更。

---

## 七、生产部署前待办

| # | 事项 | 原因 |
|---|------|------|
| 1 | 密码改由环境变量注入 | `config.ini` 明文密码违反技术文档 8.1.1 |
| 2 | `pymilvus` 锁定 `<3.1` | `connections`/`utility` 这套 ORM API 将在 3.1 移除（已在 requirements.txt 中锁定） |
| 3 | 补齐 `core/rag` 业务层 | 本仓库仅含基础设施客户端，RAG 编排、Prompt、SSE 路由属业务层 |
| 4 | 接入限流中间件 | `redis_client.incr_rate` 已就绪，待接到 API 层 |
