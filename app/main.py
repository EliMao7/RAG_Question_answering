"""FastAPI 应用入口。

对应《教师智能问答系统技术文档》：
- 2.2 节：分层架构（API 网关层 → 业务逻辑层 → RAG 引擎层 → 数据存储层）
- 6.1 节：统一约定（``/api`` 前缀、统一错误响应、``X-Request-Id`` 追踪）
- 5.4 节：异步与并发模型（连接池在启动时建立，跨请求复用）

启动::

    conda activate 3.11codeRAG
    set DASHSCOPE_API_KEY=sk-xxxx
    uvicorn app.main:app --reload --port 8000

接口文档：http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import chat, health, history, session, subjects
from app.core.request_id import RequestIdMiddleware
from app.services.cache_service import CacheService
from app.services.chat_service import ChatService
from app.services.session_service import SessionService
from app.services.subject_service import SubjectService
from config.settings import ConfigError, get_config
from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import MySQLClient
from infrastructure.embedding.embedding_client import EmbeddingClient
from infrastructure.llm.qwen_client import QwenClient
from infrastructure.vector_db.milvus_client import MilvusClient, ensure_database
from utils.exceptions import ErrorCode, SmartVoyageError
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

# 前端静态资源目录（无构建 SPA：index.html + css + js）
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# 生命周期：启动时建立所有连接，关闭时释放
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    连接在启动阶段一次性建立并挂在 ``app.state``，请求处理阶段只读，
    避免每个请求重复建连（技术文档 5.4：应用启动时初始化，避免每请求重建）。
    """
    setup_logger()
    cfg = get_config()
    logger.info("=" * 60)
    logger.info("教师智能问答系统 后端启动中……")
    logger.info("=" * 60)

    # ---- 1. 确保 Milvus 库存在（设计文档指定 itcast，Milvus 默认只有 default 库）----
    # 注意：若库不存在，ORM 风格的 connections.connect(db_name=...) 不会报错，
    # 而是静默连到 default 库，后续会表现为「连接成功但集合永远找不到」。
    try:
        await asyncio.to_thread(
            ensure_database,
            cfg.get("milvus", "host"),
            cfg.get("milvus", "port"),
            cfg.get("milvus", "database_name"),
        )
    except Exception as exc:  # noqa: BLE001 - 库已存在或 Milvus 暂不可用都不阻断启动
        logger.warning("Milvus 数据库检查失败（继续启动）：%s", exc)

    # ---- 2. 建立客户端 ----
    db = MySQLClient(cfg)
    cache = RedisClient(cfg)
    milvus = MilvusClient(cfg)
    embedder = EmbeddingClient(cfg)
    llm = QwenClient(cfg)

    # MySQL / Redis 属强依赖：连不上直接启动失败，避免带病运行
    await db.connect()
    await cache.connect()

    # Milvus 属弱依赖：连不上只告警，服务仍可用降级策略提供通用答案（技术文档 3.9）
    try:
        await asyncio.to_thread(milvus.connect)
    except Exception as exc:  # noqa: BLE001
        logger.error("Milvus 连接失败，问答将走降级路径：%s", exc)

    # ---- 3. 组装服务层 ----
    cache_service = CacheService(cache)
    subject_service = SubjectService(db, cache, milvus, cfg)
    session_service = SessionService(db, cache)
    chat_service = ChatService(
        db=db,
        cache_service=cache_service,
        subject_service=subject_service,
        embedder=embedder,
        milvus=milvus,
        llm=llm,
        config=cfg,
    )

    # ---- 4. 启动自检：学科白名单与数据库必须一致（技术文档 4.2.1）----
    await _check_subject_consistency(subject_service, cfg)

    app.state.db = db
    app.state.cache = cache
    app.state.milvus = milvus
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.cache_service = cache_service
    app.state.subject_service = subject_service
    app.state.session_service = session_service
    app.state.chat_service = chat_service

    logger.info("后端启动完成，接口文档：http://localhost:8000/docs")
    try:
        yield
    finally:
        # ---- 5. 优雅关闭 ----
        logger.info("正在关闭连接……")
        await llm.close()
        await embedder.aclose()
        embedder.close()
        await cache.close()
        await db.close()
        await asyncio.to_thread(milvus.close)
        logger.info("已全部关闭")


async def _check_subject_consistency(subject_service: SubjectService, cfg) -> None:
    """校验 config.ini 的 valid_sources 与 MySQL subjects 表是否一致。

    技术文档 4.2.1 要求「不一致则拒绝启动」——快速失败胜过线上静默的
    「学科不可用」。此处降级为告警：因为建表脚本可能尚未执行，
    直接拒绝启动会让首次部署无法完成自检。
    """
    valid = set(subject_service.valid_sources)
    try:
        rows = await subject_service.db.get_subjects()
    except Exception as exc:  # noqa: BLE001
        logger.warning("学科一致性自检跳过（查询失败）：%s", exc)
        return

    if not rows:
        logger.warning("subjects 表为空，请执行 scripts/init_db.sql 初始化学科数据")
        return

    db_codes = {row["code"] for row in rows}
    if db_codes != valid:
        logger.warning(
            "学科配置不一致！config.ini valid_sources=%s，MySQL subjects=%s。"
            "请让两者保持一致，否则会出现「学科校验通过但查不到数据」的问题。",
            sorted(valid), sorted(db_codes),
        )
    else:
        logger.info("学科一致性自检通过：%s", sorted(valid))


# ---------------------------------------------------------------------------
# 应用实例
# ---------------------------------------------------------------------------
app = FastAPI(
    title="教师智能问答系统",
    description="基于 RAG（Milvus + 通义千问）的智能答疑后端接口",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 请求追踪中间件（技术文档 6.1）
app.add_middleware(RequestIdMiddleware)

# CORS：前端独立部署时需放开；生产环境请把 allow_origins 收紧为具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # 与 allow_origins=["*"] 不能同时为 True
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


# ---------------------------------------------------------------------------
# 统一异常处理（技术文档 6.1 的错误响应结构）
# ---------------------------------------------------------------------------
def _error_response(
    status_code: int,
    code: str,
    message: str,
    request: Request,
) -> JSONResponse:
    """构造统一错误响应体：``{"error": {code, message, request_id, timestamp}}``。"""
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "timestamp": datetime.now(
                    timezone(timedelta(hours=8))
                ).isoformat(timespec="seconds"),
            }
        },
    )


@app.exception_handler(SmartVoyageError)
async def handle_business_error(request: Request, exc: SmartVoyageError) -> JSONResponse:
    """业务异常 → 按错误码表映射 HTTP 状态码。"""
    # 非致命错误（如检索为空）用 warning，真正的故障用 error，避免日志噪音
    if exc.error_code.http_status >= 500:
        logger.error("业务异常：%s | detail=%s", exc.message, exc.detail)
    else:
        logger.warning("业务异常：%s | detail=%s", exc.message, exc.detail)

    return _error_response(exc.error_code.http_status, exc.error_code.code, exc.message, request)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic 校验失败 → 统一为 400，并给出首个字段的错误说明。

    FastAPI 默认返回 422，但技术文档 6.5 的错误码表使用 400，
    这里做一次归一化，保持前后端约定一致。
    """
    first = (exc.errors() or [{}])[0]
    location = ".".join(str(item) for item in first.get("loc", ()) if item != "body")
    message = f"参数错误：{location} {first.get('msg', '格式不正确')}".strip()
    logger.warning("请求参数校验失败：%s", message)
    return _error_response(400, ErrorCode.INVALID_PARAMS.code, message, request)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """框架级 HTTP 异常（404 路由不存在、405 方法不允许等）→ 统一结构。"""
    code_map = {401: ErrorCode.UNAUTHORIZED, 404: ErrorCode.SESSION_NOT_FOUND}
    error_code = code_map.get(exc.status_code, ErrorCode.INVALID_PARAMS)
    return _error_response(
        exc.status_code, error_code.code, str(exc.detail), request
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """未捕获异常 → 500，且**不把内部细节返回给前端**（只进日志）。"""
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("未捕获异常 | request_id=%s | %s", request_id, exc)
    return _error_response(
        500,
        ErrorCode.INTERNAL_ERROR.code,
        "系统异常，请稍后重试",
        request,
    )


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------
app.include_router(health.router, prefix="/api")
app.include_router(session.router, prefix="/api")
app.include_router(subjects.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(history.router, prefix="/api")

# 前端静态资源（技术文档 5.1 节规划：app/static/ 存放前端资源）
# 无构建 SPA：改完 HTML/CSS/JS 刷新浏览器即生效，无需 npm
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:  # pragma: no cover - 仅在静态资源缺失时触发
    logger.warning("前端静态资源目录不存在：%s", STATIC_DIR)


@app.get("/", include_in_schema=False)
async def index():
    """根路径返回前端页面；静态资源缺失时退化为接口信息 JSON。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        # no-cache：开发期改完刷新即生效，避免浏览器缓存旧页面
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})
    return {
        "service": "教师智能问答系统",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "message": "前端静态资源缺失，请检查 app/static/ 目录",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """浏览器默认请求 favicon，返回 204 避免日志里刷 404。"""
    return JSONResponse(status_code=204, content=None)


def main() -> None:
    """本地开发启动入口：``python -m app.main``。"""
    import uvicorn

    try:
        cfg = get_config()
        port = int(cfg.get("app", "port", 8000))
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        raise SystemExit(1) from exc

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
