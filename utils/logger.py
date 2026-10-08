"""日志初始化。

严格对应技术设计文档第 7.1 节：
- 格式：``%(asctime)s - %(name)s - %(levelname)s - %(message)s``
- 文件处理器按大小轮转（默认 10MB × 10 份）
- logger 名称：``smartvoyage``

与原始配置的差异（必要修正）：原配置把 ``log_file`` 写成了 macOS 绝对路径
（形如 ``/Users/<用户名>/.../logs/app.log``），换到 Windows/Linux/容器内都会写失败。
这里改为相对路径，统一以项目根目录为基准拼接；同时保留控制台输出，
方便开发期直接观察，容器化时则由 stdout 收集。
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import BASE_DIR, Config, get_config

# 统一的 logger 名称，全项目复用同一实例（技术文档 7.1.1 中命名为 smartvoyage）
LOGGER_NAME = "smartvoyage"

# 日志格式（与文档 7.1.1 一致）
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_configured = False


def _resolve_log_path(log_file: str) -> Path:
    """把配置中的日志路径解析为绝对路径。

    相对路径按项目根目录解析；绝对路径原样使用。
    """
    path = Path(log_file)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def setup_logger(config: Config | None = None) -> logging.Logger:
    """初始化并返回全局 logger，重复调用不会重复添加 handler。"""
    global _configured

    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        return logger

    cfg = config or get_config()
    log_path = _resolve_log_path(cfg.get("logger", "log_file", "logs/app.log"))
    level = getattr(logging, str(cfg.get("logger", "log_level", "INFO")).upper(), logging.INFO)

    logger.setLevel(level)
    logger.propagate = False  # 避免向 root logger 冒泡导致重复打印
    formatter = logging.Formatter(LOG_FORMAT)

    # ---- 文件处理器：目录不存在时自动创建（首次启动常见问题）----
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=cfg.get_int("logger", "max_bytes", 10 * 1024 * 1024),
            backupCount=cfg.get_int("logger", "backup_count", 10),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # 日志目录不可写不应导致服务启动失败，降级为仅控制台输出
        logger.warning("日志文件初始化失败(%s)，已降级为仅控制台输出：%s", log_path, exc)

    # ---- 控制台处理器：开发期直接观察 ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _configured = True
    logger.info("日志系统初始化完成，日志文件：%s", log_path)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取子 logger。

    用法::

        logger = get_logger(__name__)
        logger.info("...")

    子 logger 继承 ``smartvoyage`` 的 handler，因此只需在程序入口调用一次
    :func:`setup_logger`。
    """
    if not _configured:
        setup_logger()
    if name:
        # 统一挂在 smartvoyage 命名空间下，避免依赖 root logger 的配置
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
