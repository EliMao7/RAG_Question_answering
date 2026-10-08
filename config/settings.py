"""配置加载器。

对应技术设计文档第 11.1 节（配置项说明）与 8.1.1 节（API 密钥保护）。

设计要点：
1. config.ini 作为配置基线，**环境变量优先级更高**，便于容器化部署时切换环境
   （同一份配置文件可在 dev / test / prod 复用，避免多份 ini 漂移）；
2. LLM 密钥 ``DASHSCOPE_API_KEY`` 只从环境变量读取，绝不写入配置文件；
3. 启动时做校验（fail-fast），把配置错误暴露在启动阶段而非首次请求时。

环境变量覆盖规则：``EDU_{SECTION}_{OPTION}``，全大写，例如
    EDU_MYSQL_HOST=mysql-prod.internal    → 覆盖 [mysql].host
    EDU_RETRIEVAL_RETRIEVAL_K=20         → 覆盖 [retrieval].retrieval_k
"""
from __future__ import annotations

import configparser
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# 项目根目录：本文件位于 <root>/config/settings.py，因此上溯两层
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载项目根目录的 .env（若存在）。
# 真实密码 / 密钥放在 .env 里，.env 已被 .gitignore 排除，不会进仓库；
# config.ini 只保留占位符。override=False 表示「已存在的真实环境变量优先」，
# 便于 CI / 容器注入覆盖 .env。
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:  # pragma: no cover - 未装 python-dotenv 时降级为纯环境变量
    pass

# 默认配置文件位置
DEFAULT_INI_PATH = BASE_DIR / "config" / "config.ini"

# LLM 密钥的环境变量名（不落配置文件）
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"


class ConfigError(RuntimeError):
    """配置缺失或非法时抛出，用于启动阶段快速失败。"""


class Config:
    """config.ini 的只读封装，支持环境变量覆盖。

    用法::

        cfg = Config()                       # 自动加载 config/config.ini
        cfg.get("mysql", "host")             # → "localhost"
        cfg.get_int("retrieval", "retrieval_k")   # → 10
        cfg.get_json("app", "valid_sources")      # → ["ai", "java", ...]
    """

    def __init__(self, ini_path: str | Path | None = None) -> None:
        self.ini_path = Path(ini_path) if ini_path else DEFAULT_INI_PATH
        if not self.ini_path.exists():
            raise ConfigError(f"配置文件不存在：{self.ini_path}")

        parser = configparser.ConfigParser()
        # 显式指定编码，避免 Windows 下默认 GBK 读取中文注释报错
        parser.read(self.ini_path, encoding="utf-8")
        self._parser = parser

        self._validate()

    # ------------------------------------------------------------------
    # 取值方法
    # ------------------------------------------------------------------
    def get(self, section: str, option: str, default: Any = None) -> Any:
        """取字符串值；环境变量 ``EDU_{SECTION}_{OPTION}`` 存在时优先生效。"""
        env_key = f"EDU_{section.upper()}_{option.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val
        return self._parser.get(section, option, fallback=default)

    def get_int(self, section: str, option: str, default: int = 0) -> int:
        value = self.get(section, option, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"[{section}].{option} 需要整数，实际为 {value!r}") from exc

    def get_float(self, section: str, option: str, default: float = 0.0) -> float:
        value = self.get(section, option, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"[{section}].{option} 需要浮点数，实际为 {value!r}") from exc

    def get_bool(self, section: str, option: str, default: bool = False) -> bool:
        value = self.get(section, option, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def get_json(self, section: str, option: str, default: Any = None) -> Any:
        """取 JSON 值，用于 ``valid_sources = ["ai", "java"]`` 这类数组配置。"""
        raw = self.get(section, option, None)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"[{section}].{option} 不是合法 JSON：{raw!r}") from exc

    @property
    def dashscope_api_key(self) -> str:
        """通义千问 API Key，只从环境变量读取（技术文档 8.1.1）。"""
        key = os.getenv(DASHSCOPE_API_KEY_ENV)
        if not key:
            raise ConfigError(
                f"环境变量 {DASHSCOPE_API_KEY_ENV} 未设置。\n"
                f"Windows : set {DASHSCOPE_API_KEY_ENV}=sk-xxxx\n"
                f"Linux   : export {DASHSCOPE_API_KEY_ENV}=sk-xxxx"
            )
        return key

    def section_to_dict(self, section: str) -> dict[str, str]:
        """整段导出为 dict，便于传给客户端构造函数。"""
        if not self._parser.has_section(section):
            raise ConfigError(f"配置缺少 [{section}] 段")
        return dict(self._parser.items(section))

    # ------------------------------------------------------------------
    # 启动校验
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """校验必填项与参数间的逻辑关系，配置错误在启动时即暴露。"""
        required = [
            ("mysql", "host"),
            ("redis", "host"),
            ("milvus", "host"),
            ("milvus", "collection_name"),
            ("llm", "model"),
            ("llm", "dashscope_base_url"),
            ("retrieval", "retrieval_k"),
            ("app", "valid_sources"),
        ]
        for section, option in required:
            if not self._parser.has_option(section, option):
                raise ConfigError(f"缺少必填配置：[{section}].{option}")

        # candidate_m 由 retrieval_k 的检索结果归并而来，不能大于 retrieval_k，
        # 否则父块数量会少于预期，且不报错 —— 属于隐蔽 bug，故在此拦截。
        retrieval_k = self.get_int("retrieval", "retrieval_k")
        candidate_m = self.get_int("retrieval", "candidate_m")
        if candidate_m > retrieval_k:
            raise ConfigError(
                f"[retrieval].candidate_m({candidate_m}) 不应大于 retrieval_k({retrieval_k})"
            )

        # 重叠长度必须小于分块长度，否则切块步长非正会导致死循环
        for chunk_option in ("parent_chunk_size", "child_chunk_size"):
            chunk_size = self.get_int("retrieval", chunk_option)
            overlap = self.get_int("retrieval", "chunk_overlap")
            if overlap >= chunk_size:
                raise ConfigError(
                    f"[retrieval].chunk_overlap({overlap}) 必须小于 {chunk_option}({chunk_size})"
                )

        # Embedding 维度必须与 Milvus 集合维度一致（技术文档 2.2.2 定义 dim=1536）
        dim = self.get_int("embedding", "dimensions")
        if dim <= 0:
            raise ConfigError(f"[embedding].dimensions 必须为正整数，实际为 {dim}")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """获取全局唯一配置实例（进程内缓存，避免重复读盘与重复校验）。"""
    return Config()
