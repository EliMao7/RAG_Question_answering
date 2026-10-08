"""环境自检：验证 3.11codeRAG 是否满足技术文档的技术栈要求。

用法：
    conda activate 3.11codeRAG
    python check_env.py
"""
import subprocess
import sys

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (模块名, 导入名, 技术文档章节)
MODULES = [
    ("fastapi",            "fastapi",             "2.3 Web 框架"),
    ("uvicorn",            "uvicorn",             "2.3 ASGI 服务器"),
    ("pydantic",           "pydantic",            "6.1 请求校验"),
    ("pydantic-settings",  "pydantic_settings",   "7.2 配置管理"),
    ("pymilvus",           "pymilvus",            "4.3 向量数据库"),
    ("openai",             "openai",              "3.7 LLM 客户端"),
    ("httpx",              "httpx",               "8.3 异步 HTTP 客户端"),
    ("sqlalchemy",         "sqlalchemy",          "4.2 MySQL ORM"),
    ("aiomysql",           "aiomysql",            "8.4 异步驱动"),
    ("pymysql",            "pymysql",             "11.3 建表脚本"),
    ("redis",              "redis",               "4.4 缓存"),
    ("pymupdf",            "fitz",                "3.3.1 PDF 解析"),
    ("python-docx",        "docx",                "3.3.1 DOCX 解析"),
    ("alembic",            "alembic",             "4.2 数据库迁移"),
    ("tenacity",           "tenacity",            "8.3 重试退避"),
    ("python-dotenv",      "dotenv",              "7.2 环境变量"),
    ("pytest",             "pytest",              "12.1 单元测试"),
]

# locust 依赖 gevent，导入时会 monkey-patch ssl。若在同一进程内与其他
# 已导入 ssl 的库（redis / anyio / httpx）共存，会触发 RecursionError。
# 因此单独放到子进程中检查，还原真实使用方式。
ISOLATED_MODULES = [
    ("locust",             "12.3 性能压测"),
]


def _check_isolated(dist_name: str) -> tuple[bool, str]:
    """在干净的子进程中导入模块，避免 monkey-patch 相互干扰。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {dist_name} as m; print(m.__version__)"],
            capture_output=True, text=True, timeout=60, encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return False, "导入超时"
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    last_line = (proc.stderr or "").strip().splitlines()
    return False, last_line[-1] if last_line else "未知错误"


def main() -> int:
    print(f"Python : {sys.version.split()[0]}")
    print(f"解释器 : {sys.executable}")
    print("-" * 66)

    ok, failed = [], []
    for dist_name, import_name, doc_ref in MODULES:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            ok.append((dist_name, version, doc_ref))
            print(f"  [OK]   {dist_name:<18} {version:<12} {doc_ref}")
        except Exception as exc:
            failed.append((dist_name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {dist_name:<18} {'':<12} {exc}")

    for dist_name, doc_ref in ISOLATED_MODULES:
        is_ok, detail = _check_isolated(dist_name)
        if is_ok:
            ok.append((dist_name, detail, doc_ref))
            print(f"  [OK]   {dist_name:<18} {detail:<12} {doc_ref}")
        else:
            failed.append((dist_name, detail))
            print(f"  [FAIL] {dist_name:<18} {'':<12} {detail}")

    total = len(MODULES) + len(ISOLATED_MODULES)
    print("-" * 66)
    print(f"通过 {len(ok)}/{total}，失败 {len(failed)}")

    if failed:
        print("\n缺失依赖请执行：")
        print("  pip install -r requirements.txt "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1
    print("\n环境就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
