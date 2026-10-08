"""知识库构建：把文档数据写入 Milvus。

完整链路（对应技术设计文档 2.1「知识库构建流程」）::

    原始文档(PDF/DOCX)
        ↓ ① 解析与清洗        core/rag/document_loader.py
    纯文本 + 来源信息
        ↓ ② 父子分块          core/rag/chunker.py   (parent=1200/child=300/overlap=50)
    父块 + 子块（含 parent_chunk_id 映射）
        ↓ ③ 在线向量化        infrastructure/embedding（DashScope text-embedding）
    子块向量（1536 维）
        ↓ ④ 写入 Milvus       infrastructure/vector_db
    可检索的向量索引

用法::

    conda activate 3.11codeRAG
    set DASHSCOPE_API_KEY=sk-xxxx

    # 冒烟测试：只处理前 3 个子块，确认链路通再全量跑
    python scripts/build_index.py --data-dir "<data dir>" --subject ai --limit 3

    # 全量构建（重建集合）
    python scripts/build_index.py --data-dir "<data dir>" --subject ai --recreate

    # 追加导入到已有集合
    python scripts/build_index.py --data-dir "<data dir>" --subject ai --append

说明：本脚本对 Milvus 是**写操作**。``--recreate`` 会删除同名集合及其全部向量数据，
需二次确认（或用 ``--yes`` 跳过）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# 允许以 `python scripts/build_index.py` 方式直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BASE_DIR, get_config  # noqa: E402
from core.rag.chunker import chunk_documents  # noqa: E402
from core.rag.document_loader import load_directory  # noqa: E402
from infrastructure.embedding.embedding_client import EmbeddingClient  # noqa: E402
from infrastructure.vector_db.milvus_client import (  # noqa: E402
    MilvusClient,
    VECTOR_DIM,
    ensure_database,
)
from utils.logger import get_logger, setup_logger  # noqa: E402

logger = get_logger(__name__)

# 每批写入 Milvus 的子块数（与嵌入批次独立：嵌入按 25 条一组，写入可更大）
INSERT_BATCH_SIZE = 100


def print_step(index: int, title: str) -> None:
    print(f"\n[{index}] {title}")
    print("-" * 70)


async def build(args: argparse.Namespace) -> int:
    cfg = get_config()
    subject_code = args.subject

    # ---- 0. 参数与配置校验 ----
    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("知识库构建 —— 文档 → 分块 → 向量化 → Milvus")
    print("=" * 70)
    print(f"  数据目录   : {data_dir}")
    print(f"  学科代码   : {subject_code}")
    print(f"  Milvus     : {cfg.get('milvus', 'host')}:{cfg.get('milvus', 'port')}"
          f"/{cfg.get('milvus', 'database_name')}.{cfg.get('milvus', 'collection_name')}")
    print(f"  嵌入模型   : {cfg.get('embedding', 'model')}"
          f"（{cfg.get('embedding', 'dimensions')} 维，在线调用）")
    print(f"  分块参数   : 父块 {cfg.get('retrieval', 'parent_chunk_size')} / "
          f"子块 {cfg.get('retrieval', 'child_chunk_size')} / "
          f"重叠 {cfg.get('retrieval', 'chunk_overlap')}")
    print(f"  写入模式   : {'重建集合（清空旧数据）' if args.recreate else '追加导入'}")

    if not data_dir.is_dir():
        print(f"\n数据目录不存在：{data_dir}")
        return 1

    # 学科白名单校验：避免把数据写到 config 未登记的学科下
    valid_sources = cfg.get_json("app", "valid_sources") or []
    if subject_code not in valid_sources:
        print(f"\n学科代码 {subject_code!r} 不在 [app].valid_sources 白名单中：{valid_sources}")
        return 1

    # ---- 1. 解析文档 ----
    print_step(1, "解析文档")
    documents = load_directory(data_dir)
    documents = [(p.name, text) for p, text in documents]
    for name, text in documents:
        print(f"  {name}: {len(text)} 字符")

    # ---- 2. 父子分块 ----
    print_step(2, "父子分块")
    chunks = chunk_documents(documents, subject_code, cfg)
    if not chunks:
        print("  未产生任何知识块，终止。")
        return 1

    parent_ids = {c.parent_chunk_id for c in chunks}
    print(f"  父块 {len(parent_ids)} 个 / 子块 {len(chunks)} 个")
    print(f"  子块平均长度：{sum(len(c.content) for c in chunks) // len(chunks)} 字符")
    if args.limit:
        chunks = chunks[: args.limit]
        print(f"  ⚠️ --limit 生效，仅处理前 {len(chunks)} 个子块（冒烟测试模式）")

    # ---- 3. 在线向量化 ----
    print_step(3, "在线向量化（DashScope）")
    embedder = EmbeddingClient(cfg)
    print(f"  模型 {embedder.model}，批大小 {embedder.batch_size}")

    started = time.perf_counter()
    try:
        vectors = await embedder.aencode_batch([c.content for c in chunks])
    finally:
        await embedder.aclose()
    embed_cost = time.perf_counter() - started

    if len(vectors) != len(chunks):
        print(f"  向量数量与子块数量不符：{len(vectors)} vs {len(chunks)}，终止。")
        return 1

    dim = len(vectors[0])
    print(f"  完成 {len(vectors)} 条，维度 {dim}，耗时 {embed_cost:.1f}s"
          f"（{embed_cost / len(vectors) * 1000:.0f} ms/条）")
    if dim != VECTOR_DIM:
        print(f"  维度与集合定义不符：{dim} != {VECTOR_DIM}，终止。")
        return 1

    # ---- 4. 写入 Milvus ----
    print_step(4, "写入 Milvus")

    # 4.1 确保数据库存在（设计文档指定 itcast，Milvus 默认只有 default 库）
    created_db = ensure_database(
        cfg.get("milvus", "host"), cfg.get("milvus", "port"),
        cfg.get("milvus", "database_name"),
    )
    print(f"  数据库 {cfg.get('milvus', 'database_name')}: "
          f"{'已创建' if created_db else '已存在'}")

    # 4.2 重建模式下删除集合属于不可逆操作，需确认
    if args.recreate and not args.yes:
        answer = input(
            f"  ⚠️ 即将删除集合 {cfg.get('milvus', 'collection_name')} 的全部向量数据，"
            f"确认？(yes/no): "
        ).strip().lower()
        if answer not in {"yes", "y"}:
            print("  已取消。")
            return 1

    milvus = MilvusClient(cfg)
    try:
        # 集合可能已存在：先建连再判断，避免 connect() 因集合缺失而抛错
        from pymilvus import connections, utility

        connections.connect(
            alias=milvus.alias,
            host=milvus.host,
            port=str(milvus.port),
            db_name=milvus.database_name,
        )
        existed = utility.has_collection(milvus.collection_name, using=milvus.alias)

        if existed and not args.append and not args.recreate:
            print(f"\n  集合 {milvus.collection_name} 已存在。")
            print("  请选择：--append 追加导入 / --recreate 删除重建")
            return 1

        if args.recreate and existed:
            utility.drop_collection(milvus.collection_name, using=milvus.alias)
            print(f"  已删除旧集合 {milvus.collection_name}")
            existed = False
            milvus._collection = None      # 让客户端重新加载集合

        if not existed:
            milvus.create_collection()
            print(f"  集合 {milvus.collection_name} 已创建："
                  f"{[f.name for f in milvus.collection.schema.fields]}")
        else:
            print(f"  追加导入到已有集合 {milvus.collection_name}")

        # 4.3 分批写入：一次性插入全部数据会占用大量内存，且失败后难以定位断点
        written = 0
        for start in range(0, len(chunks), INSERT_BATCH_SIZE):
            batch = chunks[start:start + INSERT_BATCH_SIZE]
            payload = [
                {
                    "chunk_id": c.chunk_id,
                    "parent_chunk_id": c.parent_chunk_id,
                    "subject_code": c.subject_code,
                    "content": c.content,
                    "parent_content": c.parent_content,
                    "source": c.source,
                    "page_no": c.page_no,
                    "embedding": vectors[start + offset],
                }
                for offset, c in enumerate(batch)
            ]
            written += milvus.insert_chunks(payload)
            print(f"  写入进度：{written}/{len(chunks)}")

        # flush 让数据落盘并对检索可见（不 flush 时 num_entities 可能仍是旧值）
        milvus.flush()
        total = milvus.num_entities
        print(f"  写入完成，集合当前实体总数：{total}")

        # ---- 5. 写清单，便于日后核对数据来源 ----
        manifest = {
            "subject_code": subject_code,
            "data_dir": str(data_dir),
            "embedding_model": embedder.model,
            "vector_dim": dim,
            "chunk_params": {
                "parent_chunk_size": cfg.get_int("retrieval", "parent_chunk_size"),
                "child_chunk_size": cfg.get_int("retrieval", "child_chunk_size"),
                "chunk_overlap": cfg.get_int("retrieval", "chunk_overlap"),
            },
            "collection": cfg.get("milvus", "collection_name"),
            "database": cfg.get("milvus", "database_name"),
            "parents": len(parent_ids),
            "children_written": written,
            "sources": [
                {"key": f"src{index:02d}", "file": name, "chars": len(text)}
                for index, (name, text) in enumerate(documents, start=1)
            ],
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path = BASE_DIR / "index_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  数据清单已写入：{manifest_path}")

    finally:
        milvus.close()

    # ---- 6. 检索验证 ----
    print_step(5, "检索验证")
    ok = await verify(cfg, embedder_model=manifest["embedding_model"])
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 70)
    print("构建完成" if ok else "构建完成，但检索验证未通过")
    print("=" * 70)
    print(f"  子块写入 : {written} 条")
    print(f"  总耗时   : {elapsed:.1f}s")
    return 0 if ok else 1


async def verify(cfg, embedder_model: str, top_n: int = 3) -> bool:
    """用几个真实问题检索，确认向量确实能被搜到。"""
    embedder = EmbeddingClient(cfg)
    milvus = MilvusClient(cfg)

    questions = [
        "什么是Transformer？",
        "自注意力机制是怎么工作的？",
        "课程大纲包含哪些阶段？",
    ]

    try:
        milvus.connect()
        for question in questions:
            vector = await embedder.aencode(question)
            results = milvus.retrieve(vector, subject_code="ai", top_k=top_n)

            if not results:
                print(f"  [空结果] {question}")
                continue

            top = results[0]
            preview = top["content"].replace("\n", " ")[:60]
            print(f"  {question}")
            print(f"    → 相似度 {top['score']:.4f} | 父块 {top['parent_chunk_id']}")
            print(f"    → 上下文长度 {len(top['content'])} 字符 | {preview}…")

        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  检索验证失败：{exc}")
        return False
    finally:
        await embedder.aclose()
        milvus.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把文档数据构建进 Milvus 知识库（在线嵌入模型）"
    )
    parser.add_argument("--data-dir", required=True, help="原始文档目录（PDF/DOCX/MD/TXT）")
    parser.add_argument("--subject", required=True, help="学科代码，须在 valid_sources 白名单内")
    parser.add_argument("--recreate", action="store_true",
                        help="删除同名集合并重建（会清空全部向量数据）")
    parser.add_argument("--append", action="store_true",
                        help="追加导入到已存在的集合")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 个子块（冒烟测试用，0 表示不限制）")
    parser.add_argument("--yes", action="store_true", help="跳过删除操作的二次确认")
    args = parser.parse_args()

    if args.recreate and args.append:
        print("--recreate 与 --append 互斥，请只选其一。")
        return 1

    setup_logger()
    return asyncio.run(build(args))


if __name__ == "__main__":
    sys.exit(main())
