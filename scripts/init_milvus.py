"""初始化 Milvus 集合与索引。

对应技术设计文档 2.2.2 节：集合 ``edurag_gz09``（库 ``itcast``），
字段 ``chunk_id / parent_chunk_id / subject_code / content / embedding(1536)``，
索引 IVF_FLAT + IP，nlist=1024。

用法::

    conda activate 3.11codeRAG
    python scripts/init_milvus.py                      # 集合不存在则创建
    python scripts/init_milvus.py --recreate           # 删除重建（会清空向量数据！）
    python scripts/init_milvus.py --recreate --yes     # 跳过二次确认

注意：本脚本只创建「空的集合与索引」，不导入任何知识数据。
向量数据需由建库流程（文档解析 → 父子分块 → 向量化 → 写入）填充。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许以 `python scripts/init_milvus.py` 方式直接运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_config  # noqa: E402
from infrastructure.vector_db.milvus_client import MilvusClient, VECTOR_DIM  # noqa: E402
from utils.logger import setup_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 Milvus 集合与索引")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除已存在的同名集合并重建（危险：会丢失全部向量数据）",
    )
    parser.add_argument("--yes", action="store_true", help="跳过删除操作的二次确认")
    args = parser.parse_args()

    setup_logger()
    cfg = get_config()

    print("=" * 70)
    print("Milvus 集合初始化")
    print("=" * 70)
    print(f"  服务地址 : {cfg.get('milvus', 'host')}:{cfg.get('milvus', 'port')}")
    print(f"  数据库   : {cfg.get('milvus', 'database_name')}")
    print(f"  集合名称 : {cfg.get('milvus', 'collection_name')}")
    print(f"  向量维度 : {VECTOR_DIM}（须与 [embedding].dimensions 一致）")
    print()

    # 删除是不可逆操作，必须显式确认
    if args.recreate and not args.yes:
        answer = input(
            f"⚠️  即将删除集合 {cfg.get('milvus', 'collection_name')} 及其全部向量数据，"
            f"确认继续？(yes/no): "
        ).strip().lower()
        if answer not in {"yes", "y"}:
            print("已取消。")
            return 1

    client = MilvusClient()
    try:
        # 建连前先探活：集合不存在时 connect() 会抛错，这里需要先建立连接
        from pymilvus import connections, utility
        connections.connect(
            alias=client.alias,
            host=client.host,
            port=str(client.port),
            db_name=client.database_name,
        )

        existed = utility.has_collection(client.collection_name, using=client.alias)
        print(f"  当前状态 : {'集合已存在' if existed else '集合不存在'}")

        collection = client.create_collection(drop_existing=args.recreate)

        print("\n" + "=" * 70)
        print("初始化完成")
        print("=" * 70)
        print(f"  集合      : {client.collection_name}")
        print(f"  实体数量  : {collection.num_entities}")
        print(f"  字段      : {[f.name for f in collection.schema.fields]}")
        print("\n  下一步：导入知识数据（解析 → 分块 → 向量化 → insert_chunks → flush）")
        print("  数据为空时检索会返回空结果，属于正常现象。")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n初始化失败：{exc}")
        print("\n排查建议：")
        print("  1) 确认 Milvus 已启动：curl http://localhost:19530/healthz")
        print("  2) Milvus 依赖 etcd 与 MinIO，两者需先就绪（见技术文档 6.1）")
        print("  3) 确认 config.ini 的 [milvus] 配置与实际部署一致")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
