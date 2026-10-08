"""Milvus 向量数据库客户端。

对应技术设计文档：
- 2.2.2 节：集合 ``edurag_gz09``（库 ``itcast``），字段
  ``chunk_id / parent_chunk_id / subject_code / content / embedding(1536)``
- 3.2.1 节：检索实现（子块召回 → 父块归并）
- 3.2.2 节：学科过滤（``expr`` 表达式）
- 5.1.1 节：索引 IVF_FLAT，``metric_type=IP``，``nlist=1024``，``nprobe=10``

关于「父子分块」的说明：
    知识库按父子结构切分——**子块**（约 300 字）语义聚焦，用来做向量检索；
    **父块**（约 1200 字）上下文完整，用来喂给大模型。
    检索时先用子块命中，再通过 ``parent_chunk_id`` 归并回父块。

⚠️ 与设计文档 2.2.2 的差异（有意为之）：
    文档定义的字段仅有 ``chunk_id / parent_chunk_id / subject_code / content /
    embedding``，**没有存父块原文**。这会导致「子块检索、父块返回」的设计意图
    （文档 2.1.1）落空——检索回来的其实是 300 字子块，喂给 LLM 的上下文偏短。

    因此本实现在文档 5 字段基础上**增加 parent_content 字段**，写入子块时冗余
    存储其所属父块的完整原文（约 1200 字）。这样 :meth:`_map_to_parent_chunks`
    归并出的才是真正的父块，还原文档原本的设计意图。
    代价是存储冗余（每行多存一份父块文本），对本项目的数据量可忽略。
"""
from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config.settings import Config, get_config
from utils.exceptions import VectorDBError
from utils.logger import get_logger

logger = get_logger(__name__)

# 连接别名：同进程内多客户端共用同一连接时使用，避免重复建连
DEFAULT_ALIAS = "smartvoyage"

# 向量维度（技术文档 2.2.2 定义 dim=1536），与 [embedding].dimensions 必须一致
VECTOR_DIM = 1536

# VARCHAR 上限（单位：**字节**，非字符）。
# 中文 UTF-8 下一个汉字约 3 字节，故按「字符数 × 3 + 余量」估算：
#   content        子块 300 字  → ~900 字节  → 取 2048
#   parent_content 父块 1200 字 → ~3600 字节 → 取 8192
CONTENT_MAX_LEN = 2048
PARENT_CONTENT_MAX_LEN = 8192
SOURCE_MAX_LEN = 255          # 来源文件名（技术文档 4.3.1）

# 检索时返回的标量字段（技术文档 6.3.4 的 references 需要 source / page_no）
DEFAULT_OUTPUT_FIELDS = [
    "chunk_id", "parent_chunk_id", "content", "parent_content",
    "subject_code", "source", "page_no",
]


class MilvusClient:
    """Milvus 客户端：负责知识向量的检索、写入与集合管理。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or get_config()
        self.host = cfg.get("milvus", "host")
        self.port = cfg.get_int("milvus", "port", 19530)
        self.database_name = cfg.get("milvus", "database_name", "default")
        self.collection_name = cfg.get("milvus", "collection_name")
        self.alias = DEFAULT_ALIAS

        # 检索参数（技术文档 2.2.2 / 5.1.1）
        self.retrieval_k = cfg.get_int("retrieval", "retrieval_k", 10)
        self.nprobe = cfg.get_int("retrieval", "nprobe", 10)

        self._collection: Collection | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """建立 Milvus 连接并加载集合（幂等）。

        pymilvus 是同步库，本方法为同步调用；在 async 环境中请使用
        :meth:`aretrieve` 或在 ``asyncio.to_thread`` 中调用检索方法。
        """
        if self._collection is not None:
            return
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=str(self.port),
                db_name=self.database_name,
            )
            if not utility.has_collection(self.collection_name, using=self.alias):
                raise VectorDBError(
                    f"集合不存在：{self.collection_name}（库 {self.database_name}）。"
                    f"请先执行 scripts/init_milvus.py 创建集合并构建知识库索引。"
                )

            self._collection = Collection(self.collection_name, using=self.alias)
            # load() 把索引与向量载入内存，检索才不会触发磁盘 IO。
            # 技术文档 4.1.3 要求检索 < 100ms，依赖这一步常驻内存。
            self._collection.load()
            logger.info(
                "Milvus 已连接：%s:%s/%s，集合 %s（实体数 %s）",
                self.host, self.port, self.database_name,
                self.collection_name, self.num_entities,
            )
        except VectorDBError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorDBError("向量数据库连接失败", detail=str(exc), unavailable=True) from exc

    def close(self) -> None:
        """释放集合并断开连接。"""
        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
            connections.disconnect(self.alias)
            logger.info("Milvus 连接已关闭")
        except Exception as exc:  # noqa: BLE001 - 关闭失败无需阻断退出
            logger.warning("Milvus 关闭异常：%s", exc)

    def __enter__(self) -> "MilvusClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self.connect()
        assert self._collection is not None
        return self._collection

    @property
    def num_entities(self) -> int:
        """集合内实体数量（用于健康检查与学科覆盖度统计）。"""
        try:
            return int(self.collection.num_entities)
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # 检索（技术文档 3.2）
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: list[float],
        subject_code: str = "all",
        top_k: int | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """向量检索并归并为父块结果。

        :param query_embedding: 查询向量（1536 维），由 EmbeddingClient 生成
        :param subject_code: 学科代码；``all`` 表示全学科不过滤
        :param top_k: 召回子块数量，默认取配置 ``retrieval_k``（10）
        :param output_fields: 需要返回的标量字段
        :return: 父块列表，按相似度倒序，每项含
                 ``parent_chunk_id / content / subject_code / score / child_chunks``
        """
        k = top_k or self.retrieval_k
        fields = output_fields or DEFAULT_OUTPUT_FIELDS

        # 学科过滤：采用 Milvus 的「标量过滤 + 向量检索」混合查询，
        # 过滤在向量计算同批完成，比「先查 ID 再回表」的两段式更高效。
        expr = None
        if subject_code and subject_code != "all":
            # 注意：subject_code 来源于配置白名单校验后的值，
            # 不直接拼接用户原始输入，避免表达式注入。
            expr = f'subject_code == "{subject_code}"'

        search_params = {
            "metric_type": "IP",            # 内积相似度（技术文档 5.1.1）
            "params": {"nprobe": self.nprobe},  # 探测的聚类簇数，越大召回越高、越慢
        }

        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=fields,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorDBError("向量检索失败", detail=str(exc)) from exc

        hits = results[0] if results else []
        parents = self._map_to_parent_chunks(hits)
        logger.info(
            "向量检索完成：subject=%s, 召回子块=%s, 归并父块=%s",
            subject_code, len(hits), len(parents),
        )
        return parents

    async def aretrieve(
        self,
        query_embedding: list[float],
        subject_code: str = "all",
        top_k: int | None = None,
    ) -> list[dict]:
        """异步版检索：把同步的 pymilvus 调用放到线程池执行。

        pymilvus 没有异步 API，直接在协程里调用会**阻塞事件循环**，
        在 100 并发场景下会显著拖慢整体响应（技术文档 4.1.2）。
        ``asyncio.to_thread`` 将其移出事件循环，保持并发能力。
        """
        return await asyncio.to_thread(self.retrieve, query_embedding, subject_code, top_k)

    def _map_to_parent_chunks(self, hits: list[Any]) -> list[dict]:
        """子块命中结果 → 父块归并（技术文档 3.2.1）。

        同一个父块下的多个子块被命中时只保留一个父块条目，
        并以其中的最高相似度作为该父块的得分，避免重复上下文挤占 LLM 窗口。
        """
        parent_map: dict[str, dict] = {}

        for hit in hits:
            entity = hit.entity
            parent_id = entity.get("parent_chunk_id")
            score = self._hit_score(hit)

            # 优先取父块完整原文；集合若未建 parent_content 字段则退回子块原文
            parent_text = entity.get("parent_content") or entity.get("content")

            if parent_id not in parent_map:
                parent_map[parent_id] = {
                    "parent_chunk_id": parent_id,
                    "content": parent_text,          # 父块完整上下文，用于喂给 LLM
                    "hit_content": entity.get("content"),  # 实际命中的子块，便于排查
                    "subject_code": entity.get("subject_code"),
                    "source": entity.get("source"),          # 来源文件名（溯源用）
                    "page_no": entity.get("page_no"),        # 页码，-1 表示无
                    "score": score,
                    "child_chunks": [],
                }
            elif score > parent_map[parent_id]["score"]:
                # 同一父块被多个子块命中时，用最高分的那个子块代表该父块，
                # 保证排序与「最相关片段」一致
                parent_map[parent_id]["content"] = parent_text
                parent_map[parent_id]["hit_content"] = entity.get("content")
                parent_map[parent_id]["score"] = score

            parent_map[parent_id]["child_chunks"].append({
                "chunk_id": entity.get("chunk_id"),
                "score": score,
            })

        # 按相似度倒序：LLM 上下文按相关度排列，高分内容优先保留（超出窗口时截断尾部）
        return sorted(parent_map.values(), key=lambda item: item["score"], reverse=True)

    @staticmethod
    def _hit_score(hit: Any) -> float:
        """兼容不同 pymilvus 版本的分值字段名。

        2.3 之前用 ``distance``，2.4+ 统一为 ``score``，
        同时兼容两者可避免升级依赖时静默取到 0 分把排序打乱。
        """
        score = getattr(hit, "score", None)
        if score is None:
            score = getattr(hit, "distance", 0.0)
        return float(score or 0.0)

    # ------------------------------------------------------------------
    # 写入（构建知识库索引，技术文档 2.1.1）
    # ------------------------------------------------------------------
    def insert_chunks(self, chunks: list[dict]) -> int:
        """批量写入知识块。

        :param chunks: 每项须包含
            ``chunk_id / parent_chunk_id / subject_code / content / embedding``
        :return: 写入条数
        """
        if not chunks:
            return 0

        # 字段顺序必须与 schema 声明顺序一致，否则数据会错列
        chunk_ids = [c["chunk_id"] for c in chunks]
        parent_ids = [c["parent_chunk_id"] for c in chunks]
        subject_codes = [c["subject_code"] for c in chunks]
        contents = [c["content"] for c in chunks]
        parent_contents = [c.get("parent_content", c["content"]) for c in chunks]
        sources = [c.get("source", "") for c in chunks]
        page_nos = [int(c.get("page_no") or -1) for c in chunks]
        embeddings = [c["embedding"] for c in chunks]

        # 写入前做维度自检：维度不符时 Milvus 的报错信息不直观，这里提前拦截
        for i, vec in enumerate(embeddings):
            if len(vec) != VECTOR_DIM:
                raise VectorDBError(
                    f"向量维度不匹配：chunk_id={chunk_ids[i]} 为 {len(vec)} 维，"
                    f"集合要求 {VECTOR_DIM} 维。请检查 embedding 模型与配置。"
                )

        # 字节长度自检：Milvus 的 VARCHAR max_length 以字节计，
        # 中文超长会报 "string exceeds max length"，错误信息里不含字段名，排查困难。
        # 这里提前按 UTF-8 字节数拦截，报错直接指出是哪条、哪个字段、超了多少。
        self._check_byte_length(chunk_ids, "chunk_id", 64)
        self._check_byte_length(parent_ids, "parent_chunk_id", 64)
        self._check_byte_length(subject_codes, "subject_code", 20)
        self._check_byte_length(contents, "content", CONTENT_MAX_LEN)
        self._check_byte_length(parent_contents, "parent_content", PARENT_CONTENT_MAX_LEN)
        self._check_byte_length(sources, "source", SOURCE_MAX_LEN)

        try:
            result = self.collection.insert([
                chunk_ids, parent_ids, subject_codes,
                contents, parent_contents, sources, page_nos, embeddings,
            ])
            count = result.insert_count if hasattr(result, "insert_count") else len(chunks)
            logger.info("知识块写入完成：%s 条", count)
            return int(count)
        except VectorDBError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorDBError("知识块写入失败", detail=str(exc)) from exc

    @staticmethod
    def _check_byte_length(values: list[str], field: str, max_bytes: int) -> None:
        """按 UTF-8 字节数校验字段长度，提前给出可定位的报错。"""
        for value in values:
            length = len(value.encode("utf-8"))
            if length > max_bytes:
                raise VectorDBError(
                    f"字段 {field} 超出 Milvus VARCHAR 上限："
                    f"{length} 字节 > {max_bytes} 字节。\n"
                    f"  内容开头：{value[:50]!r}\n"
                    f"  注意 max_length 单位是字节，中文 1 字约 3 字节。"
                    f"请调小分块尺寸，或调大 schema 中该字段的 max_length。"
                )

    def flush(self) -> None:
        """落盘并刷新索引，写入完成后必须调用，否则数据可能查不到。"""
        self.collection.flush()
        logger.info("Milvus flush 完成")

    # ------------------------------------------------------------------
    # 集合管理（供 scripts/init_milvus.py 使用）
    # ------------------------------------------------------------------
    def create_collection(self, drop_existing: bool = False) -> Collection:
        """创建集合与索引（技术文档 2.2.2）。

        :param drop_existing: 为 True 时先删除同名集合（**会清空全部向量数据**）
        """
        if drop_existing and utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)
            logger.warning("已删除同名集合：%s", self.collection_name)

        if utility.has_collection(self.collection_name, using=self.alias):
            logger.info("集合已存在，跳过创建：%s", self.collection_name)
            return Collection(self.collection_name, using=self.alias)

        fields = [
            # 子块主键，形如 ai_src01_0003_child_002
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            # 父块 ID，用于子块→父块归并
            FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="subject_code", dtype=DataType.VARCHAR, max_length=20),
            # 子块原文（约 300 字），**参与向量化**
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=CONTENT_MAX_LEN),
            # 父块完整原文（约 1200 字），不参与向量化，仅用于检索后提供完整上下文。
            # max_length 单位是字节：1200 字中文约 3600 字节，取 8192 留足余量。
            FieldSchema(name="parent_content", dtype=DataType.VARCHAR,
                        max_length=PARENT_CONTENT_MAX_LEN),
            # 来源文件名与页码，用于接口返回溯源信息（技术文档 6.3.4）
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=SOURCE_MAX_LEN),
            FieldSchema(name="page_no", dtype=DataType.INT32,
                        description="PDF 页码；-1 表示无页码信息（如 DOCX）"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        ]
        schema = CollectionSchema(fields=fields, description="教师智能问答知识库")

        collection = Collection(
            name=self.collection_name, schema=schema, using=self.alias
        )

        # IVF_FLAT：倒排索引 + 原始向量，精度高、内存适中（技术文档 5.1.1）
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},   # 聚类中心数，经验值约 4×√N
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logger.info("集合已创建并建立索引：%s", self.collection_name)
        self._collection = collection
        return collection

    def get_subject_distribution(self) -> dict[str, int]:
        """统计各学科的知识块数量（技术文档 4.1.6 的 doc_count）。

        用 query 做分组统计；数据量大时可改为预聚合并缓存到 Redis。
        """
        try:
            rows = self.collection.query(
                expr='chunk_id != ""',       # 匹配全部实体
                output_fields=["subject_code"],
                limit=16384,                 # Milvus 单次查询上限
            )
            distribution: dict[str, int] = {}
            for row in rows:
                code = row.get("subject_code")
                distribution[code] = distribution.get(code, 0) + 1
            return distribution
        except Exception as exc:  # noqa: BLE001
            logger.warning("学科分布统计失败：%s", exc)
            return {}

    # ------------------------------------------------------------------
    # 健康检查（技术文档 4.1.5 / 7.2）
    # ------------------------------------------------------------------
    def health_check(self) -> str:
        """探测 Milvus 可用性，返回 ``connected`` / ``disconnected``。"""
        try:
            if self._collection is None:
                self.connect()
            # 轻量查询，验证连接与集合均可访问
            self.collection.query(expr='chunk_id != ""', output_fields=["chunk_id"], limit=1)
            return "connected"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 健康检查失败：%s", exc)
            return "disconnected"


def build_default_schema_fields() -> list[FieldSchema]:
    """导出字段定义，便于测试或脚本复用（避免两处 schema 漂移）。"""
    return [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="subject_code", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=CONTENT_MAX_LEN),
        FieldSchema(name="parent_content", dtype=DataType.VARCHAR,
                    max_length=PARENT_CONTENT_MAX_LEN),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=SOURCE_MAX_LEN),
        FieldSchema(name="page_no", dtype=DataType.INT32),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]


def ensure_database(host: str, port: int | str, db_name: str) -> bool:
    """确保 Milvus 数据库存在，不存在则创建。

    ⚠️ 必要性：技术文档指定 ``database_name = itcast``，但 Milvus 默认只有
    ``default`` 库。若库不存在，ORM 风格的 ``connections.connect(db_name=...)``
    不会报错（静默连到 default 库），后续 ``has_collection`` 检查的其实是
    default 库——于是出现「连接成功但集合永远找不到」的诡异现象。
    因此必须在建集合前显式建库。

    :return: True 表示本次新建，False 表示原本已存在
    """
    from pymilvus import MilvusClient as _MilvusClient

    client = _MilvusClient(uri=f"http://{host}:{port}")
    existing = client.list_databases()
    if db_name in existing:
        logger.info("Milvus 数据库已存在：%s", db_name)
        return False

    client.create_database(db_name=db_name)
    logger.info("Milvus 数据库已创建：%s", db_name)
    return True
