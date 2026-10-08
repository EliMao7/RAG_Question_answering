"""基础设施层：外部依赖客户端（数据库 / 缓存 / 向量库 / 模型）。

对应技术设计文档 9.1 节的 ``infrastructure/`` 目录规划：
    db/           MySQL 连接与查询
    cache/        Redis 缓存
    vector_db/    Milvus 向量数据库
    llm/          通义千问模型调用
    embedding/    文本向量化

上层业务（core/rag）只依赖本层暴露的客户端，不直接接触驱动（pymilvus /
aiomysql / redis），便于替换实现与单元测试打桩。
"""
from infrastructure.cache.redis_client import RedisClient
from infrastructure.db.mysql_client import MySQLClient
from infrastructure.embedding.embedding_client import EmbeddingClient
from infrastructure.llm.qwen_client import LLMResponse, QwenClient
from infrastructure.vector_db.milvus_client import MilvusClient

__all__ = [
    "MySQLClient",
    "RedisClient",
    "MilvusClient",
    "QwenClient",
    "LLMResponse",
    "EmbeddingClient",
]
