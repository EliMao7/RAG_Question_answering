"""父子分块（Parent-Child Chunking）。

对应技术设计文档 2.1.1 节，参数取自 config.ini ``[retrieval]``：

============================  =========  ==========================================
参数                           默认值     作用
============================  =========  ==========================================
``parent_chunk_size``         1200       父块大小：喂给 LLM 的上下文单元
``child_chunk_size``          300        子块大小：**参与向量化**的检索单元
``chunk_overlap``             50         块间重叠：防止答案正好落在切分边界
============================  =========  ==========================================

为什么要分父子（文档 2.1.1 的设计意图）：

======================  ============  ==============  ==================
分块方式                 检索精度      上下文完整性     结论
======================  ============  ==============  ==================
只用 1200 字大块          低（多主题稀释） 高              检索不准
只用 300 字小块           高            低（答案被截断）  生成不全
**父子分块**              **高**        **高**          两者兼得
======================  ============  ==============  ==================

切分时优先在自然边界（空行 → 换行 → 中文句号）断开，避免把句子劈成两半。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import Config, get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# 切分优先使用的自然边界，按优先级从高到低
# （先段落、再换行、再句末标点，最后才硬切）
SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ")

# document_loader 在 PDF 每页开头插入的页标记，形如 "[第3页]"
PAGE_MARKER = re.compile(r"\[第(\d+)页\]")


@dataclass
class Chunk:
    """一个子块及其所属父块。"""

    chunk_id: str                 # 子块唯一 ID（Milvus 主键）
    parent_chunk_id: str          # 父块 ID，用于检索后归并
    subject_code: str             # 学科代码
    content: str                  # 子块原文（参与向量化）
    parent_content: str           # 父块完整原文（提供上下文）
    source: str = ""              # 来源文件名（技术文档 4.3.1 的 source 字段）
    page_no: int = -1             # 页码；-1 表示无页码信息（如 DOCX）
    metadata: dict = field(default_factory=dict)


def _extract_page_no(text: str) -> int:
    """从文本中提取页码。

    document_loader 会给 PDF 每页插入 ``[第N页]`` 标记，这里取父块中
    出现的**第一个**标记作为该父块的页码——一个父块（约 1200 字）可能跨越
    两页，取起始页更符合「这段内容从哪页开始」的直觉。

    :return: 页码；找不到标记时返回 -1（DOCX/TXT 无页码）
    """
    match = PAGE_MARKER.search(text)
    return int(match.group(1)) if match else -1


def _split_text(
    text: str,
    size: int,
    overlap: int,
    prefer_boundary: bool = True,
) -> list[str]:
    """滑动窗口切分文本。

    :param prefer_boundary: 为 True 时在窗口尾部寻找自然边界断开，
                            避免把句子/段落切断
    :return: 切分后的文本块列表（已 strip，空块被丢弃）
    """
    if size <= overlap:
        raise ValueError(f"分块大小({size})必须大于重叠长度({overlap})")

    pieces: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)

        # 只在「不是最后一段」时才调整边界，否则会把结尾内容挤到下一块
        if prefer_boundary and end < length:
            # 在窗口末 25% 区间内寻找最靠后的自然边界
            search_from = start + int(size * 0.75)
            best_cut = -1
            for separator in SEPARATORS:
                index = text.rfind(separator, search_from, end)
                if index != -1:
                    candidate = index + len(separator)
                    if candidate > best_cut:
                        best_cut = candidate
            # 边界必须落在窗口内且确实前进，否则放弃调整
            if best_cut > start:
                end = best_cut

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= length:
            break
        # 回退 overlap 形成重叠；max(..., start+1) 保证游标一定前进，防止死循环
        start = max(end - overlap, start + 1)

    return pieces


def build_parent_child_chunks(
    text: str,
    source: str,
    subject_code: str,
    source_key: str,
    config: Config | None = None,
) -> list[Chunk]:
    """把一份文档的纯文本切成父子结构的知识块。

    :param text: 文档纯文本（由 document_loader 产出）
    :param source: 来源文件名，用于溯源展示
    :param subject_code: 学科代码（如 ``ai``）
    :param source_key: 来源短标识（如 ``src01``），用于拼 chunk_id。
                       刻意使用 ASCII，避免中文文件名把 VARCHAR(64) 字节数顶满。
    """
    cfg = config or get_config()
    parent_size = cfg.get_int("retrieval", "parent_chunk_size", 1200)
    child_size = cfg.get_int("retrieval", "child_chunk_size", 300)
    overlap = cfg.get_int("retrieval", "chunk_overlap", 50)

    # ---- 第一步：切父块 ----
    parent_texts = _split_text(text, parent_size, overlap)
    if not parent_texts:
        logger.warning("文本为空，未产生任何知识块：%s", source)
        return []

    chunks: list[Chunk] = []

    # ---- 第二步：每个父块内部再切子块，并记录父子映射 ----
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{subject_code}_{source_key}_{parent_index:04d}_parent"

        # 父块头部标注来源：schema 未单独存 source 字段，
        # 把来源写进父块正文，模型在回答时才能给出出处（技术文档 3.3.1 要求标注来源）。
        parent_content = f"【来源：{source}】\n{parent_text}"

        page_no = _extract_page_no(parent_text)
        child_texts = _split_text(parent_text, child_size, overlap)
        for child_index, child_text in enumerate(child_texts):
            chunks.append(Chunk(
                chunk_id=f"{parent_id}_child_{child_index:03d}",
                parent_chunk_id=parent_id,
                subject_code=subject_code,
                content=child_text,
                parent_content=parent_content,
                source=source,
                page_no=page_no,
                metadata={
                    "parent_index": parent_index,
                    "child_index": child_index,
                },
            ))

    logger.info(
        "分块完成：%s → 父块 %s 个 / 子块 %s 个",
        source, len(parent_texts), len(chunks),
    )
    return chunks


def chunk_documents(
    documents: list[tuple[str, str]],
    subject_code: str,
    config: Config | None = None,
) -> list[Chunk]:
    """批量分块。

    :param documents: ``[(来源文件名, 纯文本), ...]``
    """
    all_chunks: list[Chunk] = []
    for index, (source, text) in enumerate(documents, start=1):
        source_key = f"src{index:02d}"
        all_chunks.extend(
            build_parent_child_chunks(text, source, subject_code, source_key, config)
        )
    logger.info("全部文档分块完成：共 %s 个子块", len(all_chunks))
    return all_chunks
