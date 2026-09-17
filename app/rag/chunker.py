"""Markdown 文本切块（阶段 1，纯函数模块）。

职责边界
--------
输入字符串，输出 Chunk 列表。不读写文件、不联网、不访问向量库，
因此 tests/test_chunker.py 无需任何 fixture 即可完整覆盖。

切块策略
--------
1. 解析 YAML front-matter，取出 doc_id / version / effective_date / expire_date 等时效字段。
2. H1 是文档标题（已由 front-matter 的 doc_title 承载），从正文移除，避免重复进块。
3. 正文按 H2 切分为语义块，H2 标题写入 metadata["heading"]。
4. 单块超过 MAX_CHARS 时先按 H3 再切；仍超长则按段落滑窗切分，保留 OVERLAP 字符重叠。
5. chunk_id = f"{doc_id}_p{index}"，确定性生成 —— 重新索引时按该 id upsert 即可覆盖，
   不会产生重复块。

日期字段的双形态（本项目最重要的一个落地细节）
------------------------------------------------
ChromaDB 0.5.x 的 where 范围运算符（$gte / $lte）只接受 int / float，传 ISO 字符串会抛：

    ValueError: Expected operand value to be an int or a float for operator $gte,
                got 2026-09-15 in query.

因此每个日期都写入两份、语义完全等价：
    effective_date / expire_date   ISO 字符串 —— 供展示、引用与评估比对
    effective_ord  / expire_ord    YYYYMMDD 整数 —— 供 Chroma where 做范围过滤

YYYYMMDD 的整数序与日期序一致，所以 {"expire_ord": {"$gte": 20260916}} 等价于
「不早于今天过期」。所有过滤一律走 *_ord 字段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import yaml

# 切块参数属于算法超参，不是部署配置，故留在代码里而不入 .env
MAX_CHARS = 800
OVERLAP = 80

# 长期有效文档的 expire 哨兵值。必须写哨兵而非留空：
# 留空（空字符串或 0）会让"长期有效"的文档在 expire_ord >= today 过滤下被误杀。
NO_EXPIRE_DATE = "9999-12-31"
NO_EXPIRE_ORD = 99991231

# 文档首个 H2 之前的内容没有小节标题，统一归到这个名字下，
# 同时避免向 Chroma 写入空字符串 metadata。
NO_HEADING = "概述"

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_LOOSE_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


@dataclass(slots=True)
class Chunk:
    """一个可索引的文本块。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _to_iso(value: Any) -> str:
    """统一成 ISO 日期字符串。

    YAML 会把 `2026-07-01` 解析成 datetime.date，而 Chroma 的 metadata 只接受
    str / int / float / bool，直接写入会抛类型错误，必须在此归一。
    """
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value).strip()


def date_to_ord(value: str) -> int:
    """ISO 日期字符串 -> YYYYMMDD 整数；无法识别时返回 0。

    供 Chroma 的 where 范围过滤使用（该过滤器不支持字符串比较）。
    容错处理 `2026-9-30` 这类未补零写法；彻底非法的输入返回 0，
    对 expire_ord 而言 0 意味着「已过期」（被过滤），对 effective_ord 而言
    意味着「很早以前生效」（被保留），两个方向的兜底都是安全的。
    """
    text = (value or "").strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return int(digits)
    matched = _LOOSE_DATE_RE.match(text)
    if matched:
        year, month, day = matched.groups()
        return int(f"{year}{int(month):02d}{int(day):02d}")
    return 0


def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """拆出 front-matter 与正文。无 front-matter 时返回空字典与原文。"""
    matched = FRONT_MATTER_RE.match(raw)
    if not matched:
        return {}, raw
    try:
        loaded = yaml.safe_load(matched.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"front-matter YAML 解析失败：{exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("front-matter 必须是键值对映射")
    return loaded, raw[matched.end():]


def build_doc_metadata(
    meta: dict[str, Any], *, fallback_doc_id: str = ""
) -> dict[str, Any]:
    """校验并归一化文档级 metadata。缺必填字段直接报错，不允许脏数据入库。

    返回的字典同时包含 ISO 字符串与 YYYYMMDD 整数两套日期字段，理由见模块文档。
    """
    doc_id = str(meta.get("doc_id") or fallback_doc_id).strip()
    if not doc_id:
        raise ValueError("文档缺少 doc_id，且无法从文件名推导")

    for required in ("version", "effective_date"):
        if not meta.get(required):
            raise ValueError(f"{doc_id}: front-matter 缺少必填字段 {required}")

    effective_date = _to_iso(meta.get("effective_date"))
    expire_date = _to_iso(meta.get("expire_date")) or NO_EXPIRE_DATE

    return {
        "doc_id": doc_id,
        "doc_title": str(meta.get("doc_title") or doc_id).strip(),
        "version": str(meta.get("version")).strip(),
        "effective_date": effective_date,
        "expire_date": expire_date,
        "source": str(meta.get("source") or "local").strip(),
        "effective_ord": date_to_ord(effective_date),
        "expire_ord": date_to_ord(expire_date),
    }


def _blocks_at(text: str, level: int) -> list[tuple[str, str]]:
    """把 text 按第 level 级标题切成 [(标题, 正文)]。

    首个标题之前的内容归入 ("", 内容)。注意 `^#{2}\\s+` 不会误匹配 `###`，
    因为第三个字符要求是空白。
    """
    pattern = re.compile(rf"^#{{{level}}}\s+(.*)$")
    blocks: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    for line in text.splitlines():
        matched = pattern.match(line)
        if matched:
            if heading or lines:
                blocks.append((heading, "\n".join(lines).strip()))
            heading = matched.group(1).strip()
            lines = []
        else:
            lines.append(line)
    if heading or lines:
        blocks.append((heading, "\n".join(lines).strip()))
    return blocks


def _split_long(
    text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP
) -> list[str]:
    """按段落滑窗切分超长文本。

    不变量：返回的每一片长度均 <= max_chars。重叠片段只在「加上后仍不超限」时
    才附加，因此该不变量恒成立（这一点由 test_chunker.py 断言）。
    """
    if len(text) <= max_chars:
        return [text]
    if overlap >= max_chars:
        raise ValueError("overlap 必须小于 max_chars")

    pieces: list[str] = []
    buffer = ""
    for raw_para in re.split(r"\n\s*\n", text):
        para = raw_para.strip()
        if not para:
            continue
        if len(para) > max_chars:  # 单段本身超长 -> 硬切
            if buffer:
                pieces.append(buffer)
                buffer = ""
            step = max_chars - overlap
            pieces.extend(
                para[start:start + max_chars] for start in range(0, len(para), step)
            )
            continue

        if buffer and len(buffer) + len(para) + 2 > max_chars:
            pieces.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            # 附加重叠片段不得突破上限，否则宁可不加重叠
            if tail and len(tail) + len(para) + 2 <= max_chars:
                buffer = f"{tail}\n\n{para}"
            else:
                buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para

    if buffer:
        pieces.append(buffer)
    return pieces


def split_sections(body: str) -> list[tuple[str, str]]:
    """按 H2 切分；超长小节先按 H3 再切，仍超长则按段落滑窗切分。"""
    sections: list[tuple[str, str]] = []
    for heading, text in _blocks_at(body, 2):
        if not text:
            continue
        if len(text) <= MAX_CHARS:
            sections.append((heading or NO_HEADING, text))
            continue
        for sub_heading, sub_text in _blocks_at(text, 3):
            if not sub_text:
                continue
            label = sub_heading or heading or NO_HEADING
            sections.extend((label, piece) for piece in _split_long(sub_text))
    return sections


def chunk_document(
    raw: str,
    *,
    fallback_doc_id: str = "",
    meta_defaults: dict[str, Any] | None = None,
) -> list[Chunk]:
    """把一篇 Markdown 全文切成 Chunk 列表。

    Args:
        raw: 文档全文。
        fallback_doc_id: front-matter 未给 doc_id 时用文件名推导。
        meta_defaults: 缺省元数据，仅在 front-matter 未提供该字段时生效。
            用于 PDF 等无法携带 front-matter 的来源（由 indexer 合成）。
    """
    meta_raw, body = parse_front_matter(raw)

    merged = dict(meta_raw)
    if meta_defaults:
        for key, value in meta_defaults.items():
            if merged.get(key) in (None, ""):
                merged[key] = value

    doc_meta = build_doc_metadata(merged, fallback_doc_id=fallback_doc_id)

    # H1 是文档标题，已由 doc_title 承载，从正文移除以免重复进块
    body = re.sub(r"^#\s+.*$", "", body, count=1, flags=re.MULTILINE).strip()
    if not body:
        raise ValueError(f"{doc_meta['doc_id']}: 正文为空")

    chunks: list[Chunk] = []
    for index, (heading, text) in enumerate(split_sections(body)):
        chunk_id = f"{doc_meta['doc_id']}_p{index}"
        metadata: dict[str, Any] = {
            **doc_meta,
            "chunk_id": chunk_id,
            "heading": heading,
            "chunk_index": index,
        }
        chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

    if not chunks:
        raise ValueError(f"{doc_meta['doc_id']}: 未切出任何内容块")
    return chunks


__all__ = [
    "Chunk",
    "MAX_CHARS",
    "OVERLAP",
    "NO_HEADING",
    "NO_EXPIRE_DATE",
    "NO_EXPIRE_ORD",
    "parse_front_matter",
    "build_doc_metadata",
    "date_to_ord",
    "split_sections",
    "chunk_document",
]
