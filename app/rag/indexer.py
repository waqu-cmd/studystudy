"""Chroma 访问层 + 文档解析与写入（阶段 1）。

本模块同时承担两件事，因为它们是同一份状态的两面：

1. **存储访问层** —— Chroma client / collection 的进程内单例。
   retriever.py 单向依赖本模块，依赖方向始终是 retriever -> indexer。
2. **写入层** —— 读文件 -> chunker 切块 -> 批量向量化 -> upsert。

Chroma 的三个关键约定（均为实测结论）
--------------------------------------
1. `embedding_function=None` 必须显式传。不传时 Chroma 会挂上默认的 ONNX
   all-MiniLM 模型，首次调用会联网下载，且其 384 维与我们 1024 维的向量冲突。
2. `hnsw:space=cosine` 必须显式设定。默认是 l2，而 text-embedding 系列按余弦
   相似度训练，用 l2 会明显劣化排序质量。
3. collection 名需满足 3~63 字符、仅含字母数字与下划线/连字符。
4. 日期范围过滤只能用整数派生字段（expire_ord / effective_ord），因为 Chroma 的
   $gte/$lte 只接受 int/float，传 ISO 字符串直接抛 ValueError。
   （详见 chunker.py 的模块文档）

幂等性
------
chunk_id = f"{doc_id}_p{index}" 是确定性的，写入用 upsert。但块数变少时旧块会残留，
因此每次写入都先 `delete(where={"doc_id": ...})` 再 upsert，保证「先删后写」。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from pypdf import PdfReader

from app.core.config import PROJECT_ROOT, settings
from app.core.llm import embed_texts
from app.core.logging import logger
from app.rag.chunker import Chunk, chunk_document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

# 写入 metadata 用的字段名
HASH_FIELD = "content_hash"
INDEXED_AT_FIELD = "indexed_at"


# --------------------------------------------------------------------------- #
# 存储访问层
# --------------------------------------------------------------------------- #

_client_cache: dict[str, chromadb.ClientAPI] = {}
_collection_cache: dict[str, Collection] = {}


def get_chroma_client(path: str | Path | None = None) -> chromadb.ClientAPI:
    """PersistentClient 单例。

    同一进程内对同一路径重复构造 PersistentClient 会拿到同一底层实例，
    这里再缓存一层以避免重复的 settings 校验开销。
    """
    resolved = Path(path) if path is not None else settings.chroma_path
    key = str(resolved)
    if key not in _client_cache:
        resolved.mkdir(parents=True, exist_ok=True)
        _client_cache[key] = chromadb.PersistentClient(
            path=key,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.debug("Chroma client 就绪 | path={}", key)
    return _client_cache[key]


def get_collection(name: str | None = None) -> Collection:
    """collection 单例（余弦距离、不使用内置 embedding function）。"""
    collection_name = name or settings.chroma_collection
    if collection_name not in _collection_cache:
        client = get_chroma_client()
        _collection_cache[collection_name] = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("Chroma collection 就绪 | name={}", collection_name)
    return _collection_cache[collection_name]


def drop_collection(name: str | None = None) -> bool:
    """删除 collection 并清空本地缓存。返回是否确实删除了已存在的 collection。"""
    collection_name = name or settings.chroma_collection
    client = get_chroma_client()
    existed = collection_name in {c.name for c in client.list_collections()}
    if existed:
        client.delete_collection(collection_name)
    _collection_cache.pop(collection_name, None)
    return existed


def _notify_corpus_changed() -> None:
    """索引发生写入后通知检索层刷新语料缓存。

    这里用函数内导入：retriever 需要本模块的 get_collection，若在模块级互相
    import 会形成循环。延迟到调用时导入可避开该循环，依赖方向仍是单向下行。
    """
    from app.rag.retriever import invalidate_corpus_cache

    invalidate_corpus_cache()


# --------------------------------------------------------------------------- #
# 写入层
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class IndexOutcome:
    """单文件索引结果（与 schemas.ingest.FileIngestResult 字段一一对应）。"""

    path: str
    status: str  # indexed | skipped | failed
    doc_id: str = ""
    doc_title: str = ""
    version: str = ""
    chunks: int = 0
    message: str = ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Indexer:
    """把文档写进 Chroma collection。"""

    def __init__(self, collection: Collection | None = None) -> None:
        self._collection = collection

    # ---------------- 属性 ----------------

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = get_collection()
        return self._collection

    @property
    def collection_name(self) -> str:
        return self.collection.name

    def count(self) -> int:
        return self.collection.count()

    # ---------------- 生命周期 ----------------

    def reset(self) -> bool:
        """清空 collection 内全部数据（危险操作，调用方需先确认）。"""
        self._collection = None
        existed = drop_collection()
        _notify_corpus_changed()
        logger.warning("已清空 collection | name={}", settings.chroma_collection)
        return existed

    def delete_document(self, doc_id: str) -> None:
        """按 doc_id 删除该文档的全部块。"""
        self.collection.delete(where={"doc_id": doc_id})
        _notify_corpus_changed()

    # ---------------- 索引 ----------------

    def index_paths(self, paths: list[str], *, force: bool = True) -> list[IndexOutcome]:
        return [self.index_file(path, force=force) for path in paths]

    def index_directory(
        self, directory: str | Path | None = None, *, force: bool = True
    ) -> list[IndexOutcome]:
        """索引目录下的全部受支持文件（按文件名排序，保证确定性）。"""
        target = Path(directory) if directory is not None else settings.docs_dir
        if not target.is_dir():
            return [
                IndexOutcome(
                    path=str(target), status="failed", message="目录不存在或不是目录"
                )
            ]
        files = sorted(
            p
            for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        logger.info("开始批量索引 | dir={} | files={}", target, len(files))
        return [self.index_file(p, force=force) for p in files]

    def index_file(self, raw_path: str | Path, *, force: bool = True) -> IndexOutcome:
        """索引单个文件。任何异常都收敛为 status="failed"，不向上抛。"""
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        display = str(path)

        try:
            if not path.exists():
                return IndexOutcome(path=display, status="failed", message="文件不存在")
            if not path.is_file():
                return IndexOutcome(path=display, status="failed", message="不是文件")
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                return IndexOutcome(
                    path=display,
                    status="failed",
                    message=f"不支持的后缀 {path.suffix}，仅支持 {sorted(SUPPORTED_SUFFIXES)}",
                )

            raw_text, meta_defaults = self._read_document(path)
            doc_hash = _sha(raw_text)

            chunks = chunk_document(
                raw_text, fallback_doc_id=path.stem, meta_defaults=meta_defaults
            )
            doc_id = str(chunks[0].metadata["doc_id"])
            doc_title = str(chunks[0].metadata["doc_title"])
            version = str(chunks[0].metadata["version"])

            if not force and self._existing_hash(doc_id) == doc_hash:
                logger.info("内容未变化，跳过 | doc_id={} | chunks={}", doc_id, len(chunks))
                return IndexOutcome(
                    path=display,
                    status="skipped",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    version=version,
                    chunks=len(chunks),
                    message="内容 hash 未变化",
                )

            vectors = embed_texts([c.text for c in chunks])
            self._write(chunks, vectors, doc_hash=doc_hash)
            _notify_corpus_changed()

            logger.info(
                "索引完成 | doc_id={} | version={} | chunks={} | collection={}",
                doc_id,
                version,
                len(chunks),
                self.count(),
            )
            return IndexOutcome(
                path=display,
                status="indexed",
                doc_id=doc_id,
                doc_title=doc_title,
                version=version,
                chunks=len(chunks),
            )
        except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断整批
            logger.exception("索引失败 | path={}", display)
            return IndexOutcome(
                path=display, status="failed", message=f"{type(exc).__name__}: {exc}"
            )

    # ---------------- 内部实现 ----------------

    def _write(
        self, chunks: list[Chunk], vectors: list[list[float]], *, doc_hash: str
    ) -> None:
        """先删同 doc_id 的旧块，再 upsert 新块。"""
        doc_id = str(chunks[0].metadata["doc_id"])
        self.collection.delete(where={"doc_id": doc_id})

        indexed_at = _now_iso()
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    **chunk.metadata,
                    HASH_FIELD: doc_hash,
                    INDEXED_AT_FIELD: indexed_at,
                }
            )

        self.collection.upsert(
            ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
        )

    def _existing_hash(self, doc_id: str) -> str | None:
        result = self.collection.get(
            where={"doc_id": doc_id}, limit=1, include=["metadatas"]
        )
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return metadatas[0].get(HASH_FIELD)

    def _read_document(self, path: Path) -> tuple[str, dict[str, Any] | None]:
        """读取文件全文。PDF 无 front-matter 时合成默认元数据。

        Returns:
            (全文, meta_defaults)。meta_defaults 只在 front-matter 缺该字段时生效，
            因此 Markdown 一律返回 None。
        """
        if path.suffix.lower() == ".pdf":
            text = self._read_pdf(path)
            # PDF 无法携带 YAML front-matter；抽出文本若含 front-matter 则照常解析，
            # 否则用文件名 + 文件修改日期合成，保证 doc_id / version / effective_date 齐全。
            defaults = {
                "doc_id": path.stem,
                "doc_title": path.stem,
                "version": "v1",
                "effective_date": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%Y-%m-%d"),
                "source": "pdf",
            }
            logger.info("PDF 使用合成元数据 | path={} | doc_id={}", path.name, path.stem)
            return text, defaults

        return path.read_text(encoding="utf-8"), None

    @staticmethod
    def _read_pdf(path: Path) -> str:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(p for p in pages if p)
        if not text.strip():
            raise ValueError("PDF 未抽出任何文本（可能是扫描件，需先做 OCR）")
        return text


__all__ = [
    "SUPPORTED_SUFFIXES",
    "HASH_FIELD",
    "INDEXED_AT_FIELD",
    "IndexOutcome",
    "Indexer",
    "get_chroma_client",
    "get_collection",
    "drop_collection",
]
