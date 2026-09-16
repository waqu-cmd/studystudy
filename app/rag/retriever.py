"""向量 + BM25 混合检索（阶段 1）。

两路召回
--------
1. **稠密**：Chroma 向量检索，把 `expire_ord >= today` 的过滤条件下推到库。
2. **稀疏**：BM25（rank_bm25）关键词检索，语料为库内全部块。

融合用 RRF（见 rrf.py）：只使用排名、不使用分数，避免把余弦距离与 BM25 分数
按不同量纲相加。

为什么 BM25 也要做时效过滤
--------------------------
BM25 在本地内存里算，没有下推能力，因此拉取全量语料后在构建索引时就按
expire_ord 过滤。若只在向量一路过滤，已过期的历史版本会仅从稀疏这一路混进
结果集 —— 阶段 7 的「版本准确率」指标会直接失真。

中文分词
--------
rank_bm25 是纯词袋实现、不做分词。中文若整串当一个 token 则几乎无法命中，
因此按「单字 + 相邻二元组」切分："销售政策" -> 销/售/政/策 + 销售/售政/政策。
bigram 提供短语级精度，单字提供召回兜底。

语料缓存
--------
BM25 需要全量语料才能算 IDF，故在进程内缓存，(count, Corpus) 作为缓存项；
indexer 写入后会调用 invalidate_corpus_cache() 立即失效。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.llm import embed_query
from app.core.logging import logger
from app.rag.chunker import NO_EXPIRE_ORD, date_to_ord
from app.rag.indexer import get_collection
from app.rag.rrf import DEFAULT_K, ranks_of, rrf_fuse
from app.schemas.query import RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def today_ord() -> int:
    """今天对应的 YYYYMMDD 整数，与 chunker 写入的 expire_ord 同量纲。"""
    return int(date.today().strftime("%Y%m%d"))


def tokenize(text: str) -> list[str]:
    """中文「单字 + 二元组」、英文数字整词小写。

    纯标点输入会得到空列表，调用方需对空 token 做兜底。
    """
    tokens: list[str] = []
    for matched in _TOKEN_RE.finditer(text or ""):
        piece = matched.group(0)
        if piece.isascii():
            tokens.append(piece.lower())
            continue
        tokens.extend(piece)
        tokens.extend(piece[i:i + 2] for i in range(len(piece) - 1))
    return tokens


@dataclass(slots=True)
class Corpus:
    """全量语料快照，含两个 BM25 索引（全量 / 仅未过期）。"""

    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    bm25_all: BM25Okapi | None
    bm25_active: BM25Okapi | None
    active_positions: list[int]

    def position_of(self, chunk_id: str) -> int | None:
        try:
            return self.ids.index(chunk_id)
        except ValueError:
            return None


_corpus_cache: dict[str, tuple[int, Corpus]] = {}


def invalidate_corpus_cache() -> None:
    """清空语料缓存。indexer 每次写入后调用。"""
    _corpus_cache.clear()


def load_corpus(collection: Any | None = None) -> Corpus:
    """加载（或命中缓存）全量语料。"""
    col = collection if collection is not None else get_collection()
    name = col.name
    count = col.count()

    cached = _corpus_cache.get(name)
    if cached is not None and cached[0] == count:
        return cached[1]

    data = col.get(include=["documents", "metadatas"])
    ids = [str(i) for i in (data.get("ids") or [])]
    documents = [d or "" for d in (data.get("documents") or [])]
    metadatas = [m or {} for m in (data.get("metadatas") or [])]

    tokens = [(tokenize(doc) or ["<empty>"]) for doc in documents]
    bm25_all = BM25Okapi(tokens) if tokens else None

    expiry = today_ord()
    active_positions = [
        pos
        for pos, meta in enumerate(metadatas)
        if int(meta.get("expire_ord", NO_EXPIRE_ORD)) >= expiry
    ]
    active_tokens = [tokens[pos] for pos in active_positions]
    bm25_active = BM25Okapi(active_tokens) if active_tokens else None

    corpus = Corpus(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        bm25_all=bm25_all,
        bm25_active=bm25_active,
        active_positions=active_positions,
    )
    _corpus_cache[name] = (count, corpus)
    logger.debug(
        "语料缓存已重建 | collection={} | chunks={} | active={}",
        name,
        len(ids),
        len(active_positions),
    )
    return corpus


class HybridRetriever:
    """混合检索：向量 + BM25，RRF 融合。"""

    def __init__(
        self,
        *,
        collection: Any | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
        include_expired: bool = False,
    ) -> None:
        self._collection = collection
        self.top_k = top_k or settings.top_k
        self.rrf_k = rrf_k or settings.rrf_k
        self.include_expired = include_expired

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = get_collection()
        return self._collection

    # ---------------- 对外接口 ----------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_expired: bool | None = None,
    ) -> list[RetrievedChunk]:
        """检索并返回融合排序后的 chunk 列表（长度 <= top_k）。"""
        text = (query or "").strip()
        if not text:
            return []

        limit = top_k or self.top_k
        allow_expired = (
            self.include_expired if include_expired is None else include_expired
        )

        corpus = load_corpus(self.collection)
        if not corpus.ids:
            logger.warning("collection 为空，检索无结果 | collection={}", self.collection.name)
            return []

        vector_ranking, distance_by_id = self._vector_search(text, limit, allow_expired)
        bm25_ranking = self._bm25_search(corpus, text, limit, allow_expired)

        fused = rrf_fuse([vector_ranking, bm25_ranking], k=self.rrf_k)
        vector_ranks = ranks_of(vector_ranking)
        bm25_ranks = ranks_of(bm25_ranking)

        lookup = {chunk_id: pos for pos, chunk_id in enumerate(corpus.ids)}
        hits: list[RetrievedChunk] = []
        for chunk_id, score in fused[:limit]:
            pos = lookup.get(chunk_id)
            if pos is None:
                continue
            meta = corpus.metadatas[pos]
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta.get("doc_id", "")),
                    doc_title=str(meta.get("doc_title", "")),
                    version=str(meta.get("version", "")),
                    heading=str(meta.get("heading", "")),
                    effective_date=str(meta.get("effective_date", "")),
                    expire_date=str(meta.get("expire_date", "")),
                    text=corpus.documents[pos],
                    score=round(score, 6),
                    vector_rank=vector_ranks.get(chunk_id),
                    bm25_rank=bm25_ranks.get(chunk_id),
                    vector_distance=distance_by_id.get(chunk_id),
                )
            )

        logger.debug(
            "检索完成 | query={!r} | vector={} | bm25={} | fused={}",
            text[:30],
            len(vector_ranking),
            len(bm25_ranking),
            len(hits),
        )
        return hits

    def search_chunk_ids(self, query: str, *, top_k: int | None = None) -> list[str]:
        """只返回 chunk_id 有序列表，供评估脚本计算 Hit@K / MRR。"""
        return [hit.chunk_id for hit in self.search(query, top_k=top_k)]

    # ---------------- 内部实现 ----------------

    def _vector_search(
        self, query: str, limit: int, include_expired: bool
    ) -> tuple[list[str], dict[str, float]]:
        vector = embed_query(query)
        where = None if include_expired else {"expire_ord": {"$gte": today_ord()}}
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=max(limit, 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = list((result.get("ids") or [[]])[0])
        distances = list((result.get("distances") or [[]])[0])
        return ids, {cid: float(dist) for cid, dist in zip(ids, distances)}

    def _bm25_search(
        self, corpus: Corpus, query: str, limit: int, include_expired: bool
    ) -> list[str]:
        bm25 = corpus.bm25_all if include_expired else corpus.bm25_active
        if bm25 is None:
            return []

        positions = (
            list(range(len(corpus.ids))) if include_expired else corpus.active_positions
        )
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)
        # 过滤 score <= 0：语料很小时 IDF 可能为负，负分候选没有召回价值
        ranked = sorted(
            ((pos, float(score)) for pos, score in zip(positions, scores) if score > 0),
            key=lambda item: -item[1],
        )[:limit]
        return [corpus.ids[pos] for pos, _ in ranked]


__all__ = [
    "Corpus",
    "HybridRetriever",
    "load_corpus",
    "invalidate_corpus_cache",
    "tokenize",
    "today_ord",
]
