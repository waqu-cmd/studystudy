"""POST /ingest —— 文档上传与索引（阶段 1）。

同步接口：内部是阻塞 IO（读文件、Embedding HTTP、Chroma 写），
因此路由函数用 def 而非 async def，由 FastAPI 自动调度到线程池，
避免阻塞事件循环。

files 留空时默认索引 .env 中 DOCS_DIR 指向目录下的全部受支持文件 ——
这样 /ingest 可以直接当作「重建知识库」按钮使用。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_indexer
from app.core.logging import logger
from app.rag.indexer import Indexer
from app.schemas.ingest import FileIngestResult, IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, summary="索引文档")
def ingest(
    request: IngestRequest,
    indexer: Indexer = Depends(get_indexer),
) -> IngestResponse:
    if request.reset:
        logger.warning("收到 reset 请求，将清空 collection 后重建")
        indexer.reset()

    if request.files:
        outcomes = indexer.index_paths(request.files, force=request.force)
    else:
        outcomes = indexer.index_directory(force=request.force)

    results = [
        FileIngestResult(
            path=outcome.path,
            doc_id=outcome.doc_id,
            doc_title=outcome.doc_title,
            version=outcome.version,
            chunks=outcome.chunks,
            status=outcome.status,  # type: ignore[arg-type]
            message=outcome.message,
        )
        for outcome in outcomes
    ]

    indexed = sum(1 for r in results if r.status == "indexed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")

    logger.info(
        "ingest 完成 | files={} | indexed={} | skipped={} | failed={} | collection={}",
        len(results),
        indexed,
        skipped,
        failed,
        indexer.count(),
    )

    return IngestResponse(
        collection=indexer.collection_name,
        total_files=len(results),
        total_chunks=sum(r.chunks for r in results),
        indexed=indexed,
        skipped=skipped,
        failed=failed,
        collection_count=indexer.count(),
        results=results,
    )


@router.get("/ingest/stats", summary="查看索引现状")
def ingest_stats(indexer: Indexer = Depends(get_indexer)) -> dict[str, object]:
    """返回 collection 内的块数与文档清单，便于确认索引是否生效。"""
    from app.rag.retriever import load_corpus

    corpus = load_corpus(indexer.collection)
    docs: dict[str, dict[str, object]] = {}
    for meta in corpus.metadatas:
        doc_id = str(meta.get("doc_id", ""))
        entry = docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "doc_title": meta.get("doc_title", ""),
                "version": meta.get("version", ""),
                "expire_date": meta.get("expire_date", ""),
                "chunks": 0,
            },
        )
        entry["chunks"] = int(entry["chunks"]) + 1

    return {
        "collection": indexer.collection_name,
        "chunk_count": indexer.count(),
        "document_count": len(docs),
        "documents": sorted(docs.values(), key=lambda item: str(item["doc_id"])),
    }


__all__ = ["router"]
