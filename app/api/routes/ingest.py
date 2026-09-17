"""POST /ingest —— 文档上传与索引。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_indexer
from app.rag.indexer import Indexer
from app.schemas.ingest import FileIngestResult, IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, summary="索引文档")
def ingest(
    request: IngestRequest,
    indexer: Indexer = Depends(get_indexer),
) -> IngestResponse:
    if request.reset:
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
