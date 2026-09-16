"""ingest 请求 / 响应模型（阶段 1）。

字段命名与蓝图「三、核心数据结构设计」保持一致；蓝图未规定处按最小必要补充。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IngestStatus = Literal["indexed", "skipped", "failed"]


class IngestRequest(BaseModel):
    """POST /ingest 请求体。"""

    files: list[str] = Field(
        default_factory=list,
        description=(
            "待索引文件路径列表。支持绝对路径，或相对项目根目录的路径；"
            "支持 .md / .markdown / .txt / .pdf。"
        ),
    )
    force: bool = Field(
        default=True,
        description=(
            "内容未变化时是否强制重新索引。默认 true（幂等重写）；"
            "置 false 可在文档 hash 未变时跳过，省下 Embedding 调用。"
        ),
    )
    reset: bool = Field(
        default=False,
        description=(
            "索引前清空整个 collection。危险操作：会删除全部已索引文档，"
            "仅用于重建基线。"
        ),
    )


class FileIngestResult(BaseModel):
    """单个文件的索引结果。"""

    path: str
    doc_id: str = ""
    doc_title: str = ""
    version: str = ""
    chunks: int = 0
    status: IngestStatus
    message: str = ""


class IngestResponse(BaseModel):
    """POST /ingest 响应体。"""

    collection: str
    total_files: int
    total_chunks: int
    indexed: int
    skipped: int
    failed: int
    collection_count: int = Field(description="本次操作完成后 collection 内的块总数")
    results: list[FileIngestResult] = Field(default_factory=list)


__all__ = ["IngestStatus", "IngestRequest", "FileIngestResult", "IngestResponse"]
