"""GET /health —— 服务存活与依赖状态检查。

阶段 0：仅反映进程存活与已加载配置。
阶段 4：把 MCP Server 连通性（chroma / filesystem / search）写入 checks，
        任一失败则把 status 置为 "degraded"。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import APP_VERSION, settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    version: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    chroma_collection: str
    max_retry: int
    checks: dict[str, str] = Field(
        default_factory=dict, description="各依赖项的连通状态，阶段 4 起填充"
    )


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        chroma_collection=settings.chroma_collection,
        max_retry=settings.max_retry,
        checks={},
    )

