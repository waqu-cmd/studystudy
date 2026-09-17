"""GET /health —— 服务存活与依赖状态检查。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_mcp
from app.core.config import APP_VERSION, settings
from app.mcp_client import MCPToolClient

router = APIRouter(tags=["health"])

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded。degraded 表示部分依赖不可用，服务仍可降级运行")
    version: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    chroma_collection: str
    max_retry: int
    retrieval_channel: str = Field(
        description="当前检索通道：mcp（工具层解耦）| in-process（进程内直连）"
    )
    mcp_tools: list[str] = Field(
        default_factory=list, description="启动时动态发现的工具名，来自 list_tools()"
    )
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="各 MCP Server 的实时连通状态：ok / error: 原因 / 未启动",
    )


def _evaluate_status(checks: dict[str, str]) -> str:
    if not checks:
        return STATUS_OK
    return STATUS_OK if all(state == STATUS_OK for state in checks.values()) else STATUS_DEGRADED


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(client: MCPToolClient = Depends(get_mcp)) -> HealthResponse:
    if settings.mcp_enabled:
        checks = await asyncio.to_thread(client.probe)
        channel = "mcp"
    else:
        checks = {}
        channel = "in-process"

    return HealthResponse(
        status=_evaluate_status(checks),
        version=APP_VERSION,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        chroma_collection=settings.chroma_collection,
        max_retry=settings.max_retry,
        retrieval_channel=channel,
        mcp_tools=client.tool_names() if settings.mcp_enabled else [],
        checks=checks,
    )
