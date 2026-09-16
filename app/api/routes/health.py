"""GET /health —— 服务存活与依赖状态检查。

阶段 0：仅反映进程存活与已加载配置。
阶段 4：逐个 ping MCP Server，把结果写入 ``checks``；任一失败即 ``degraded``。

为什么用「实时 ping」而不是读客户端缓存的状态
------------------------------------------
Server 是独立进程，可能在运行中崩溃（Chroma 目录被删、依赖缺失、被手动 kill）。
缓存的启动状态会一直显示 ok，而 ``probe()`` 每次都真实调用一次 ``list_tools()``
—— 代价仅数毫秒，却能反映真实存活状态。这正是蓝图「关掉 chroma_server，
/health 应报告 unhealthy」的落地方式。

``degraded`` 而不是 ``unhealthy`` 的语义
--------------------------------------
只要进程本身活着、`/query` 仍能以降级模式（拒答而非 500）对外服务，整体就不算
不健康。``degraded`` 表达「部分能力缺失但服务可用」，调用方据此决定是否告警
而不是直接摘流量。
"""

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
    """全部 ok 才算健康。空字典（未启用 MCP）视为健康。"""
    if not checks:
        return STATUS_OK
    return STATUS_OK if all(state == STATUS_OK for state in checks.values()) else STATUS_DEGRADED


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(client: MCPToolClient = Depends(get_mcp)) -> HealthResponse:
    if settings.mcp_enabled:
        # probe() 会在 future.result() 上阻塞，必须挪出事件循环，
        # 否则一次 Server 卡死会连带整个服务失去响应。
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
