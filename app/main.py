"""FastAPI 入口：启动 MCP 通道、挂载路由、管理生命周期。

阶段 1 挂载：/health、/ingest、/ingest/stats、/query。
阶段 4 在 lifespan 内启动 MultiServerMCPClient 并动态发现工具。

主进程在 MCP 模式下不直接打开 Chroma
-----------------------------------
阶段 1~3 的启动探测是 `get_collection().count()`，那会让主进程持有 Chroma 连接。
阶段 4 起检索由 chroma_server 子进程负责，主进程若同时持有连接，就会形成
「同目录多进程访问」的局面 —— Chroma 0.5.x 底层是 SQLite + hnswlib 本地文件，
并发读写存在锁争用风险。因此 MCP 模式下检索全程经 MCP 工具完成，主进程彻底
不碰向量库；仅在关闭 MCP（``MCP_ENABLED=false``）时才直连。

MCP 启动失败不阻止服务启动
-------------------------
`client.start()` 任一 Server 失败都只把结果写进 ``status_map`` 而不抛异常。服务
照常起来，`/health` 报 degraded、`/query` 降级为「无参考资料」。这是阶段 4 的
验收要求：工具层是独立进程，它挂了不应连带推理层一起挂。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import health, ingest, query
from app.core.config import APP_VERSION, settings
from app.mcp_client import get_default_mcp, reset_default_mcp


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    client = get_default_mcp()

    if settings.mcp_enabled:
        # start() 要 spawn 子进程并完成 JSON-RPC 握手（实测单个 Server 约 0.7s，
        # 真实 Server 需 import chromadb 会更久），必须挪出事件循环。
        await asyncio.to_thread(client.start)

    yield

    if settings.mcp_enabled:
        await asyncio.to_thread(client.stop)
        # 让单例回到干净状态：uvicorn --reload 会重建应用，
        # 残留的旧实例会让下一轮 lifespan 复用一个已关闭的客户端。
        reset_default_mcp()


app = FastAPI(
    title="企业知识库智能分析 Agent",
    description="基于 LangGraph + MCP + ChromaDB 的 Supervisor 多 Agent 知识库问答系统",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "enterprise-kb-agent",
        "docs": "/docs",
        "health": "/health",
        "ingest": "/ingest",
        "ingest_stats": "/ingest/stats",
        "query": "/query",
        "query_stream": "/query/stream",
    }
