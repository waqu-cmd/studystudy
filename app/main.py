"""FastAPI 入口：初始化日志、启动 MCP 通道、挂载路由、管理生命周期。

阶段 1 挂载：/health、/ingest、/ingest/stats、/query。
阶段 4 在 lifespan 内启动 MultiServerMCPClient 并动态发现工具。

主进程在 MCP 模式下不直接打开 Chroma
-----------------------------------
阶段 1~3 的启动探测是 `get_collection().count()`，那会让主进程持有 Chroma 连接。
阶段 4 起检索由 chroma_server 子进程负责，主进程若同时持有连接，就会形成
「同目录多进程访问」的局面 —— Chroma 0.5.x 底层是 SQLite + hnswlib 本地文件，
并发读写存在锁争用风险。因此 MCP 模式下改为通过 `collection_stats` 工具取统计，
主进程彻底不碰向量库；仅在关闭 MCP（``MCP_ENABLED=false``）时才直连。

MCP 启动失败不阻止服务启动
-------------------------
`client.start()` 任一 Server 失败都只记录状态而不抛异常。服务照常起来，
`/health` 报 degraded、`/query` 降级为「无参考资料」。这是阶段 4 的验收要求：
工具层是独立进程，它挂了不应连带推理层一起挂。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import health, ingest, query
from app.core.config import APP_VERSION, settings
from app.core.logging import logger, setup_logging
from app.mcp_client import get_default_mcp, reset_default_mcp

TOOL_COLLECTION_STATS = "collection_stats"


async def _report_index_state(client) -> None:
    """启动时报告索引状态。MCP 模式下经工具查询，不直连向量库。"""
    if settings.mcp_enabled:
        result = await asyncio.to_thread(
            client.call_tool, TOOL_COLLECTION_STATS, {}
        )
        data = result.data if isinstance(result.data, dict) else {}
        if result.ok and data:
            logger.info(
                "索引状态（经 MCP）| 总块数={} | 未过期={} | 集合={}",
                data.get("total"),
                data.get("active"),
                data.get("collection"),
            )
        else:
            logger.warning("索引状态查询失败：{}", result.error or "返回为空")
        return

    try:
        from app.rag.indexer import get_collection

        logger.info("已索引块数={}（进程内直连）", get_collection().count())
    except Exception as exc:  # noqa: BLE001 - 探测失败不应阻止服务启动
        logger.warning("Chroma 启动探测失败：{}: {}", type(exc).__name__, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 放在 lifespan 而非模块导入期：避免 pytest 收集测试时被 basicConfig(force=True)
    # 拆掉 root handlers，导致 caplog 之类的 fixture 失效。
    setup_logging()

    logger.info(
        "服务启动 | llm={} | embedding={}({}d) | top_k={} | max_retry={}",
        settings.llm_model,
        settings.embedding_model,
        settings.embedding_dim,
        settings.top_k,
        settings.max_retry,
    )
    logger.info("向量库={} | 集合={}", settings.chroma_path, settings.chroma_collection)

    client = get_default_mcp()

    if settings.mcp_enabled:
        # start() 要 spawn 子进程并完成 JSON-RPC 握手（实测单个 Server 约 0.7s，
        # 真实 Server 需 import chromadb 会更久），必须挪出事件循环。
        started = await asyncio.to_thread(client.start)
        if started:
            snapshot = client.write_tools_snapshot()
            logger.info(
                "MCP 通道就绪 | servers={} | tools={} | 快照={}",
                client.server_names,
                client.tool_names(),
                snapshot.name if snapshot else "未写入",
            )
        else:
            logger.error(
                "MCP 通道未能就绪，检索将降级为「无参考资料」| status={}",
                client.status_map,
            )
        await _report_index_state(client)
    else:
        logger.warning(
            "MCP_ENABLED=false：检索退回进程内直连通道（阶段 3 行为），"
            "用于 A/B 对比与故障回退"
        )
        await _report_index_state(client)

    yield

    if settings.mcp_enabled:
        await asyncio.to_thread(client.stop)
        # 让单例回到干净状态：uvicorn --reload 会重建应用，
        # 残留的旧实例会让下一轮 lifespan 复用一个已关闭的客户端。
        reset_default_mcp()

    logger.info("服务停止")
    logger.complete()  # 等待 enqueue 的文件 sink 把队列刷完


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
    }
