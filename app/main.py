"""FastAPI 入口：初始化日志、挂载路由、管理应用生命周期（接入 MCP Client）。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import health
from app.core.config import APP_VERSION, settings
from app.core.logging import logger, setup_logging


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
    logger.info(
        "向量库={} | 集合={}", settings.chroma_path, settings.chroma_collection
    )

    # 阶段 4：在此启动 MultiServerMCPClient，并用 list_tools() 动态发现工具
    yield
    # 阶段 4：在此关闭 MCP Client

    logger.info("服务停止")
    logger.complete()  # 等待 enqueue 的文件 sink 把队列刷完


app = FastAPI(
    title="企业知识库智能分析 Agent",
    description="基于 LangGraph + MCP + ChromaDB 的 Supervisor 多 Agent 知识库问答系统",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(health.router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "enterprise-kb-agent", "docs": "/docs", "health": "/health"}
