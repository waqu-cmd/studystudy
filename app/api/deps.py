"""依赖注入。

阶段 1：Indexer / HybridRetriever。
阶段 2：追加编译好的 LangGraph 实例（/query 由图执行）。
阶段 4：追加 MCPToolClient（工具层通道）。

实例的归属
----------
单例本体定义在各自的「使用方所在层」，本模块只做转发：

- 检索器 → ``graph/nodes/retriever.py``（图既被 API 调用、也被评估脚本调用）
- MCP 客户端 → ``mcp_client.py``（图与 /health 都要用）

这样 API 层不会持有第二份实例，避免「两个客户端各自维护连接」这种难查的漂移。

用 lru_cache 而非模块级全局：FastAPI 的 Depends 每次请求都会调用工厂函数，
不加缓存会反复构造 Chroma client。
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.graph.builder import get_graph as _build_default_graph
from app.graph.nodes.retriever import get_default_retriever
from app.mcp_client import MCPToolClient, get_default_mcp
from app.rag.indexer import Indexer
from app.rag.retriever import HybridRetriever


@lru_cache(maxsize=1)
def get_indexer() -> Indexer:
    """索引器单例。"""
    return Indexer()


def get_retriever() -> HybridRetriever:
    """检索器单例（转发到图层的单例，保证全局只有一个实例）。"""
    return get_default_retriever()


def get_mcp() -> MCPToolClient:
    """MCP 客户端单例（转发到 mcp_client 层）。"""
    return get_default_mcp()


def get_graph() -> CompiledStateGraph:
    """编译好的问答图（进程内单例）。

    单独包一层而不直接暴露 builder.get_graph，是为了让 FastAPI 能通过
    ``app.dependency_overrides[get_graph]`` 注入测试专用图（替换检索器与 LLM 替身）。
    """
    return _build_default_graph()


__all__ = ["get_indexer", "get_retriever", "get_mcp", "get_graph"]
