"""依赖注入。"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.graph.builder import get_graph as _build_default_graph
from app.mcp_client import MCPToolClient, get_default_mcp
from app.rag.indexer import Indexer


@lru_cache(maxsize=1)
def get_indexer() -> Indexer:
    return Indexer()


def get_mcp() -> MCPToolClient:
    return get_default_mcp()


def get_graph() -> CompiledStateGraph:
    return _build_default_graph()


__all__ = ["get_indexer", "get_mcp", "get_graph"]
