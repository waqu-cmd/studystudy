"""LangGraph 编排层。

对外只暴露图本身（``get_graph``）、构建入口（``build_graph``）、状态契约
（``GraphState`` / ``initial_state``）与流程图（``graph_mermaid``）。
节点函数属内部实现，外部（API 层、脚本、评估）不应直接调用 —— 阶段 3 起
节点之间插入了核查与回退，绕过图直接调节点会丢失自纠正能力。
"""

from __future__ import annotations

from app.graph.builder import ALL_NODES, build_graph, get_graph, graph_mermaid, reset_graph
from app.graph.state import (
    ACCUMULATING_KEYS,
    NODE_RESET,
    NODE_RETRIEVER,
    NODE_SUPERVISOR,
    NODE_SYNTHESIZER,
    NODE_VERIFIER,
    ROUTE_DIRECT,
    ROUTE_RETRIEVE,
    VERDICT_FAIL,
    VERDICT_PASS,
    GraphState,
    initial_state,
    make_event,
)

__all__ = [
    "build_graph",
    "get_graph",
    "reset_graph",
    "graph_mermaid",
    "ALL_NODES",
    "GraphState",
    "initial_state",
    "make_event",
    "ACCUMULATING_KEYS",
    "ROUTE_RETRIEVE",
    "ROUTE_DIRECT",
    "VERDICT_PASS",
    "VERDICT_FAIL",
    "NODE_RESET",
    "NODE_SUPERVISOR",
    "NODE_RETRIEVER",
    "NODE_SYNTHESIZER",
    "NODE_VERIFIER",
]
