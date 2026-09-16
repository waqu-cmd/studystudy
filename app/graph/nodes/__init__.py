"""图节点集合。

节点自行获取单例的约定：节点函数都接受仅关键字注入参数
（``retriever`` / ``llm``），默认取进程内单例。因此本模块的导出只暴露函数本身，
不做任何实例化。

阶段 3 起包含 ``verifier_node`` —— 它是自纠正回路的决策点。外部（API 层、脚本、
评估）不应绕过图直接调用它：``retry_count`` 的递增与 ``search_query`` 的改写都由
它完成，单独调用会得到一个没有回退能力的「半成品」核查。
"""

from __future__ import annotations

from app.graph.nodes.retriever import get_default_retriever, retriever_node
from app.graph.nodes.supervisor import decide_route, reset_turn_node, supervisor_node
from app.graph.nodes.synthesizer import (
    REFUSE_TEXT,
    extract_citations,
    format_context,
    synthesizer_node,
)
from app.graph.nodes.verifier import (
    VerificationResult,
    build_retry_query,
    normalize_verdict,
    verifier_node,
)

__all__ = [
    "retriever_node",
    "get_default_retriever",
    "supervisor_node",
    "reset_turn_node",
    "decide_route",
    "synthesizer_node",
    "REFUSE_TEXT",
    "format_context",
    "extract_citations",
    "verifier_node",
    "VerificationResult",
    "normalize_verdict",
    "build_retry_query",
]
