"""StateGraph 构建与编译（阶段 2 建立，阶段 3 插入自纠正回路，
阶段 4 接入 MCP 工具层，阶段 5 引入 Supervisor 并行分发与多轮会话）。

图结构
------
::

    单路检索（唯一检索词 —— 阶段 2~4 的原路径）
    START → reset_turn → supervisor → retriever → synthesizer → verifier → END
                                                                  ↑           │
                                                    (fail & retry<max)────────┘

    并行检索（拆出 ≥2 个子问题 —— 阶段 5 新增）
    reset_turn → supervisor ─ Send × N ─┬→ retriever(子问题1) ┐
                                        ├→ retriever(子问题2) ┤ 全部落定后
                                        └→ retriever(子问题3) ┘ 汇合到同一节点
                                                              │
                                                              ↓
                                                         synthesizer →（后续同上）

    direct 路由（寒暄 / 元问题，知识库里没有可查的内容）
    supervisor ────────────────────────────────────────────→ synthesizer

**阶段 4 没有改动图结构** —— 这正是阶段 2 把流程图定型的价值：MCP 只替换了
retriever 节点的**内部实现**（本地 HybridRetriever → ``call_tool("search_documents")``），
节点名、边、状态契约全部不变。图不需要为工具层的进程边界付出任何复杂度。

自纠正回路的三个要点：

1. **回退复用 retriever 节点**，不为补检单独建节点 —— 首检与补检的差别只有检索词，
   而检索词由 ``search_query`` 承载，因此 retriever 至今不知道自纠正循环的存在。
2. **``retry_count`` 的闸门在边上看，递增在 retriever 里做**。verifier 只提建议，
   「建议是否被采纳」由 ``edges.route_after_verifier`` 裁定，因此递增必须发生在
   建议被采纳之后（即 retriever 真正执行补检时）—— 否则被闸门拦下的建议也会计数，
   ``max_retry=0`` 时就会出现「回退 0 次却记 1 次」的歧义。路由函数不写状态：
   它可能被多次调用，任何副作用写在那儿都会丢失或重复执行。
3. **回退到 retriever 而不是 synthesizer**：重检的意义在于引入新证据，
   没有新块的重新生成只会得到同一份答案。

阶段 5：把 supervisor 从「选边」升级为「调度」
--------------------------------------------
两处结构性变化，其余节点几乎零改动：

1. **并行分发**。``route_after_supervisor`` 在拆出 ≥2 个子问题时返回
   ``list[Send]``，每个子问题一路并发检索，全部落定后由 LangGraph 汇合到
   ``synthesizer``。实测语义：N 个 ``Send`` ⇒ 目标节点执行 N 次，下游汇合节点
   **只执行一次**，且看到的是全部 N 支写完之后的合并状态。图里**没有聚合节点** ——
   带 reducer 的通道本身就是汇合点，这是把并发能力做进状态契约（而不是做进节点）
   换来的简洁。
2. **多轮会话**。``get_graph()`` 编译时挂进程级 ``MemorySaver``，
   ``session_id`` 作为 ``thread_id``。跨轮保留状态的同时，``reset_turn``
   在每轮入口清空累加字段，避免上一轮污染下一轮。

节点职责与蓝图的对应关系：``supervisor`` 是 Supervisor，``retriever`` /
``synthesizer`` / ``verifier`` 是三个 Worker。``reset_turn`` 是蓝图未列出、
为本项目多轮会话预留的入口节点 —— 阶段 2 只是预留，阶段 5 挂上 checkpointer 后
它成为必需品（详见 ``nodes/supervisor.py::reset_turn_node``）。

为什么 ``checkpointer`` / ``mcp`` / ``llm`` / ``supervisor_llm`` 都是可注入参数
------------------------------------------------------------------------------
同一个理由：**可测性**。测试要能挂 ``MemorySaver`` 验证跨轮隔离、要能注入
「指向不存在 Server」的客户端验证降级路径、要能注入替身 LLM 覆盖全部分支而不打网络。
生产路径一律传 None，节点内部取进程内单例。

``supervisor_llm`` 单独成一个参数（而不是复用 ``llm``）是阶段 5 的新增需求：
supervisor 是**路由节点**，它调用模型只是为了做判断，与「生成答案」是两件事。
测试里这两者常常需要不同的替身 —— 例如验证「无参考资料时不得调用生成模型」，
就必须让 supervisor 用另一个（或空）替身，否则断言会误伤路由节点的那次调用。

单例
----
``get_graph()`` 用 lru_cache 缓存编译结果：图编译有固定开销，且节点直接复用
进程内的检索器/LLM/MCP 单例，重复编译没有收益。**checkpointer 是进程级单例**，
这既是多轮会话生效的前提（同一进程内共享历史），也是它的边界：
多副本部署时各副本的历史互相独立，会话不共享（本项目单副本运行，不构成问题）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.logging import logger
from app.graph.edges import (
    AFTER_RETRIEVER_MAP,
    AFTER_SUPERVISOR_MAP,
    AFTER_SYNTHESIZER_MAP,
    AFTER_VERIFIER_MAP,
    make_route_after_verifier,
    route_after_retriever,
    route_after_supervisor,
    route_after_synthesizer,
)
from app.graph.nodes.retriever import retriever_node
from app.graph.nodes.supervisor import reset_turn_node, supervisor_node
from app.graph.nodes.synthesizer import synthesizer_node
from app.graph.nodes.verifier import verifier_node
from app.graph.state import (
    NODE_RESET,
    NODE_RETRIEVER,
    NODE_SUPERVISOR,
    NODE_SYNTHESIZER,
    NODE_VERIFIER,
    GraphState,
)

ALL_NODES: tuple[str, ...] = (
    NODE_RESET,
    NODE_SUPERVISOR,
    NODE_RETRIEVER,
    NODE_SYNTHESIZER,
    NODE_VERIFIER,
)
"""图内全部节点名，按执行顺序排列（供日志与流程图测试引用）。"""


def _bind(node_fn: Callable[..., dict], **kwargs: Any) -> Callable[[GraphState], dict]:
    """把带关键字参数的节点函数绑定成单参数节点。

    用闭包而不是 ``functools.partial``：langgraph 会检查节点函数的签名来决定
    是否注入 ``config`` / ``store``，闭包对 ``(state)`` 的签名更明确，
    partial 对象的 ``__name__`` 缺失也更容易在调试时踩坑。
    """
    if not kwargs or all(value is None for value in kwargs.values()):
        return node_fn

    def bound(state: GraphState) -> dict:
        return node_fn(state, **kwargs)

    bound.__name__ = f"bound_{getattr(node_fn, '__name__', 'node')}"
    bound.__doc__ = node_fn.__doc__
    return bound


def build_graph(
    *,
    checkpointer: Any | None = None,
    retriever: Any | None = None,
    mcp: Any | None = None,
    llm: Any | None = None,
    supervisor_llm: Any | None = None,
    verifier_llm: Any | None = None,
    max_retry: int | None = None,
) -> CompiledStateGraph:
    """构建并编译问答图。

    Args:
        checkpointer: 挂 ``MemorySaver()`` 以支持多轮会话；None 表示无状态。
        retriever: 注入检索器（测试替身 / 强制走直连通道）。None 时按
            ``settings.mcp_enabled`` 决定通道。
        mcp: 注入 MCP 客户端（测试替身 / 强制走 MCP 通道）。None 时节点内部取
            进程内单例。测试可用「指向不存在 Server 的客户端」验证降级路径。
        llm: 注入生成用 LLM（测试替身）。None 时取 ``core.llm.get_llm()``。
        supervisor_llm: 注入路由用 LLM（测试替身）。**None 时回落到 ``llm``** ——
            只传一个替身的测试不会意外打网络；两者都为空才用生产单例。
        verifier_llm: 注入核查用 LLM（测试替身）。**None 时回落到 ``llm``**，
            理由同上。
        max_retry: 回退次数上限，默认取 ``settings.max_retry``。测试用它构造
            「回退一次即停」与「回退触顶」两种场景，避免 monkeypatch 全局配置。
    """
    limit = settings.max_retry if max_retry is None else max(0, int(max_retry))
    graph: StateGraph = StateGraph(GraphState)

    graph.add_node(NODE_RESET, reset_turn_node)
    graph.add_node(NODE_SUPERVISOR, _bind(supervisor_node, llm=supervisor_llm or llm))
    graph.add_node(
        NODE_RETRIEVER, _bind(retriever_node, retriever=retriever, mcp=mcp)
    )
    graph.add_node(NODE_SYNTHESIZER, _bind(synthesizer_node, llm=llm))
    graph.add_node(
        NODE_VERIFIER,
        _bind(verifier_node, llm=verifier_llm or llm, max_retry=limit),
    )

    graph.add_edge(START, NODE_RESET)
    graph.add_edge(NODE_RESET, NODE_SUPERVISOR)

    graph.add_conditional_edges(
        NODE_SUPERVISOR, route_after_supervisor, AFTER_SUPERVISOR_MAP
    )
    graph.add_conditional_edges(
        NODE_RETRIEVER, route_after_retriever, AFTER_RETRIEVER_MAP
    )
    graph.add_conditional_edges(
        NODE_SYNTHESIZER, route_after_synthesizer, AFTER_SYNTHESIZER_MAP
    )
    graph.add_conditional_edges(
        NODE_VERIFIER, make_route_after_verifier(limit), AFTER_VERIFIER_MAP
    )

    compiled = graph.compile(checkpointer=checkpointer)
    logger.debug(
        "图已编译 | nodes={} | max_retry={} | checkpointer={} | mcp_enabled={}",
        list(ALL_NODES),
        limit,
        type(checkpointer).__name__ if checkpointer else "None",
        settings.mcp_enabled,
    )
    return compiled


_checkpointer: MemorySaver | None = None


def default_checkpointer() -> MemorySaver:
    """进程级 checkpointer 单例。

    必须是单例：多轮会话的历史要跨请求保留，每次编译新建一个 ``MemorySaver``
    等于把上一轮的上下文直接丢掉。它只活在进程内存里 —— 进程重启即清空，
    这对演示与开发是合适的行为（要持久化换 ``SqliteSaver`` / ``PostgresSaver`` 即可）。

    **已知边界**：``MemorySaver`` 不会淘汰旧线程，长跑进程里历史会持续占用内存。
    本项目单副本、会话量小，暂不做清理；若要上线，需按 session 加 TTL 清理，
    或换成自带持久化与过期策略的 checkpointer。
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
        logger.info("多轮会话已启用 | checkpointer=MemorySaver（进程内存，重启即清空）")
    return _checkpointer


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """进程内编译好的图单例，供 API 与脚本共用。

    ``settings.memory_enabled`` 决定是否挂 checkpointer。图被缓存在 lru_cache 里，
    因此运行期改这个开关不会影响已编译的实例 —— 测试要切换状态请先调
    :func:`reset_graph` 清缓存，或直接调 :func:`build_graph` 自己编译一份。
    """
    return build_graph(
        checkpointer=default_checkpointer() if settings.memory_enabled else None
    )


def reset_graph() -> None:
    """清空图缓存。供测试与「改了图结构后热重载」使用。"""
    get_graph.cache_clear()


def reset_checkpointer() -> None:
    """丢弃 checkpointer 并清空图缓存，等价于「清掉全部会话历史」。"""
    global _checkpointer
    _checkpointer = None
    reset_graph()


def graph_mermaid() -> str:
    """返回 Mermaid 流程图源码。

    阶段 2 的验收要求就是把这个输出贴进 README，因此单独暴露成函数，
    而不是让调用方去记 ``get_graph().get_graph().draw_mermaid()`` 这种
    两层 get_graph 的写法。
    """
    return get_graph().get_graph().draw_mermaid()


__all__ = [
    "ALL_NODES",
    "build_graph",
    "get_graph",
    "reset_graph",
    "default_checkpointer",
    "reset_checkpointer",
    "graph_mermaid",
]
