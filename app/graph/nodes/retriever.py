"""Retriever 节点：把检索能力封装为图节点。

阶段 4 起有两条检索通道
----------------------
::

    MCP 通道（默认）      retriever 节点 ──call_tool────► chroma_server 子进程 ──► Chroma
    直连通道（可切换）    retriever 节点 ──HybridRetriever─────────────────► Chroma

选择由 ``settings.mcp_enabled`` 决定。两条通道**共用同一套检索算法** ——
chroma_server 内部就是把 ``HybridRetriever`` 原样暴露成 MCP 工具，
所以阶段 1 的 Hit@3 / MRR 结论对两条通道都成立，A/B 对比的是进程边界带来的
开销与解耦收益，而不是算法差异。

保留直连通道的三个理由
--------------------
1. **可回归**：MCP 通道出问题时，一个开关就能退回阶段 3 的行为，便于二分定位。
2. **可测试**：单元测试注入替身即可，不必为每个用例 spawn 子进程。
3. **可对比**：阶段 7 的评估需要量化 MCP 引入的额外延迟，这需要基线。

降级策略（阶段 4 的验收点）
--------------------------
MCP 不可用时（未启动、Server 崩溃、调用超时）**不抛异常**，而是把原因写进
``error`` 并返回空增量。合成器拿不到块会走拒答分支 —— 整轮问答因此优雅降级为
「暂时无法确认」而不是 500。这是「关掉 chroma_server 后 /query 仍应返回 200」
的实现方式。

阶段 5 起的第二种输入形态（fan-out）
-----------------------------------
``supervisor`` 判定问题为复合问题时会拆出多个子问题，图通过 ``Send`` 把**每个
子问题**投递给本节点，形成 N 个并发执行的独立任务。本节点因此有两条进入路径：

- **普通路径**：拿到完整状态，``search_query`` 是唯一检索词。
- **fan-out 路径**：拿到的是 ``edges.build_fanout_payload`` 构造的精简状态，
  ``search_query`` 即本分支负责的子问题，且带 ``fanout=True`` 标记。

节点本身**不需要区分两者**：它只认 ``search_query`` 一个字段，拆解对它完全透明
（与阶段 3 的自纠正循环同一个套路）。唯一用到 ``fanout`` 的地方是事件文案，
让执行轨迹能看出「这是子问题检索」而不是唯一的一次首检。

值得记下的两点约束：

1. ``Send`` 的 payload 是**整体替换**而不是合并，分支只拥有被显式传入的字段。
2. fan-out 分支只能写带 reducer 的通道。本节点写的五个通道里，
   ``retrieved_chunks`` / ``events`` 本来就是累加通道，
   ``retrieval_attempts`` / ``retry_count`` / ``error`` 在阶段 5 升级为 reducer
   通道（取最大轮次、保留首个错误）。因此这里**无需任何并发特判**就能安全并行 ——
   把「能不能并发」从节点逻辑挪进状态契约，是本阶段最重要的设计取舍。

为什么检索词用 search_query 而不是 query
--------------------------------------
state 里两个问题字段职责不同：

- ``query``：用户原话，全程只读，用于生成答案与评估对齐。
- ``search_query``：真正送去检索的文本。supervisor 初始化为 query；
  阶段 3 回退时由 verifier 用 ``unsupported_claims`` 重组成新问题后写入。

阶段 3 回退用缺口重组成新检索词（而不是原问题重检 —— 那只会得到同一批块，
退化成无意义空转）。本节点不感知这件事：它只认 ``search_query`` 一个字段，
首检与回退共用同一条代码路径。

为什么返回增量而不是全量
----------------------
``retrieved_chunks`` 声明了累加 reducer。若本节点每次返回完整结果，第二轮回退时
同一批块会被追加两次，合成器会看到重复证据。因此这里按 chunk_id 去重，
只返回尚未出现过的块。

``retry_count`` 的递增为什么放在这里
----------------------------------
``retry_count`` 会随响应暴露给调用方，语义必须是**实际已执行的回退次数**。
verifier 只提出回退建议（它无法知道闸门阈值），建议是否被采纳由
``edges.route_after_verifier`` 按 ``max_retry`` 裁定。若在 verifier 里递增，
被闸门拦下的那次建议也会被计入 —— ``max_retry=0`` 时就会出现「回退 0 次却记 1 次」
的歧义。本节点是唯一知道「这次检索确实是重检」的地方（``attempt > 1``），
因此递增在这里，语义与事实严格一致。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.graph.state import NODE_RETRIEVER, GraphState, make_event
from app.mcp_client import MCPToolClient, get_default_mcp
from app.rag.retriever import HybridRetriever
from app.schemas.query import RetrievedChunk

TOOL_SEARCH = "search_documents"
"""chroma_server 暴露的检索工具名。改名需同步 mcp_servers/chroma_server.py。"""


@lru_cache(maxsize=1)
def get_default_retriever() -> HybridRetriever:
    """进程内检索器单例（直连通道）。

    单例的意义是让 top_k / rrf_k 只解析一次配置。语料缓存（BM25 的 IDF 依赖
    全量语料）本身是 ``rag.retriever`` 模块级的，即使构造多个实例也共享同一份
    缓存，因此这里不做全局状态，只是省去重复解析配置的开销。

    放在节点模块而不是 api/deps.py：图既被 API 层调用，也被评估脚本调用，
    单例应该属于图这一层；api/deps.py 反向委托到这里，保证只有一个实例。
    """
    return HybridRetriever()


def get_mcp_client() -> MCPToolClient:
    """MCP 客户端单例（MCP 通道）。转发到 mcp_client 层，保证全局只有一个实例。"""
    return get_default_mcp()


def _to_chunk(item: dict[str, Any]) -> RetrievedChunk | None:
    """把 MCP 返回的一条字典转成 RetrievedChunk。

    只取模型已知字段：Server 端在失败时会返回 ``{"error": ..., "chunk_id": ""}``
    这类诊断条目，它们没有完整的 chunk 字段，必须被过滤掉而不是让整轮报错。
    """
    allowed = RetrievedChunk.model_fields.keys()
    payload = {key: value for key, value in item.items() if key in allowed}
    try:
        return RetrievedChunk(**payload)
    except Exception:  # noqa: BLE001 - 单条脏数据不应中断整轮检索
        return None


def _search_via_mcp(
    client: MCPToolClient, query: str, top_k: int | None, include_expired: bool
) -> tuple[list[RetrievedChunk] | None, str]:
    """经 MCP 通道检索。

    Returns:
        (hits, error)。hits 为 None 表示通道不可用（调用方据此降级）；
        为空列表表示通道正常但无命中。
    """
    result = client.call_tool(
        TOOL_SEARCH,
        {
            "query": query,
            "top_k": int(top_k) if top_k else 0,
            "include_expired": bool(include_expired),
        },
    )
    if not result.ok:
        return None, result.error

    chunks = [chunk for chunk in map(_to_chunk, result.as_items()) if chunk is not None]
    return chunks, ""


def _search_via_local(
    retriever: HybridRetriever, query: str, top_k: int | None, include_expired: bool
) -> list[RetrievedChunk]:
    """经直连通道检索。异常由调用方捕获处理。"""
    return retriever.search(query, top_k=top_k, include_expired=include_expired)


def retriever_node(
    state: GraphState,
    *,
    retriever: HybridRetriever | None = None,
    mcp: MCPToolClient | None = None,
) -> dict:
    """检索节点。

    通道选择的优先级（前者覆盖后者）：
    1. ``retriever`` 注入 → 强制直连。测试替身走这条路。
    2. ``mcp`` 注入 → 强制 MCP。测试用可指向不存在的 Server 以验证降级。
    3. 二者皆无 → 按 ``settings.mcp_enabled`` 决定：true 用进程内 MCP 单例，
       false 用进程内直连单例。

    异常处理策略：检索失败**不抛出**，而是把错误写进 ``error`` 并返回空增量。
    整轮问答因此仍能继续 —— 合成器拿不到块会走拒答分支。
    """
    search_query = (state.get("search_query") or state.get("query") or "").strip()
    attempt = int(state.get("retrieval_attempts") or 0) + 1
    retry_count = int(state.get("retry_count") or 0)
    existing = {chunk.chunk_id for chunk in (state.get("retrieved_chunks") or [])}

    # attempt > 1 即「这不是首检，而是一次回退重检」，此刻回退才算真正发生。
    retry_delta = {"retry_count": retry_count + 1} if attempt > 1 else {}

    if not search_query:
        return {
            "retrieval_attempts": attempt,
            **retry_delta,
            "error": "检索词为空，已跳过检索",
            "events": [make_event(NODE_RETRIEVER, "检索词为空，跳过检索")],
        }

    top_k = state.get("top_k")
    include_expired = bool(state.get("include_expired", False))

    # ---------------- 通道选择 ----------------
    use_mcp = mcp is not None or (retriever is None and settings.mcp_enabled)
    channel = "MCP" if use_mcp else "直连"
    hits: list[RetrievedChunk] | None = None

    try:
        if use_mcp:
            client = mcp or get_mcp_client()
            if mcp is None and not client.ready:
                # 惰性兜底：评估脚本与「直接调用 build_graph()」的入口没有
                # FastAPI lifespan 来启动单例，不启动就会永远静默降级
                # —— 这个坑在阶段 4 的真实验证里实际踩到过。
                # 只发生在第一次检索，失败会落进下面的降级分支。
                #
                # 判据是 ready 而非 running（阶段 5 修正）：running 在后台事件循环
                # 起来时为真，但各 Server 的工具此刻可能还没进路由表；阶段 5 的
                # Send fan-out 会让 N 个分支同时走到这里，用 running 判定会带着
                # 空路由表去调用，整轮复合问题降级。ensure_started 现已并发安全
                # （leader/follower），重复调用只会一起等同一个握手完成事件。
                client.ensure_started()
            hits, channel_error = _search_via_mcp(
                client, search_query, top_k, include_expired
            )
            if hits is None:
                return {
                    "retrieval_attempts": attempt,
                    **retry_delta,
                    "error": f"MCP 检索不可用：{channel_error}",
                    "events": [
                        make_event(
                            NODE_RETRIEVER,
                            f"MCP 检索不可用（{channel_error}），本轮降级为无参考资料",
                        )
                    ],
                }
        else:
            engine = retriever or get_default_retriever()
            hits = _search_via_local(engine, search_query, top_k, include_expired)

    except Exception as exc:  # noqa: BLE001 - 单节点失败不应中断整轮问答
        return {
            "retrieval_attempts": attempt,
            **retry_delta,
            "error": f"检索失败：{type(exc).__name__}: {exc}",
            "events": [
                make_event(
                    NODE_RETRIEVER,
                    f"{channel}检索失败：{type(exc).__name__}，已降级为无参考资料",
                )
            ],
        }

    delta = [hit for hit in hits if hit.chunk_id not in existing]


    if state.get("fanout"):
        # 子问题检索：用子问题文本替代「第 N 轮」标签，让执行轨迹能分辨
        # 「唯一的一次首检」与「三路并行中的一路」。截断到 20 字，避免事件膨胀。
        round_label = f"子问题「{search_query[:20]}」检索"
    else:
        round_label = f"第 {attempt} 轮检索" if attempt > 1 else "检索"

    return {
        "retrieved_chunks": delta,
        "retrieval_attempts": attempt,
        **retry_delta,
        "error": "",
        "events": [
            make_event(
                NODE_RETRIEVER,
                (
                    f"经 MCP 工具 {TOOL_SEARCH} {round_label}："
                    f"命中 {len(hits)} 个片段（新增 {len(delta)}）"
                    if use_mcp
                    else f"{round_label}：命中 {len(hits)} 个片段（新增 {len(delta)}）"
                ),
            )
        ],
    }


__all__ = [
    "TOOL_SEARCH",
    "get_default_retriever",
    "get_mcp_client",
    "retriever_node",
]
