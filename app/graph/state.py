"""GraphState：LangGraph 状态机的中枢数据契约（阶段 2）。

本文件是各节点之间唯一的耦合面。节点只读写这里的字段，不互相 import，
因此新增阶段（3 核查循环 / 5 多 Agent）时，改动集中在本文件而不是散落到各节点。

四个实测确认的 LangGraph 0.2.60 约束
------------------------------------
1. **累加字段必须显式声明 reducer**。否则后一个节点的返回值会整体覆盖前一个：
   实测两个节点分别写 ``{"plain": ["1"]}`` 与 ``{"plain": ["2"]}``，
   无 reducer 时终态是 ``["2"]``，有 reducer 时是 ``["1","2"]``。
2. **reducer 必须定义在模块级**。langgraph 构造 StateGraph 时用
   ``get_type_hints(schema, include_extras=True)`` 在 **模块全局作用域** 解析注解，
   把 reducer 写成局部函数会直接抛 ``NameError``。
3. **``invoke({})`` 会被拒绝**。即使所有字段都是可选的，空字典也会让
   ``__start__`` 无通道可写，抛 ``InvalidUpdateError: Must write to at least one of [...]``。
   因此统一用 :func:`initial_state` 构造入参，不要直接传空字典。
4. **并行分支不得写「无 reducer 的通道」**。阶段 5 用 ``Send`` 把子问题并行分发到
   ``retriever`` 后，实测 N 个分支同时写一个普通通道会抛
   ``InvalidUpdateError: At key 'errors': Can receive only one value per step.``
   带 reducer 的通道则接受同一超步内的多次写入，且按 Send 的派发顺序依次应用
   （顺序确定、可复现）。因此 ``retrieval_attempts`` / ``retry_count`` / ``error``
   在本阶段从普通通道升级为 reducer 通道，三者的 reducer 都恰好满足
   「多个分支各自主张一个值」的语义：取最大轮次、保留首个错误。

为什么需要 None 重置
--------------------
挂上 checkpointer 后状态跨轮次保留：实测 MemorySaver 下第二轮会把第一轮的事件再累加
一遍（``['a','b']`` -> ``['a','b','a','b']``）。每轮入口必须清零累加字段，而
``operator.add`` 无法表达「清空」，因此这里用 :func:`append_or_reset`：
**update 为 None 即清空**。阶段 5 的多轮会话、阶段 6 的事件流都依赖这一语义。
本阶段新增的三个 reducer 统一沿用同一约定（``None`` = 重置回初始值），
这样 ``reset_turn`` 的清零写法无需为每个字段记一套特例。
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from app.schemas.query import Citation, RetrievedChunk

# ---------------- 路由取值 ----------------

ROUTE_RETRIEVE = "retrieve"
"""需要走知识库检索。"""
ROUTE_DIRECT = "direct"
"""绕过检索，直接生成（仅寒暄 / 元问题）。"""

# ---------------- 核查结论取值（阶段 3） ----------------

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"

# ---------------- 节点名 ----------------
# 图里的节点名与 SSE 事件里的 node 字段共用同一组常量，避免两处字符串漂移。

NODE_RESET = "reset_turn"
NODE_SUPERVISOR = "supervisor"
NODE_RETRIEVER = "retriever"
NODE_SYNTHESIZER = "synthesizer"
NODE_VERIFIER = "verifier"


def append_or_reset(current: list | None, update: list | None) -> list:
    """累加型字段的 reducer：正常追加，``None`` 表示清空。

    ``update is None`` 是第一等语义（清空），不是「无更新」——
    要表达「本节点不改这个字段」，直接不在返回值里出现该键即可。
    """
    if update is None:
        return []
    return [*(current or []), *update]


def _chunk_key(chunk: Any) -> str:
    """一条召回块的去重键。同时兼容 pydantic 模型与字典两种形态。

    MCP 通道把块以 JSON 字典送回，直连通道给的是 ``RetrievedChunk`` 模型，
    两种形态都会流经 reducer，因此不能假设只有一种。
    """
    if isinstance(chunk, dict):
        return str(chunk.get("chunk_id") or "")
    return str(getattr(chunk, "chunk_id", "") or "")


def append_chunks(
    current: list[RetrievedChunk] | None,
    update: list[RetrievedChunk] | None,
) -> list[RetrievedChunk]:
    """``retrieved_chunks`` 的 reducer：追加并按 ``chunk_id`` 去重，``None`` 表示清空。

    为什么去重必须在 reducer 兜底，而不是只靠 retriever
    --------------------------------------------------
    retriever 只能看见**自己这一支**的状态。阶段 5 并行检索时，两条子问题
    完全可能命中同一个块（「Q2 返点比例」与「Q2 相比 Q1 的变化」都会召回
    Q2 政策正文）。若不在汇合处去重，同一份证据会以两份的身份进入上下文与
    引用列表 —— 既浪费 prompt 预算，也会让 ``citations`` 出现重复条目。
    reducer 是所有分支写入的唯一汇聚点，因此兜底在这里；
    retriever 内部的去重依然保留，作用是避免同一分支重复计算。
    """
    if update is None:
        return []
    merged = list(current or [])
    seen = {key for key in map(_chunk_key, merged) if key}
    for chunk in update:
        key = _chunk_key(chunk)
        if key:
            if key in seen:
                continue
            seen.add(key)
        merged.append(chunk)
    return merged


def keep_max(current: int | None, update: int | None) -> int:
    """单调计数的 reducer：取较大值，``None`` 表示归零。

    用于 ``retrieval_attempts`` / ``retry_count``。并行分支各自都会主张
    「这是第 1 轮检索」，取最大值即真实轮次；回退重检只有单分支写入，
    此时 max 退化为直接赋值，与阶段 3 的语义完全一致。
    """
    if update is None:
        return 0
    return max(int(current or 0), int(update))


def merge_error(current: str | None, update: str | None) -> str:
    """``error`` 的 reducer：保留首个非空错误，``None`` 表示清空。

    并行分支可能多支同时失败（例如 MCP Server 整体不可用）。这里刻意**只保留
    第一条**而不拼接：写入顺序即 Send 派发顺序（确定、可复现），而多支失败
    通常同因同源，拼接只会把同一句话重复若干遍。分支返回空串是「本支无错」的
    正常表态，不得覆盖已存在的错误。
    """
    if update is None:
        return ""
    if (current or "").strip():
        return current
    return update


class GraphState(TypedDict, total=False):
    """图状态。

    ``total=False`` 让每个字段都成为可选键 —— 节点只需返回自己关心的字段。
    带 ``Annotated`` 的字段是累加语义（见模块 docstring 约束 1）。
    """

    # ---------- 输入（由 initial_state 填充，全程只读） ----------
    query: str
    """用户原始问题。永不改写 —— 回退重检索时改的是 search_query。"""
    session_id: str
    """会话 ID。阶段 2 仅透传，阶段 5 接入 checkpointer 后作为 thread_id。"""
    top_k: int | None
    """覆盖 .env 的 TOP_K；None 表示用默认值。"""
    include_expired: bool
    """是否纳入已过期文档。默认 False，仅评估对照时置 True。"""
    with_answer: bool
    """False 时只检索不生成，用于阶段 7 不消耗 LLM 配额地评估检索指标。"""

    # ---------- supervisor 决策 ----------
    route: str
    """ROUTE_RETRIEVE | ROUTE_DIRECT。"""
    route_reason: str
    """人类可读的路由理由，直接进日志与 /query 响应，便于解释与排错。"""

    # ---------- 检索 ----------
    search_query: str
    """实际送去检索的文本。默认等于 query；阶段 3 回退时由 verifier
    用 unsupported_claims 重组成新 query 写入这里，实现「换问题重检」
    而不是拿原问题重检。"""
    retrieved_chunks: Annotated[list[RetrievedChunk], append_chunks]
    """已召回块。累加语义并按 ``chunk_id`` 去重：回退重检索时追加新增块，
    让合成器掌握两轮证据；阶段 5 的并行分支结果也在同一个 reducer 里汇合。
    节点返回的必须是**增量**（新块）—— retriever 内部的去重只是省算力，
    真正的兜底在 reducer（见 :func:`append_chunks`）。"""
    retrieval_attempts: Annotated[int, keep_max]
    """检索轮次。阶段 2 恒为 1，阶段 3 随回退递增，用于区分首检与补检。
    阶段 5 起改为 reducer 通道：并行分支各自主张「第 1 轮」，取最大即真实轮次。"""

    # ---------- 生成 ----------
    answer: str
    citations: list[Citation]
    """答案中实际引用到的块。已与召回集合对齐，模型编造的 id 会被丢弃。"""
    llm_model: str
    """实际用于生成的模型名；未调用 LLM（如拒答）时为空字符串。"""

    # ---------- 核查与自纠正（阶段 3 写入，阶段 2 仅占位） ----------
    verdict: str
    """VERDICT_PASS | VERDICT_FAIL。"""
    unsupported_claims: Annotated[list[str], append_or_reset]
    """无法在召回块中找到依据的断言。verifier 返回的必须是**增量**。"""
    retry_count: Annotated[int, keep_max]
    """已回退次数。阶段 3 的防死循环闸门：>= max_retry 即强制输出。
    阶段 5 起是 reducer 通道，原因同 ``retrieval_attempts``：
    并行分支不存在于回退轮，max 在此退化为直接赋值。"""

    # ---------- 多 Agent（阶段 5 写入） ----------
    sub_queries: Annotated[list[str], append_or_reset]
    """复合问题拆解出的子问题。长度 ≥2 时 supervisor 用 Send 并行分发到 retriever。
    由 supervisor 单点写入（唯一写者），因此不需要额外的去重 reducer。"""
    fanout: bool
    """**仅存在于 Send 的 payload 里**的标记，图级状态中不承载语义。

    Send 的 payload 会整体替换该分支看到的输入状态，因此分支需要知道自己
    「不是唯一的检索者」。retriever 据此把事件文案区分为子问题检索，
    并据此回避对标量通道的写入（详见 ``nodes/retriever.py`` 文档）。"""

    # ---------- 执行事件（阶段 6 消费） ----------
    events: Annotated[list[dict[str, Any]], append_or_reset]
    """节点执行轨迹：[{"node", "msg", "ts"}]。
    ts 用 time.time() 浮点秒 —— 阶段 6 需要它算节点耗时。"""

    # ---------- 错误 ----------
    error: Annotated[str, merge_error]
    """非致命错误描述。检索失败或生成失败时填充，空串表示无错误。
    设计为「不抛异常」：单节点失败不应让整轮问答失去已获得的信息。
    阶段 5 起是 reducer 通道：并行分支可能多支同时失败，取首个非空者
    （写入顺序 = Send 派发顺序，确定可复现）。"""


ACCUMULATING_KEYS: tuple[str, ...] = (
    "retrieved_chunks",
    "unsupported_claims",
    "sub_queries",
    "events",
)
"""带 reducer 的字段。每轮入口由 reset_turn 节点用 None 清零。"""


def make_event(node: str, msg: str) -> dict[str, Any]:
    """构造一条执行事件（阶段 6 的 SSE 数据源）。"""
    return {"node": node, "msg": msg, "ts": time.time()}


def initial_state(
    query: str,
    *,
    session_id: str | None = None,
    top_k: int | None = None,
    include_expired: bool = False,
    with_answer: bool = True,
) -> GraphState:
    """构造一轮问答的初始状态。

    始终至少包含一个键（``query``），以规避模块 docstring 约束 3 的
    ``invoke({})`` 报错问题；同时把默认值收敛到一处，避免调用方各写一遍。
    """
    return GraphState(
        query=query,
        session_id=session_id or "",
        top_k=top_k,
        include_expired=include_expired,
        with_answer=with_answer,
        events=[],
    )


__all__ = [
    "GraphState",
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
    "append_or_reset",
    "make_event",
    "initial_state",
]
