"""条件边路由函数（阶段 2 建立，阶段 3 扩展自纠正回路，阶段 5 引入并行分发）。

图里的分支决策全部集中在本模块，节点函数因此保持「只写状态、不做跳转判断」
的单一职责。阶段 5 之后共四处：

1. :func:`route_after_supervisor` —— 走检索还是直接生成；复合问题在此**并行分发**。
2. :func:`route_after_retriever` —— 检索完是继续生成还是收工。
3. :func:`route_after_synthesizer` —— 生成完要不要送去核查（阶段 3 新增）。
4. :func:`make_route_after_verifier` —— 核查不通过则回退重检，否则收工（阶段 3 新增）。

约定
----
路由函数是**纯函数**：只读 state、返回节点名（或 ``Send`` 列表），绝不修改 state。
例如 ``retry_count += 1`` 必须写在 verifier 节点里 —— 路由函数可能被多次调用，
且返回值只用于选边，任何副作用都会丢失或重复执行。

阶段 5 的并行分发把这条约定推到了「必须严格执行」的程度：实测条件边函数在
fan-out 时**每个分支各被调用一次**（3 个子问题 ⇒ 调用 3 次），
任何副作用都会被放大三倍。

为什么 fan-out 用 ``Send`` 而不是「让 retriever 自己循环检索」
----------------------------------------------------------
``Send`` 让 N 个子问题变成 N 个**独立任务**，由 LangGraph 调度器在同一超步内
并发执行，并在全部落定之后才推进到下游 —— 也就是天然的 map-reduce：
``supervisor`` 发起 map，``retriever`` 是 mapper，汇合点是 ``synthesizer``。
若改成在 retriever 内部顺序循环，两路检索的延迟会相加（每路约 200ms，
且各自含一次 embedding 网络请求），而并行版本的总延迟约等于最慢的一路。

实测确认的三条语义（``langgraph 0.2.60``）：

- N 个 Send ⇒ 目标节点执行 N 次，下游汇合节点**只执行一次**，
  且看到的是全部 N 支写完之后的合并状态。
- 写入顺序等于 Send 的派发顺序，与各分支实际完成的先后无关，
  因此 reducer 的累加次序是确定、可复现的。
- 并行分支**不得**写没有 reducer 的通道，否则抛
  ``InvalidUpdateError: Can receive only one value per step``。这条约束直接
  决定了 ``GraphState`` 里几个标量字段必须升级为 reducer 通道（见 state.py 约束 4）。

为什么 ``route_after_verifier`` 是工厂函数
------------------------------------------
阶段 2 的两个路由函数都是模块级纯函数，阶段 3 把 ``max_retry`` 提成参数，原因是
**可测性**：闸门阈值必须能在测试里独立控制，才能构造出「回退 1 次即停」与
「回退到上限」两种场景。若直接在函数体里读 ``settings.max_retry``，测试就只能靠
monkeypatch 改全局配置，用例之间会互相污染。

模块级仍保留一个默认实例 ``route_after_verifier``，参数取自 ``settings.max_retry``，
供直接调用与文档引用；``builder`` 则用工厂按传入值构造。
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END
from langgraph.types import Send

from app.core.config import settings
from app.graph.state import (
    NODE_RETRIEVER,
    NODE_SYNTHESIZER,
    NODE_VERIFIER,
    ROUTE_DIRECT,
    ROUTE_RETRIEVE,
    VERDICT_FAIL,
    GraphState,
)

MIN_SUB_QUERIES_FOR_FANOUT = 2
"""触发并行分发的最小子问题数。只有 1 条等于没拆解，走普通边省一次调度开销。"""


def effective_sub_queries(state: GraphState) -> list[str]:
    """取出真正可用于并行检索的子问题（去空白、去重、保序）。

    单独成函数是为了让「这一轮要检索几次」可被直接断言，
    不必构造整张图去数 ``Send`` 的个数。
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in state.get("sub_queries") or []:
        text = (raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_fanout_payload(state: GraphState, sub_query: str) -> GraphState:
    """构造一条子问题分支的输入状态。

    为什么必须显式搬运这么多字段
    ----------------------------
    ``Send`` 的第二个参数是**整体替换**该分支的输入状态，而不是与父状态合并。
    分支看不见没被显式传入的字段，因此 retriever 需要的一切都要在这里列全：
    检索词、top_k、时效开关，以及「已经有哪些块」—— 少了最后一项，
    分支就无法判断自己返回的哪些块属于新增。
    """
    return GraphState(
        query=state.get("query") or "",
        session_id=state.get("session_id") or "",
        # 子问题即本分支的检索词：retriever 只认 search_query 一个字段，
        # 因此「拆解」这个动作对节点完全透明。
        search_query=sub_query,
        top_k=state.get("top_k"),
        include_expired=bool(state.get("include_expired", False)),
        with_answer=bool(state.get("with_answer", True)),
        # 已召回集合作为去重基线（供分支内部去重）
        retrieved_chunks=list(state.get("retrieved_chunks") or []),
        retrieval_attempts=int(state.get("retrieval_attempts") or 0),
        retry_count=int(state.get("retry_count") or 0),
        route=state.get("route") or "",
        sub_queries=list(state.get("sub_queries") or []),
        # 分支据此知道「我不是唯一的检索者」
        fanout=True,
    )


def route_after_supervisor(state: GraphState) -> str | list[Send]:
    """supervisor 之后：走检索、绕过检索直接生成，还是并行分发。

    三种出口：

    - **direct** → synthesizer（寒暄 / 元问题，知识库里没有可查的内容）。
    - **单一检索词** → retriever（普通边，即阶段 2~4 的原路径）。
    - **多个子问题** → ``list[Send]``，每个子问题一路并行检索。

    为什么单子问题保留普通边、不统一用 Send
    --------------------------------------
    单路时 ``Send`` 只多一层调度、没有任何并行收益，却让图结构从「一条主干」
    变成「永远是 map-reduce」。保留普通边意味着单问题场景的图与阶段 4 完全一致，
    回归风险最小 —— 新能力只在确实需要它的输入上生效。

    ``with_answer=False``（评估模式）强制走检索：该模式的语义是「只测检索、
    不生成」，因此无论路由判定如何都必须检索，且不进入生成节点。该模式下
    supervisor 不调用 LLM（见 ``nodes/supervisor.py``），``sub_queries`` 恒为空，
    因此评估路径永远是单路检索 —— 这也保证了「评估零 LLM 配额」的契约不变。
    """
    sub_queries = effective_sub_queries(state)
    if len(sub_queries) >= MIN_SUB_QUERIES_FOR_FANOUT:
        fanout: list[Send] | None = [
            Send(NODE_RETRIEVER, build_fanout_payload(state, sub))
            for sub in sub_queries
        ]
    else:
        fanout = None

    if not state.get("with_answer", True):
        return fanout or NODE_RETRIEVER
    if state.get("route") == ROUTE_RETRIEVE:
        return fanout or NODE_RETRIEVER
    return NODE_SYNTHESIZER


def route_after_retriever(state: GraphState) -> str:
    """retriever 之后：评估模式在此收工，否则进入生成。

    回退路径同样经过这里：第二轮检索之后回到 synthesizer 重新生成，
    因此首检与补检共用同一条边，无需为自纠正单独分支。

    并行分发时本函数会被**每个分支各调用一次**，返回值都是同一个节点名，
    由 LangGraph 去重成一次执行（实测确认）。因此这里必须保持纯函数。
    """
    if not state.get("with_answer", True):
        return END
    return NODE_SYNTHESIZER


def route_after_synthesizer(state: GraphState) -> str:
    """synthesizer 之后：是否送去核查。

    四种情况直接收工，它们的共性是「核查没有意义」：

    - ``with_answer=False``：评估模式在 retriever 之后就结束了，兜底判断。
    - ``route == direct``：寒暄 / 元问题没有参考资料，核查员无从比对。
    - 没有召回块：合成器走的是固定拒答分支，答案里没有任何事实性断言。
    - ``error`` 非空或 ``answer`` 为空：检索失败或生成失败，没有可核查的正文。

    其余情况一律核查。这里刻意不做「答案太短就不核查」之类的省流优化 ——
    忠实度是本系统的核心指标，一次核查的 token 成本换一个可量化的指标，
    这个交换在任何情况下都划算。
    """
    if not state.get("with_answer", True):
        return END
    if state.get("route") == ROUTE_DIRECT:
        return END
    if not state.get("retrieved_chunks"):
        return END
    if (state.get("error") or "").strip():
        return END
    if not (state.get("answer") or "").strip():
        return END
    return NODE_VERIFIER


def make_route_after_verifier(
    max_retry: int | None = None,
) -> Callable[[GraphState], str]:
    """构造 verifier 之后的路由函数，``max_retry`` 为回退次数上限。

    三道闸门，任一不满足即收工：

    1. ``verdict != fail`` → 核查通过，正常输出。
    2. ``retry_count >= max_retry`` → **防死循环闸门**。``retry_count`` 是
       **已实际执行**的回退次数（由 retriever 在补检时递增；verifier 只提建议、
       不递增），因此这里比较的就是「已经回退了几次」，语义与响应字段一致。
       ``max_retry=0`` 即完全禁用回退。
    3. ``search_query`` 为空 → 核查判失败但没能重组出检索词，回退无方向。
       （retriever 对空检索词会回落到用户原话，那等于原问题重检，是空转，
       因此在这一层就拦住，不进 retriever。）

    回退轮**不走并行分发**：补检的检索词由 verifier 用缺口重组而成，
    是一条完整的、已经包含了各子问题的合并语义的查询词。再按子问题拆一次
    只会重复召回同一批块。
    """
    limit = settings.max_retry if max_retry is None else max(0, int(max_retry))

    def route_after_verifier(state: GraphState) -> str:
        if state.get("verdict") != VERDICT_FAIL:
            return END
        if int(state.get("retry_count") or 0) >= limit:
            return END
        if not (state.get("search_query") or "").strip():
            return END
        return NODE_RETRIEVER

    route_after_verifier.__name__ = f"route_after_verifier(max_retry={limit})"
    return route_after_verifier


route_after_verifier = make_route_after_verifier()
"""默认实例，阈值取 ``settings.max_retry``。builder 会按需用工厂重建。"""


# 供 builder 注册条件边时使用：显式映射表让图结构在代码里一目了然，
# 也避免路由函数返回未注册的节点名（langgraph 会直接抛错）。
# 注意：``Send`` 的目标节点也必须在映射表里，否则 langgraph 拒绝注册。
AFTER_SUPERVISOR_MAP: dict[str, str] = {
    NODE_RETRIEVER: NODE_RETRIEVER,
    NODE_SYNTHESIZER: NODE_SYNTHESIZER,
}

AFTER_RETRIEVER_MAP: dict[str, str] = {
    NODE_SYNTHESIZER: NODE_SYNTHESIZER,
    END: END,
}

AFTER_SYNTHESIZER_MAP: dict[str, str] = {
    NODE_VERIFIER: NODE_VERIFIER,
    END: END,
}

AFTER_VERIFIER_MAP: dict[str, str] = {
    NODE_RETRIEVER: NODE_RETRIEVER,
    END: END,
}


__all__ = [
    "MIN_SUB_QUERIES_FOR_FANOUT",
    "effective_sub_queries",
    "build_fanout_payload",
    "route_after_supervisor",
    "route_after_retriever",
    "route_after_synthesizer",
    "make_route_after_verifier",
    "route_after_verifier",
    "AFTER_SUPERVISOR_MAP",
    "AFTER_RETRIEVER_MAP",
    "AFTER_SYNTHESIZER_MAP",
    "AFTER_VERIFIER_MAP",
]
