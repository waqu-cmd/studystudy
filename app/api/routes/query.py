"""POST /query —— 同步问答（阶段 2 起改由图执行，阶段 5 起支持多轮会话）。

阶段 1 的做法是「路由函数里手写：检索 → 调 LLM → 提取引用」；阶段 2 把这套
流程搬进 LangGraph，本路由退化为**薄适配层**：把请求体转成初始状态、调图、
把终态映射成响应体。

薄适配层的收益在阶段 3 得到验证：自纠正循环（核查节点、两条条件边、
检索词改写）全部落在 ``graph/`` 内，本文件只多了 3 个响应字段的映射，
没有任何流程改动。阶段 5 的并行子问题分发同样只多了一个 ``sub_queries`` 字段 ——
**「新增能力只改图、不改接口」是本项目一路验证下来的结构性收益。**

阶段 5 起本文件多承担一件事：把 ``session_id`` 翻译成 checkpointer 需要的
``thread_id``。这是路由层与图之间唯一的协议（见 :func:`_thread_config`）。

阶段 6 之后
-----------
SSE 流式端点会复用同一份状态契约（``state["events"]``），但走
``astream`` 而不是 ``invoke``，因此本文件与流式端点互不干扰。
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends
from langgraph.graph.state import CompiledStateGraph

from app.api.deps import get_graph
from app.core.config import settings
from app.core.logging import logger
from app.graph.state import initial_state
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


def _thread_config(session_id: str | None) -> dict | None:
    """把会话 ID 翻译成 LangGraph 的调用配置。

    阶段 5 起图挂了 MemorySaver，``thread_id`` 即会话标识：同一 session_id 的
    多轮请求共享状态，不同 session_id 之间完全隔离。

    为什么没传 session_id 时不直接省略 config
    ---------------------------------------
    挂了 checkpointer 的图**必须**拿到 thread_id，否则 langgraph 直接抛
    ``ValueError``。而 ``session_id`` 在本项目的请求体里一直是可选字段
    （阶段 2 起就没强制过），因此无 ID 时退化为「一次性匿名会话」：
    用 uuid 生成只属于本次请求的 thread_id。语义上等价于无状态，
    同时满足 checkpointer 的硬要求。

    **为什么不共用一个固定的匿名 ID**：那样所有匿名请求会互相串状态，
    第一个人的追问记忆会泄漏给第二个人 —— 这是比「多占一点内存」严重得多的问题。
    """

    if not settings.memory_enabled:
        return None
    thread_id = (session_id or "").strip() or f"anon-{uuid.uuid4().hex}"
    return {"configurable": {"thread_id": thread_id}}


@router.post("/query", response_model=QueryResponse, summary="知识库问答（同步）")
def query(
    request: QueryRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> QueryResponse:
    started = time.perf_counter()

    state = initial_state(
        request.query,
        session_id=request.session_id,
        top_k=request.top_k,
        include_expired=request.include_expired,
        with_answer=request.with_answer,
    )

    try:
        result = graph.invoke(state, config=_thread_config(request.session_id))
    except Exception as exc:  # noqa: BLE001 - 图级失败也要给出结构化响应
        logger.exception("图执行失败 | query={!r}", request.query[:30])
        return QueryResponse(
            query=request.query,
            include_expired=request.include_expired,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error=f"图执行失败：{type(exc).__name__}: {exc}",
        )

    chunks = result.get("retrieved_chunks") or []
    citations = result.get("citations") or []
    answer = result.get("answer") or ""
    sub_queries = list(result.get("sub_queries") or [])
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "query 完成 | query={!r} | route={} | sub_queries={} | hits={} | attempts={} | "
        "verdict={} | retry={} | answer={}ch | citations={} | elapsed={}ms",
        request.query[:30],
        result.get("route", ""),
        len(sub_queries),
        len(chunks),
        result.get("retrieval_attempts", 0),
        result.get("verdict") or "-",
        result.get("retry_count", 0),
        len(answer),
        len(citations),
        elapsed_ms,
    )

    return QueryResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
        llm_model=result.get("llm_model") or "",
        include_expired=request.include_expired,
        elapsed_ms=elapsed_ms,
        route=result.get("route") or "",
        route_reason=result.get("route_reason") or "",
        sub_queries=sub_queries,
        retrieval_attempts=int(result.get("retrieval_attempts") or 0),
        verdict=result.get("verdict") or "",
        unsupported_claims=list(result.get("unsupported_claims") or []),
        retry_count=int(result.get("retry_count") or 0),
        error=result.get("error") or "",
    )


__all__ = ["router"]
