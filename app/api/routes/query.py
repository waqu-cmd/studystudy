"""
``POST /query`` —— 同步问答。
``POST /query/stream`` —— SSE 流式问答。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph

from app.api.deps import get_graph
from app.core.config import settings
from app.graph.state import (
    NODE_RETRIEVER,
    NODE_SUPERVISOR,
    NODE_SYNTHESIZER,
    NODE_VERIFIER,
    initial_state,
)
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


# --------------------------------------------------------------------------- #
# 阶段 6：SSE 帧契约
# --------------------------------------------------------------------------- #

STAGE_OF_NODE: dict[str, str] = {
    NODE_SUPERVISOR: "routing",
    NODE_RETRIEVER: "retrieving",
    NODE_SYNTHESIZER: "synthesizing",
    NODE_VERIFIER: "verifying",
}
"""节点名 → 对外阶段名。

为什么不直接把节点名推给客户端：节点名是**图内部**的实现细节（重命名、拆分、
合并都不该影响调用方），而 ``retrieving`` / ``verifying`` 这组阶段名是稳定的
对外词汇，也与蓝图里「断言事件序列包含 retrieving、verifying、final」的验收
口径一致。映射表放在路由层，图侧不需要知道 SSE 的存在。

**为什么没有 ``reset_turn``（曾拟名 ``preparing``）**：它在每轮入口用
``{"events": None}`` 清零累加字段，而 ``None`` 是「清空」语义而非一条事件 ——
「同一通道只能写一次」的约束使它在清空的同时无法再追加事件（这正是阶段 2
单独拆出该节点的原因）。因此它**在设计上就不可能产出一帧**，
映射表里留它只会给出一个永不触发的契约。客户端需要「请求已被受理」的信号时，
应当以「连接建立」为准，而不是等一帧事件。"""

EVENT_STAGE = "stage"
"""一帧节点事件。``data`` 形如
``{"seq", "stage", "node", "msg", "ts", "elapsed_ms"}``。"""
EVENT_FINAL = "final"
"""收尾帧。``data`` 与 ``QueryResponse`` **同构**（见 :func:`_build_response`）。"""
EVENT_ERROR = "error"
"""图级异常帧。出现后仍会补一帧 ``final``，保证客户端永远等得到收口。"""

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_frame(event: str, payload: dict[str, Any]) -> str:
    """把一条事件序列化成 SSE 帧。 """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {body}\n\n"


# --------------------------------------------------------------------------- #
# 会话配置
# --------------------------------------------------------------------------- #


def _thread_config(session_id: str | None) -> dict | None:
    """同一 session_id 的多轮请求共享状态，不同 session_id 之间完全隔离。"""
    if not settings.memory_enabled:
        return None
    thread_id = (session_id or "").strip() or f"anon-{uuid.uuid4().hex}"
    return {"configurable": {"thread_id": thread_id}}


def _build_response(
    request: QueryRequest, result: dict[str, Any], elapsed_ms: int
) -> QueryResponse:
    """把图的终态映射成响应体。"""
    return QueryResponse(
        query=request.query,
        answer=result.get("answer") or "",
        citations=result.get("citations") or [],
        retrieved_chunks=result.get("retrieved_chunks") or [],
        llm_model=result.get("llm_model") or "",
        include_expired=request.include_expired,
        elapsed_ms=elapsed_ms,
        route=result.get("route") or "",
        route_reason=result.get("route_reason") or "",
        sub_queries=list(result.get("sub_queries") or []),
        retrieval_attempts=int(result.get("retrieval_attempts") or 0),
        verdict=result.get("verdict") or "",
        unsupported_claims=list(result.get("unsupported_claims") or []),
        retry_count=int(result.get("retry_count") or 0),
        error=result.get("error") or "",
    )




# --------------------------------------------------------------------------- #
# POST /query —— 同步 JSON
# --------------------------------------------------------------------------- #


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
        return QueryResponse(
            query=request.query,
            include_expired=request.include_expired,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error=f"图执行失败：{type(exc).__name__}: {exc}",
        )

    return _build_response(
        request, result, int((time.perf_counter() - started) * 1000)
    )


# --------------------------------------------------------------------------- #
# POST /query/stream —— SSE
# --------------------------------------------------------------------------- #


async def _sse_events(
    request: QueryRequest, graph: CompiledStateGraph
) -> AsyncIterator[str]:
    started = time.perf_counter()
    state = initial_state(
        request.query,
        session_id=request.session_id,
        top_k=request.top_k,
        include_expired=request.include_expired,
        with_answer=request.with_answer,
    )
    config = _thread_config(request.session_id)

    seq = 0
    last_values: dict[str, Any] = {}
    failure: str = ""

    try:
        async for mode, chunk in graph.astream(
            state, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                last_values = chunk
                continue

            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                # reset_turn 用 None 表达「清零」，不是一条事件。
                for event in update.get("events") or []:
                    seq += 1
                    yield _sse_frame(
                        EVENT_STAGE,
                        {
                            "seq": seq,
                            "stage": STAGE_OF_NODE.get(node, node),
                            "node": node,
                            "msg": event.get("msg", ""),
                            "ts": event.get("ts"),
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        },
                    )
    except Exception as exc:  # noqa: BLE001 - 流已开始，只能把错误变成一帧
        failure = f"图执行失败：{type(exc).__name__}: {exc}"
        yield _sse_frame(EVENT_ERROR, {"node": "", "msg": failure})

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if failure:
        # 与同步端点保持同一形状：失败时 answer 为空、error 有值，
        # 但 elapsed_ms 与其余字段照常给出，调用方无需分支解析。
        response = QueryResponse(
            query=request.query,
            include_expired=request.include_expired,
            elapsed_ms=elapsed_ms,
            error=failure,
        )
    else:
        response = _build_response(request, last_values, elapsed_ms)

    # 收口帧：结构 = QueryResponse，客户端解析一份 schema 即可同时处理两种端点。
    yield _sse_frame(EVENT_FINAL, response.model_dump())


@router.post(
    "/query/stream",
    summary="知识库问答（SSE 流式）",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "逐节点推送事件。帧类型：`stage`（节点事件，`stage` 取值 "
                "preparing/routing/retrieving/synthesizing/verifying）、"
                "`error`（图级异常）、`final`（收尾，data 与 QueryResponse 同构）"
            ),
        }
    },
)
async def query_stream(
    request: QueryRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> StreamingResponse:
    """SSE 流式问答。

    与 ``POST /query`` 共用同一份请求体、同一个图、同一个终态映射，
    区别只在**返回时机**：同步端点等图跑完再一次性返回，
    本端点每产生一条节点事件就推一帧。

    演示方式（``-N`` 关闭 curl 自身的缓冲）：

    .. code-block:: bash

        curl -N -X POST http://127.0.0.1:8000/query/stream \\
          -H 'Content-Type: application/json' \\
          -d '{"query":"2026Q2 销售政策相比 Q1 有哪些变化？"}'
    """
    return StreamingResponse(
        _sse_events(request, graph),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


__all__ = [
    "router",
    "STAGE_OF_NODE",
    "EVENT_STAGE",
    "EVENT_FINAL",
    "EVENT_ERROR",
    "SSE_HEADERS",
]

