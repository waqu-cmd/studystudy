"""接口层测试。

覆盖三件事：

1. ``POST /query`` 的 JSON 形态 —— 阶段 2~5 的既有契约，改动时不许回归。
2. ``POST /query/stream`` 的 SSE 形态 —— 阶段 6 的交付物：帧格式、事件序列、
   并行分支可见性、``final`` 与 ``QueryResponse`` 同构、异常收口。
3. ``POST /ingest`` 的结果汇总 —— 接口层的计数逻辑（真正的 Chroma 写入由
   阶段 1 的端到端验收与 ``test_chunker`` / ``test_rrf`` 覆盖）。

为什么全部离线
-------------
图用替身 LLM + 替身检索器编译，索引器用替身替换，并且**自建一个只挂路由的
FastAPI 实例**（不导入 ``app.main``）—— 这样既不打网络、不访问 Chroma，
也不会触发 ``app.main`` 的 lifespan 去 spawn MCP 子进程。因此这些用例可以在
CI 里零成本反复跑。

``_build_response`` 是 JSON 端点与 SSE ``final`` 帧的**唯一**终态映射实现，
因此本文件专门有一条用例逐个字段比对两个端点的输出 —— 漂移会被立刻抓住。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.api.deps import get_graph, get_indexer
from app.api.routes import ingest as ingest_router
from app.api.routes import query as query_router
from app.graph.builder import build_graph
from app.schemas.query import QueryResponse, RetrievedChunk

# --------------------------------------------------------------------------- #
# 测试语料
# --------------------------------------------------------------------------- #

SIMPLE_QUERY = "新客户返点比例是多少？"
COMPOSITE_QUERY = "2026Q2 销售政策相比 Q1 有哪些变化？对华东区返点有什么影响？"
COMPOSITE_SUB_QUERIES = ["2026Q2 相比 Q1 的变化", "对华东区返点的影响"]
# 不含 KB_HINTS 里的任何业务关键词，因此 supervisor 判 direct 时不会被硬闸门改判。
DIRECT_QUERY = "你好，请用一句话介绍你自己"

ANSWER = "新客户返点为 3.0% [doc1_p1]"
"""答案里引用 doc1_p1 —— 恰好是替身检索器**首次调用**返回的第一个块的 id，
因此 citations 能被稳定断言（模型编造的 id 会被 extract_citations 丢弃）。"""


# --------------------------------------------------------------------------- #
# 替身
# --------------------------------------------------------------------------- #


class StubRetriever:
    """按调用次序产出不同 ``chunk_id`` 的检索替身。

    刻意让每次调用的块 id 不同（``doc1_*`` / ``doc2_*`` …），这样「并行分支真的
    各检各的」可以被观测到；若两支返回同一批结果，去重 reducer 会把它们合成一份，
    用例就分不清「并行了两支」与「只跑了一支」。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_expired: bool | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        index = len(self.calls)
        return [
            RetrievedChunk(
                chunk_id=f"doc{index}_p{n}",
                doc_id=f"doc{index}",
                doc_title="2026Q2 销售政策",
                version="2026Q2",
                heading="返点政策",
                effective_date="2026-04-01",
                expire_date="2026-06-30",
                text="标准返点比例：新客户签约金额的 3.0%。",
                score=0.03 - n * 0.001,
                vector_rank=n,
                bm25_rank=n,
                vector_distance=0.2,
            )
            for n in (1, 2)
        ]


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class StubLLM:
    """固定回复的 LLM 替身。

    故意**不**实现 ``with_structured_output`` —— 于是 supervisor / verifier 走
    文本 JSON 解析路径（阶段 3 实测确认这是可用的兜底通道），既省一次网络调用，
    也让用例覆盖到该路径。
    """

    model_name = "stub-model"

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages: list[Any]) -> _Message:
        self.calls += 1
        if len(self._replies) == 1:
            return _Message(self._replies[0])
        return _Message(self._replies.pop(0))


class ExplodingGraph:
    """``astream`` 立即抛异常的图替身，用于验证 SSE 的异常收口路径。"""

    async def astream(self, *args: Any, **kwargs: Any):
        raise RuntimeError("boom")
        yield  # pragma: no cover - 仅为让本函数成为 async generator


@dataclass
class StubOutcome:
    """``Indexer.index_*`` 返回值的替身。字段与 ``FileIngestResult`` 对齐。"""

    path: str
    doc_id: str
    doc_title: str
    version: str
    chunks: int
    status: str
    message: str = ""


class StubIndexer:
    """索引器替身：不碰磁盘、不碰 Chroma，只记录调用并返回预置结果。"""

    collection_name = "stub_kb"

    def __init__(self, outcomes: list[StubOutcome] | None = None) -> None:
        self._outcomes = (
            outcomes
            if outcomes is not None
            else [
                StubOutcome("data/docs/a.md", "doc_a", "A 政策", "v1", 3, "indexed"),
                StubOutcome(
                    "data/docs/b.md", "doc_b", "B 政策", "v1", 2, "skipped", "未变更"
                ),
            ]
        )
        self.reset_called = False
        self.indexed_paths: list[list[str]] = []
        self.directory_calls = 0

    def reset(self) -> None:
        self.reset_called = True

    def index_paths(self, files: list[str], *, force: bool = False) -> list[StubOutcome]:
        self.indexed_paths.append(list(files))
        return list(self._outcomes)

    def index_directory(self, *, force: bool = False) -> list[StubOutcome]:
        self.directory_calls += 1
        return list(self._outcomes)

    def count(self) -> int:
        return 20


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #


def make_graph(
    *,
    intent: str = "retrieve",
    sub_queries: list[str] | None = None,
    verdict: str = "pass",
    answer: str = ANSWER,
) -> tuple[Any, StubRetriever]:
    """编译一份全替身图。返回 (图, 检索替身) 便于断言检索确实（或没有）发生。"""
    retriever = StubRetriever()
    supervisor_reply = json.dumps(
        {"intent": intent, "sub_queries": sub_queries or [], "reason": "用例"},
        ensure_ascii=False,
    )
    verifier_reply = json.dumps(
        {"verdict": verdict, "unsupported_claims": []}, ensure_ascii=False
    )
    graph = build_graph(
        checkpointer=MemorySaver(),
        retriever=retriever,
        llm=StubLLM(answer),
        supervisor_llm=StubLLM(supervisor_reply),
        verifier_llm=StubLLM(verifier_reply),
        max_retry=0,
    )
    return graph, retriever


def make_client(graph: Any, indexer: Any | None = None) -> TestClient:
    """自建只挂路由的应用，绕开 ``app.main`` 的 lifespan。"""
    app = FastAPI()
    app.include_router(query_router.router)
    app.include_router(ingest_router.router)
    app.dependency_overrides[get_graph] = lambda: graph
    if indexer is not None:
        app.dependency_overrides[get_indexer] = lambda: indexer
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """把 SSE 响应体拆成 ``[(event 名, data 字典)]``。

    按规范实现到「够用」的程度：``data:`` 可多行（以换行拼接），
    ``event:`` 缺省为 message。
    """
    frames: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data.append(line[len("data: ") :])
        frames.append((name, json.loads("\n".join(data))))
    return frames


def stream_frames(client: TestClient, query: str) -> list[tuple[str, dict[str, Any]]]:
    response = client.post("/query/stream", json={"query": query})
    assert response.status_code == 200, response.text
    return parse_sse(response.text)


def stages_of(frames: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [data["stage"] for name, data in frames if name == "stage"]


def final_of(frames: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    finals = [data for name, data in frames if name == "final"]
    assert len(finals) == 1, f"应恰好有一帧 final，实际 {len(finals)}"
    return finals[0]


# --------------------------------------------------------------------------- #
# 帧格式
# --------------------------------------------------------------------------- #


def test_sse_frame_keeps_data_on_a_single_line() -> None:
    """``data:`` 必须单行 —— 多行会被客户端解析成多个字段。"""
    frame = query_router._sse_frame("stage", {"msg": "第一行\n第二行", "中文": "可读"})
    assert frame.startswith("event: stage\ndata: ")
    assert frame.endswith("\n\n")

    body = frame[len("event: stage\ndata: ") : -2]
    assert "\n" not in body, "data 段出现了裸换行"
    assert "\\n" in body, "换行未被转义"
    assert "中文" in body, "不该做 ascii 转义（中文要能在 curl -N 下直接看）"


def test_sse_headers_disable_proxy_buffering() -> None:
    """反代会缓冲整段响应，把「流式」压成一次性吐出，故必须显式关闭。"""
    client = make_client(make_graph()[0])
    response = client.post("/query/stream", json={"query": SIMPLE_QUERY})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


# --------------------------------------------------------------------------- #
# 事件序列（蓝图阶段 6 的验收项）
# --------------------------------------------------------------------------- #


def test_sse_stream_contains_retrieving_verifying_and_final() -> None:
    """蓝图验收：事件序列包含 retrieving、verifying、final。"""
    client = make_client(make_graph()[0])
    frames = stream_frames(client, SIMPLE_QUERY)

    names = [name for name, _ in frames]
    stages = stages_of(frames)

    assert "retrieving" in stages
    assert "verifying" in stages
    assert names[-1] == "final", "流必须以 final 收口"
    # seq 必须连续递增，客户端据此判断有没有丢帧
    assert [d["seq"] for _, d in frames if "seq" in d] == list(range(1, len(stages) + 1))


def test_sse_stage_frames_carry_node_message_and_timing() -> None:
    client = make_client(make_graph()[0])
    frames = stream_frames(client, SIMPLE_QUERY)

    for name, data in frames:
        if name != "stage":
            continue
        assert data["node"] in query_router.STAGE_OF_NODE
        assert data["msg"], "事件文案不该为空"
        assert isinstance(data["ts"], (int, float))
        assert isinstance(data["elapsed_ms"], int)


def test_sse_final_frame_matches_query_response_schema() -> None:
    """``final`` 的 data 与 ``QueryResponse`` 同构，客户端只需一份解析逻辑。"""
    client = make_client(make_graph()[0])
    frames = stream_frames(client, SIMPLE_QUERY)

    parsed = QueryResponse.model_validate(final_of(frames))
    assert parsed.answer == ANSWER
    assert parsed.route == "retrieve"
    assert parsed.retrieval_attempts == 1
    assert parsed.verdict == "pass"
    assert [c.chunk_id for c in parsed.citations] == ["doc1_p1"]
    assert len(parsed.retrieved_chunks) == 2
    assert parsed.error == ""
    assert parsed.elapsed_ms >= 0


def test_sse_stream_shows_each_parallel_branch_separately() -> None:
    """阶段 5 的并行 fan-out 在流式下应表现为多帧 retrieving。

    注意帧数是「每支一帧」，但**不是「谁先返回谁先到」**：fan-out 的各支属于同一个
    super-step，langgraph 在该超步落定后统一发出 updates —— 真实链路实测两帧
    ``retrieving`` 的时间戳完全相同。因此这里只断言「每支都被单独呈现」，
    不断言到达次序（那会是一条依赖内部调度实现、随时可能失效的断言）。
    """
    graph, retriever = make_graph(sub_queries=COMPOSITE_SUB_QUERIES)
    client = make_client(graph)
    frames = stream_frames(client, COMPOSITE_QUERY)

    stages = stages_of(frames)
    assert stages.count("retrieving") == len(COMPOSITE_SUB_QUERIES)
    assert len(retriever.calls) == len(COMPOSITE_SUB_QUERIES)

    messages = [
        data["msg"]
        for name, data in frames
        if name == "stage" and data["stage"] == "retrieving"
    ]
    assert len(set(messages)) == len(COMPOSITE_SUB_QUERIES), "两条子问题的文案应不同"


def test_direct_route_streams_without_retrieval_or_verification() -> None:
    """寒暄走 direct：不检索、不核查，但仍以 final 收口。"""
    graph, retriever = make_graph(intent="direct")
    client = make_client(graph)
    frames = stream_frames(client, DIRECT_QUERY)

    stages = stages_of(frames)
    assert "retrieving" not in stages
    assert "verifying" not in stages
    assert stages[-1] == "synthesizing"
    assert retriever.calls == []
    assert [name for name, _ in frames][-1] == "final"


# --------------------------------------------------------------------------- #
# 异常收口
# --------------------------------------------------------------------------- #


def test_sse_stream_reports_failure_then_still_finalizes() -> None:
    """图级异常：先推 error 帧，再补 final —— 客户端永远等得到收口。"""
    client = make_client(ExplodingGraph())
    frames = stream_frames(client, SIMPLE_QUERY)

    names = [name for name, _ in frames]
    assert names == ["error", "final"]
    assert "boom" in frames[0][1]["msg"]
    assert "boom" in final_of(frames)["error"]


# --------------------------------------------------------------------------- #
# 两个端点的一致性
# --------------------------------------------------------------------------- #


def test_query_json_endpoint_is_unchanged() -> None:
    """阶段 6 只新增流式端点，同步端点的行为必须一字未改。"""
    client = make_client(make_graph()[0])
    response = client.post("/query", json={"query": SIMPLE_QUERY})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == ANSWER
    assert body["route"] == "retrieve"
    assert body["retrieval_attempts"] == 1
    assert body["verdict"] == "pass"
    assert [c["chunk_id"] for c in body["citations"]] == ["doc1_p1"]
    assert body["error"] == ""


def test_json_and_sse_final_agree_field_by_field() -> None:
    """同一份终态映射驱动两种表示 —— 逐个字段比对，漂移会立刻暴露。"""
    json_client = make_client(make_graph()[0])
    sse_client = make_client(make_graph()[0])

    sync_body = json_client.post("/query", json={"query": SIMPLE_QUERY}).json()
    stream_body = final_of(stream_frames(sse_client, SIMPLE_QUERY))

    assert set(sync_body) == set(stream_body), "两种表示的字段集不一致"
    for key in sync_body:
        if key == "elapsed_ms":
            continue  # 两次运行的耗时天然不同
        assert sync_body[key] == stream_body[key], f"字段 {key} 出现漂移"


# --------------------------------------------------------------------------- #
# /ingest 的接口层计数
# --------------------------------------------------------------------------- #


def test_ingest_aggregates_counts_from_indexer() -> None:
    indexer = StubIndexer()
    client = make_client(make_graph()[0], indexer=indexer)

    response = client.post("/ingest", json={"files": ["data/docs/a.md"]})
    assert response.status_code == 200

    body = response.json()
    assert body["collection"] == "stub_kb"
    assert body["total_files"] == 2
    assert body["total_chunks"] == 5
    assert body["indexed"] == 1
    assert body["skipped"] == 1
    assert body["failed"] == 0
    assert body["collection_count"] == 20
    assert indexer.indexed_paths == [["data/docs/a.md"]]
    assert indexer.reset_called is False


def test_ingest_without_files_indexes_whole_directory() -> None:
    indexer = StubIndexer()
    client = make_client(make_graph()[0], indexer=indexer)

    response = client.post("/ingest", json={})
    assert response.status_code == 200
    assert indexer.directory_calls == 1, "files 留空时应退化为「索引整个 DOCS_DIR」"


def test_ingest_reset_clears_collection_before_indexing() -> None:
    indexer = StubIndexer()
    client = make_client(make_graph()[0], indexer=indexer)

    response = client.post("/ingest", json={"reset": True})
    assert response.status_code == 200
    assert indexer.reset_called is True
    assert indexer.directory_calls == 1
