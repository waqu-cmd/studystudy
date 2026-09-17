"""图节点测试：路由、状态传递、累加字段清零、优雅降级、自纠正回路、
并行子问题分发、多轮会话隔离。

全部用例使用注入替身（FakeRetriever / FakeLLM / 核查替身 / 路由替身），不联网、
不访问 Chroma —— 图逻辑（路由、状态合并、分支选择、回退闸门、并行汇合、
跨轮隔离）与外部依赖完全解耦，因此这些用例可以在 CI 里零成本反复跑。

阶段 3 追加的部分集中在「自纠正回路」与「核查结果解析」两节；
阶段 5 追加「Supervisor 的 LLM 层」「并行分发与汇合」「多轮会话隔离」三节。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.builder import build_graph
from app.graph.edges import (
    build_fanout_payload,
    effective_sub_queries,
    route_after_supervisor,
)
from app.graph.nodes.supervisor import (
    MAX_SUB_QUERIES,
    MAX_SUB_QUERY_CHARS,
    clean_sub_queries,
    decide_route,
    normalize_intent,
    reset_turn_node,
)
from app.graph.nodes.synthesizer import REFUSE_TEXT
from app.graph.nodes.verifier import (
    MAX_CLAIMS,
    MAX_CLAIM_CHARS,
    MAX_RETRY_QUERY_CHARS,
    build_retry_query,
    clean_claims,
    normalize_verdict,
    parse_verification_payload,
)
from app.graph.state import (
    ACCUMULATING_KEYS,
    ROUTE_DIRECT,
    ROUTE_RETRIEVE,
    VERDICT_FAIL,
    VERDICT_PASS,
    initial_state,
)
from app.schemas.query import RetrievedChunk

# --------------------------------------------------------------------------- #
# 替身
# --------------------------------------------------------------------------- #


def make_chunk(
    chunk_id: str = "sales_policy_2026q3_p1",
    *,
    text: str = "标准返点比例：新客户签约金额的 3.5%，存量客户续约金额的 2.0%。",
) -> RetrievedChunk:
    """构造一条召回结果。doc_id 由 chunk_id 前缀推导，保持与 chunker 一致。"""
    doc_id = chunk_id.rsplit("_p", 1)[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title="2026Q3 销售政策",
        version="2026Q3",
        heading="返点政策",
        effective_date="2026-07-01",
        expire_date="2026-09-30",
        text=text,
        score=0.0325,
        vector_rank=1,
        bm25_rank=1,
        vector_distance=0.21,
    )


class FakeRetriever:
    """可编程检索器替身：记录调用参数，返回预置结果或抛指定异常。

    ``responses`` 用于构造回退场景（第 N 轮召回不同的块）；脚本用尽后重复最后一项。
    ``hits`` 则始终返回同一批结果。
    """

    def __init__(
        self,
        hits: list[RetrievedChunk] | None = None,
        *,
        error: Exception | None = None,
        responses: list[list[RetrievedChunk]] | None = None,
    ) -> None:
        self._hits = hits if hits is not None else []
        self._error = error
        self._responses = list(responses) if responses else None
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_expired: bool | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "include_expired": include_expired}
        )
        if self._error is not None:
            raise self._error
        if self._responses:
            batch = (
                self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
            )
            return list(batch)
        return list(self._hits)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """生成用 LLM 替身：记录收到的消息，返回预置答案。"""

    model_name = "fake-model"

    def __init__(self, answer: str = "新客户返点为 3.5% [sales_policy_2026q3_p1]") -> None:
        self._answer = answer
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> _FakeMessage:
        self.calls.append(messages)
        return _FakeMessage(self._answer)

    @property
    def system_prompt(self) -> str:
        return self.calls[-1][0].content if self.calls else ""


class ScriptedVerifierLLM:
    """核查替身：按调用次序返回预置核查结论。

    故意**不**实现 ``with_structured_output``，因此走的是文本 JSON 解析路径 ——
    这正是「结构化通道不可用时」的兜底行为，值得单独覆盖。
    脚本用尽后重复最后一项，便于构造「一直失败直到触顶」的场景。
    """

    model_name = "scripted-verifier"

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results or [])
        self._error = error
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> _FakeMessage:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        if len(self._results) > 1:
            item = self._results.pop(0)
        else:
            item = self._results[0] if self._results else {"verdict": "pass"}
        return _FakeMessage(json.dumps(item, ensure_ascii=False))


class _StructuredRunner:
    """模拟 ``with_structured_output`` 的返回对象。"""

    def __init__(self, schema: Any, payload: dict[str, Any], error: Exception | None) -> None:
        self._schema = schema
        self._payload = payload
        self._error = error

    def invoke(self, messages: list[Any]) -> Any:
        if self._error is not None:
            raise self._error
        return self._schema(**self._payload)


class StructuredVerifierLLM:
    """核查替身（结构化输出路径）：记录 method 参数，返回 pydantic 对象。"""

    model_name = "structured-verifier"

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        structured_error: Exception | None = None,
        text_payload: dict[str, Any] | None = None,
    ) -> None:
        self._payload = payload or {"verdict": "pass", "unsupported_claims": []}
        self._structured_error = structured_error
        self._text_payload = text_payload or {"verdict": "pass"}
        self.methods: list[str | None] = []
        self.text_calls = 0

    def with_structured_output(self, schema: Any, *, method: str | None = None) -> Any:
        self.methods.append(method)
        return _StructuredRunner(schema, self._payload, self._structured_error)

    def invoke(self, messages: list[Any]) -> _FakeMessage:
        self.text_calls += 1
        return _FakeMessage(json.dumps(self._text_payload, ensure_ascii=False))


FAIL_RESULT: dict[str, Any] = {
    "verdict": "fail",
    "unsupported_claims": ["华东区区域补贴为 0.8 个百分点"],
    "reason": "区域补贴在参考资料中未提及",
}
PASS_RESULT: dict[str, Any] = {"verdict": "pass", "unsupported_claims": [], "reason": ""}

QUERY = "2026Q3 销售政策的返点比例是多少？"

COMPOUND_QUERY = "2026Q2 的销售政策相比 Q1 有哪些变化，对华东区的影响是什么？"
"""蓝图阶段 5 验收用的复合问题（两个独立诉求，应被拆成两路并行检索）。"""


class FakeSubQueryRetriever:
    """按检索词返回不同结果的检索替身。

    为什么不用 ``FakeRetriever.responses``（按调用次序取结果）
    --------------------------------------------------------
    阶段 5 的并行分支是**并发执行**的，调用次序不确定。按次序取结果的替身会让
    用例时而通过时而失败；这里改为按检索词内容匹配，结果与并发顺序无关。
    """

    def __init__(
        self,
        mapping: dict[str, list[RetrievedChunk]] | None = None,
        *,
        default: list[RetrievedChunk] | None = None,
    ) -> None:
        self._mapping = dict(mapping or {})
        self._default = list(default or [])
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_expired: bool | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        for key, hits in self._mapping.items():
            if key in query:
                return list(hits)
        return list(self._default)


class RuleOnlySupervisorLLM:
    """路由替身：调用即失败，强制 supervisor 回退到规则路由。

    存在的理由：supervisor 自阶段 5 起也会调用模型。若它与生成节点共用同一个替身，
    ``llm.calls == []`` 这类「生成侧没被调用」的断言会被路由节点的那次调用污染 ——
    断言就会失效（通过或失败都不再说明它本来想说明的事）。
    """

    model_name = "rule-only-supervisor"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[list[Any]] = []
        self._error = error or RuntimeError("路由替身按设计不可用")

    def invoke(self, messages: list[Any]) -> _FakeMessage:
        self.calls.append(messages)
        raise self._error


class ScriptedSupervisorLLM:
    """路由替身：返回预置的决策 JSON（走文本 JSON 解析路径）。"""

    model_name = "scripted-supervisor"

    def __init__(
        self,
        decision: dict[str, Any] | None = None,
        *,
        text: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self._decision = decision or {"intent": "retrieve"}
        self._text = text
        self._error = error
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> _FakeMessage:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        if self._text is not None:
            return _FakeMessage(self._text)
        return _FakeMessage(json.dumps(self._decision, ensure_ascii=False))


class StructuredSupervisorLLM:
    """路由替身（结构化输出路径）：记录 method 参数，返回 pydantic 对象。"""

    model_name = "structured-supervisor"

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        structured_error: Exception | None = None,
        text_payload: dict[str, Any] | None = None,
    ) -> None:
        self._payload = payload or {"intent": "retrieve"}
        self._structured_error = structured_error
        self._text_payload = text_payload or {"intent": "retrieve"}
        self.methods: list[str | None] = []
        self.text_calls = 0

    def with_structured_output(self, schema: Any, *, method: str | None = None) -> Any:
        self.methods.append(method)
        return _StructuredRunner(schema, self._payload, self._structured_error)

    def invoke(self, messages: list[Any]) -> _FakeMessage:
        self.text_calls += 1
        return _FakeMessage(json.dumps(self._text_payload, ensure_ascii=False))


def node_sequence(result: dict[str, Any]) -> list[str]:
    """从终态事件里取出节点执行序列，回退时会自然出现重复节点。"""
    return [event["node"] for event in result["events"]]


# --------------------------------------------------------------------------- #
# 路由决策（规则表）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # 业务关键词：强信号，优先于寒暄规则
        ("2026Q3 销售政策的返点比例是多少？", ROUTE_RETRIEVE),
        ("你好，请问差旅住宿上限是多少", ROUTE_RETRIEVE),
        ("员工报销流程怎么走", ROUTE_RETRIEVE),
        ("密码多久必须更换", ROUTE_RETRIEVE),
        # 无关键词但足够长 → 默认检索（忠实度优先）
        ("华东区这个季度的情况怎么样", ROUTE_RETRIEVE),
        # 寒暄与元问题 → direct
        ("你好", ROUTE_DIRECT),
        ("谢谢！", ROUTE_DIRECT),
        ("Hello", ROUTE_DIRECT),
        ("你是谁？", ROUTE_DIRECT),
        ("你能做什么", ROUTE_DIRECT),
        # 过短且无关键词 → direct
        ("在吗", ROUTE_DIRECT),
        # 空问题 → direct
        ("", ROUTE_DIRECT),
        ("   ", ROUTE_DIRECT),
    ],
)
def test_supervisor_routing(query: str, expected: str) -> None:
    """不同意图走对分支，并始终给出非空理由。"""
    route, reason = decide_route(query)
    assert route == expected
    assert reason.strip(), "路由必须附带人类可读理由，便于排错与响应回显"


def test_kb_hint_beats_greeting() -> None:
    """礼貌前缀不得掩盖真实意图：同时命中寒暄与业务关键词时必须检索。"""
    route, reason = decide_route("你好，请问密码最少几位")
    assert route == ROUTE_RETRIEVE
    assert "关键词" in reason


# --------------------------------------------------------------------------- #
# 图端到端（替身）
# --------------------------------------------------------------------------- #


def test_state_passing() -> None:
    """chunks 能被下游节点读取，答案与引用绑定正确，核查通过后正常收工。"""
    chunk = make_chunk()
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    graph = build_graph(retriever=retriever, llm=llm, verifier_llm=ScriptedVerifierLLM([PASS_RESULT]))

    result = graph.invoke(initial_state(QUERY))

    # 检索节点把块写进 state
    assert [c.chunk_id for c in result["retrieved_chunks"]] == [chunk.chunk_id]
    assert result["retrieval_attempts"] == 1
    # 合成节点读到了这些块，并按 chunk_id 绑定引用
    assert result["citations"][0].chunk_id == chunk.chunk_id
    assert result["citations"][0].doc_version == "2026Q3"
    assert result["answer"]
    # 核查通过，未发生回退
    assert result["verdict"] == VERDICT_PASS
    assert result["retry_count"] == 0
    assert result["unsupported_claims"] == []
    # 事件流按节点顺序累积，供阶段 6 的 SSE 消费
    assert node_sequence(result) == [
        "supervisor",
        "retriever",
        "synthesizer",
        "verifier",
    ]


def test_direct_route_skips_retrieval() -> None:
    """寒暄问题不检索，且提示词里明确禁止编造制度细节。"""
    retriever = FakeRetriever([make_chunk()])
    llm = FakeLLM(answer="你好，我是企业知识库助手。")
    graph = build_graph(retriever=retriever, llm=llm)

    result = graph.invoke(initial_state("你好"))

    assert result["route"] == ROUTE_DIRECT
    assert retriever.calls == [], "direct 分支不得调用检索"
    assert result["retrieved_chunks"] == []
    assert "绝对不要提及任何具体制度条款" in llm.system_prompt


def test_refuse_when_no_chunks() -> None:
    """检索无命中时不调用生成模型，直接拒答 —— 堵住编造路径且省一次 token。

    路由节点自阶段 5 起也会调模型，因此这里显式区分两个替身：``supervisor_llm``
    用规则兜底替身，``llm`` 才是被断言「不该被调用」的生成模型。
    """
    retriever = FakeRetriever([])
    llm = FakeLLM()
    graph = build_graph(
        retriever=retriever, llm=llm, supervisor_llm=RuleOnlySupervisorLLM()
    )

    result = graph.invoke(initial_state(QUERY))

    assert result["answer"] == REFUSE_TEXT
    assert result["citations"] == []
    assert llm.calls == [], "无参考资料时不得调用生成模型"
    assert result["llm_model"] == ""


def test_with_answer_false_skips_synthesizer() -> None:
    """评估模式只检索不生成，用于不消耗 LLM 配额地计算 Hit@K / MRR。"""
    chunk = make_chunk()
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    graph = build_graph(retriever=retriever, llm=llm)

    result = graph.invoke(initial_state(QUERY, with_answer=False))

    assert len(retriever.calls) == 1
    assert [c.chunk_id for c in result["retrieved_chunks"]] == [chunk.chunk_id]
    assert result["answer"] == ""
    assert llm.calls == []
    assert node_sequence(result) == ["supervisor", "retriever"]


def test_retrieval_failure_degrades_gracefully() -> None:
    """检索失败不抛异常：错误进 state，本轮仍返回结构化拒答。

    这对应阶段 4 的验收项「关掉 chroma_server 后 /query 应优雅降级」。
    """
    retriever = FakeRetriever(error=RuntimeError("模拟向量库不可用"))
    llm = FakeLLM()
    graph = build_graph(
        retriever=retriever, llm=llm, supervisor_llm=RuleOnlySupervisorLLM()
    )

    result = graph.invoke(initial_state(QUERY))

    assert "模拟向量库不可用" in result["error"]
    assert result["answer"] == REFUSE_TEXT
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# 自纠正回路（阶段 3 核心）
# --------------------------------------------------------------------------- #


def test_verifier_retry_loop() -> None:
    """必失败场景：核查不通过 → 回退到 retriever 用**新检索词**重检。

    这是阶段 3 最核心的一条不变量：回退必须换检索词。若沿用原问题，
    同一检索器只会召回同一批块，回退就退化成空转。
    """
    first = make_chunk("sales_policy_2026q3_p1")
    second = make_chunk(
        "sales_policy_2026q3_p2",
        text="华东区因份额争夺加剧，区域补贴由 0.5 个百分点提高至 0.8 个百分点。",
    )
    retriever = FakeRetriever(responses=[[first], [second]])
    verifier = ScriptedVerifierLLM([FAIL_RESULT, PASS_RESULT])
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        verifier_llm=verifier,
        max_retry=1,
    )

    result = graph.invoke(initial_state(QUERY))

    # 检索发生两次，且第二次用的是重组后的检索词
    assert len(retriever.calls) == 2
    assert retriever.calls[0]["query"] == QUERY
    assert retriever.calls[1]["query"] != QUERY
    assert "0.8 个百分点" in retriever.calls[1]["query"]

    # 两轮的块都留在状态里，合成器掌握完整证据
    assert [c.chunk_id for c in result["retrieved_chunks"]] == [
        first.chunk_id,
        second.chunk_id,
    ]
    assert result["retrieval_attempts"] == 2
    assert result["retry_count"] == 1
    # 第二轮回退后核查通过，缺口记录仍保留在状态里（增量累加）
    assert result["verdict"] == VERDICT_PASS
    assert result["unsupported_claims"] == FAIL_RESULT["unsupported_claims"]

    # 节点执行序列出现完整的「生成 → 核查 → 回退重检 → 生成 → 核查」
    assert node_sequence(result) == [
        "supervisor",
        "retriever",
        "synthesizer",
        "verifier",
        "retriever",
        "synthesizer",
        "verifier",
    ]
    assert len(verifier.calls) == 2


def test_max_retry_guard() -> None:
    """核查始终不通过时，回退次数严格不超过 max_retry，不死循环。"""
    retriever = FakeRetriever([make_chunk()])
    verifier = ScriptedVerifierLLM([FAIL_RESULT])  # 脚本用尽后重复 fail
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        verifier_llm=verifier,
        max_retry=2,
    )

    result = graph.invoke(initial_state(QUERY))

    # max_retry=2 → 回退 2 次 → 总计 3 次检索、3 次生成、3 次核查
    assert len(retriever.calls) == 3
    assert result["retrieval_attempts"] == 3
    assert result["retry_count"] == 2
    # 触顶后如实输出 fail，不伪装成通过
    assert result["verdict"] == VERDICT_FAIL
    assert node_sequence(result).count("verifier") == 3
    # 三轮检出的是同一批缺口，写回状态时做差集，不得重复堆叠
    assert result["unsupported_claims"] == FAIL_RESULT["unsupported_claims"]


def test_max_retry_zero_disables_retry() -> None:
    """max_retry=0 时完全禁用回退。

    ``retry_count`` 必须为 0 —— 它记录的是「实际已执行的回退次数」，
    而不是「verifier 提出过几次建议」。被闸门拦下的建议不该计数，
    否则响应里会出现「回退了 0 次却报 1 次」的自相矛盾。
    """
    retriever = FakeRetriever([make_chunk()])
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        verifier_llm=ScriptedVerifierLLM([FAIL_RESULT]),
        max_retry=0,
    )

    result = graph.invoke(initial_state(QUERY))

    assert len(retriever.calls) == 1
    assert result["retry_count"] == 0
    assert result["verdict"] == VERDICT_FAIL


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        ("你好", None),  # direct 分支：没有参考资料，核查无处比对
        (QUERY, REFUSE_TEXT),  # 无召回块：合成器走固定拒答，无断言可核查
    ],
)
def test_verifier_skips_when_nothing_to_ground(
    query: str, expected_answer: str | None
) -> None:
    """两条路径合并为一条不变量：没有可核查的事实性断言就不该调核查模型。"""
    verifier = ScriptedVerifierLLM([FAIL_RESULT])
    graph = build_graph(
        retriever=FakeRetriever([]),
        llm=FakeLLM(answer="你好，我是企业知识库助手。"),
        verifier_llm=verifier,
    )

    result = graph.invoke(initial_state(query))

    assert verifier.calls == [], "无可核查内容时不得触发核查"
    assert result["verdict"] == ""
    if expected_answer is not None:
        assert result["answer"] == expected_answer


def test_verifier_fail_open_on_llm_error() -> None:
    """核查自身失败时放行答案，且不消耗回退配额。

    若这里判 fail，缺口根本没被识别出来，重检毫无方向；同时会把 retry 配额烧光。
    """
    retriever = FakeRetriever([make_chunk()])
    verifier = ScriptedVerifierLLM(error=RuntimeError("核查模型不可用"))
    graph = build_graph(retriever=retriever, llm=FakeLLM(), verifier_llm=verifier)

    result = graph.invoke(initial_state(QUERY))

    assert result["verdict"] == VERDICT_PASS
    assert result["retry_count"] == 0
    assert len(retriever.calls) == 1
    assert result["answer"], "核查失败不得丢掉已生成的答案"
    assert any("核查未执行" in event["msg"] for event in result["events"])


def test_verifier_fail_without_claims_is_not_a_retry() -> None:
    """判 fail 但拿不出具体缺口时不回退 —— 没有方向的回退是纯空转。"""
    retriever = FakeRetriever([make_chunk()])
    verifier = ScriptedVerifierLLM(
        [{"verdict": "fail", "unsupported_claims": [], "reason": "说不清哪里有问题"}]
    )
    graph = build_graph(retriever=retriever, llm=FakeLLM(), verifier_llm=verifier)

    result = graph.invoke(initial_state(QUERY))

    assert len(retriever.calls) == 1
    assert result["retry_count"] == 0
    assert result["verdict"] == VERDICT_PASS
    assert any("未给出具体缺口" in event["msg"] for event in result["events"])


def test_verifier_dedups_claims_across_rounds_ignoring_punctuation() -> None:
    """跨轮去重必须忽略标点差异：模型每轮写同一句断言的标点并不稳定。

    实测第二轮回退时会把「…800 元」写成「…800 元，」，只差一个逗号。
    若按字面比较，这条重复项会绕过差集继续堆叠。
    """
    retriever = FakeRetriever([make_chunk()])
    verifier = ScriptedVerifierLLM(
        [
            {"verdict": "fail", "unsupported_claims": ["住宿上限为每晚 800 元"]},
            {"verdict": "fail", "unsupported_claims": ["住宿上限为每晚 800 元，"]},
        ]
    )
    graph = build_graph(
        retriever=retriever, llm=FakeLLM(), verifier_llm=verifier, max_retry=1
    )

    result = graph.invoke(initial_state(QUERY))

    # 带逗号的变体被视为同一条，状态里只保留首次记录
    assert result["unsupported_claims"] == ["住宿上限为每晚 800 元"]
    # 但回退判定不受影响：第二轮仍然照常回退过
    assert len(retriever.calls) == 2
    assert result["retry_count"] == 1


def test_structured_output_path_is_used_when_available() -> None:
    """模型支持结构化输出时，走 json_schema 通道（实测唯一可靠的 method）。"""
    retriever = FakeRetriever([make_chunk()])
    verifier = StructuredVerifierLLM(payload=dict(FAIL_RESULT))
    graph = build_graph(
        retriever=retriever, llm=FakeLLM(), verifier_llm=verifier, max_retry=0
    )

    result = graph.invoke(initial_state(QUERY))

    assert verifier.methods == ["json_schema"], "function_calling 会静默返回空对象，不可用"
    assert verifier.text_calls == 0, "结构化通道可用时不应回退到文本解析"
    assert result["verdict"] == VERDICT_FAIL
    assert result["unsupported_claims"] == FAIL_RESULT["unsupported_claims"]


def test_structured_failure_falls_back_to_text_parsing() -> None:
    """结构化通道报错时回退到文本解析，而不是放弃核查。"""
    retriever = FakeRetriever([make_chunk()])
    verifier = StructuredVerifierLLM(
        structured_error=RuntimeError("端点不支持 response_format"),
        text_payload=dict(FAIL_RESULT),
    )
    graph = build_graph(
        retriever=retriever, llm=FakeLLM(), verifier_llm=verifier, max_retry=0
    )

    result = graph.invoke(initial_state(QUERY))

    assert verifier.text_calls == 1
    assert result["verdict"] == VERDICT_FAIL
    assert result["unsupported_claims"] == FAIL_RESULT["unsupported_claims"]


# --------------------------------------------------------------------------- #
# 核查结果的解析与归一（纯函数）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pass", VERDICT_PASS),
        ("PASS", VERDICT_PASS),
        ("通过", VERDICT_PASS),
        # 实测模型会输出 partial —— 必须落到 fail，因为它意味着确有断言无依据
        ("partial", VERDICT_FAIL),
        ("fail", VERDICT_FAIL),
        ("", VERDICT_FAIL),
        (None, VERDICT_FAIL),
        ("unknown-value", VERDICT_FAIL),
    ],
)
def test_normalize_verdict(raw: Any, expected: str) -> None:
    """非 pass 即 fail：未知取值一律保守处理。"""
    assert normalize_verdict(raw) == expected


def test_clean_claims_strips_noise_and_dedups() -> None:
    """断言的序号、项目符号、包裹引号都要清掉，重复项只留一条。"""
    raw = [
        "1. 华东区补贴为 0.8 个百分点",
        "- 华东区补贴为 0.8 个百分点",
        "「差旅住宿上限为 800 元」",
        "   ",
        "差旅住宿上限为 800 元",
    ]
    claims = clean_claims(raw)

    assert claims == ["华东区补贴为 0.8 个百分点", "差旅住宿上限为 800 元"]


def test_clean_claims_limits_count_and_length() -> None:
    """断言条数与单条长度都有上限，防止把检索词淹没。"""
    claims = clean_claims([f"断言{i}" + "长" * 200 for i in range(MAX_CLAIMS + 5)])

    assert len(claims) == MAX_CLAIMS
    assert all(len(claim) <= MAX_CLAIM_CHARS for claim in claims)


def test_clean_claims_accepts_single_string() -> None:
    """模型偶尔把列表写成单个字符串，也要能处理。"""
    assert clean_claims("华东区补贴为 0.8 个百分点") == ["华东区补贴为 0.8 个百分点"]
    assert clean_claims(None) == []


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "fail", "unsupported_claims": ["a"], "reason": "r"}',
        '```json\n{"verdict": "fail", "unsupported_claims": ["a"], "reason": "r"}\n```',
        '核查完成：{"verdict": "fail", "unsupported_claims": ["a"], "reason": "r"} 以上。',
    ],
)
def test_parse_verification_payload_tolerates_formats(text: str) -> None:
    """裸 JSON、围栏代码块、前后带解释文字，三种形态都要能解析。"""
    payload = parse_verification_payload(text)

    assert payload is not None
    assert payload["verdict"] == "fail"
    assert payload["unsupported_claims"] == ["a"]


@pytest.mark.parametrize("text", ["", "   ", "完全不是 JSON", "[1, 2, 3]"])
def test_parse_verification_payload_returns_none(text: str) -> None:
    """解析不出来返回 None，由调用方决定 fail-open。"""
    assert parse_verification_payload(text) is None


def test_build_retry_query_appends_claims() -> None:
    """重组后的检索词同时包含原问题与缺口，这是「换角度重检」的落地形态。"""
    query = "2026Q3 华东区新客户返点比例是多少？"
    claims = ["区域补贴为 0.8 个百分点", "结算日期为次月 10 日"]

    retry_query = build_retry_query(query, claims)

    assert retry_query.startswith(query)
    assert all(claim in retry_query for claim in claims)


def test_build_retry_query_respects_length_budget() -> None:
    """检索词有长度上限 —— 查询越长其向量越趋近平均语义，区分度反而下降。"""
    query = "问题" * 300
    claims = ["缺口" * 100 for _ in range(10)]

    retry_query = build_retry_query(query, claims)

    assert len(retry_query) <= MAX_RETRY_QUERY_CHARS
    # 原问题与缺口各占一半预算，保证缺口不会被长问题挤掉
    assert len(query) > MAX_RETRY_QUERY_CHARS // 2


def test_build_retry_query_handles_empty_claims() -> None:
    """没有缺口时返回原问题，且不重复拼接。"""
    assert build_retry_query("返点是多少", []) == "返点是多少"
    assert build_retry_query("返点是多少", ["返点是多少"]) == "返点是多少"


# --------------------------------------------------------------------------- #
# 累加字段的清零语义
# --------------------------------------------------------------------------- #


def test_reset_turn_returns_none_for_accumulating_keys() -> None:
    """reset_turn 对累加字段与单调计数返回 None（清空），对普通字段返回零值。

    ``retry_count`` / ``retrieval_attempts`` / ``error`` 自阶段 5 起也是 reducer
    通道，**必须**用 None 而不是 0 / "" 来清空：它们的 reducer 是「取最大值」与
    「保留首个非空」，写 0 或空串对旧值完全无效（``max(2, 0) == 2``），
    上一轮的重试次数会原样穿透到下一轮。这是阶段 5 最容易漏掉的一处。
    """
    dirty = {
        "query": "上一轮的问题",
        "events": [{"node": "stale", "msg": "上一轮残留", "ts": 0.0}],
        "retrieved_chunks": [make_chunk("stale_doc_p0")],
        "unsupported_claims": ["上一轮的未支撑断言"],
        "sub_queries": ["上一轮的子问题"],
        "retry_count": 2,
        "retrieval_attempts": 3,
        "error": "上一轮的错误",
        "answer": "上一轮的答案",
    }

    update = reset_turn_node(dirty)  # type: ignore[arg-type]

    for key in ACCUMULATING_KEYS:
        assert update[key] is None, f"{key} 必须用 None 触发清空"
    for key in ("retry_count", "retrieval_attempts", "error"):
        assert update[key] is None, f"{key} 是 reducer 通道，只能用 None 清空"
    # 普通通道用零值覆盖，避免上一轮的残留被下游误读
    assert update["answer"] == ""
    assert update["verdict"] == ""
    assert update["route"] == ""


def test_reset_turn_clears_seeded_state_in_graph() -> None:
    """真实走图验证清零生效：入参里塞进的上一轮残留不得出现在终态。

    这一条覆盖的是「挂 checkpointer 后跨轮污染」这一类 bug —— 阶段 5 真的挂上了
    MemorySaver，因此这里用种子状态把不变量钉死。除了累加列表，还刻意塞入
    非零的 ``retry_count`` / ``retrieval_attempts``：它们是 reducer 通道，
    清不干净就会让本轮凭空背上上一轮的回退次数。
    """
    stale_event = {"node": "stale", "msg": "上一轮残留", "ts": 0.0}
    stale_chunk = make_chunk("stale_doc_p0")
    graph = build_graph(
        retriever=FakeRetriever([make_chunk()]),
        llm=FakeLLM(),
        supervisor_llm=RuleOnlySupervisorLLM(),
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(
        {
            "query": QUERY,
            "events": [stale_event],
            "retrieved_chunks": [stale_chunk],
            "unsupported_claims": ["上一轮的未支撑断言"],
            "sub_queries": ["上一轮的子问题"],
            "retry_count": 2,
            "retrieval_attempts": 3,
            "error": "上一轮的错误",
        }
    )

    assert stale_event not in result["events"]
    assert "stale_doc_p0" not in [c.chunk_id for c in result["retrieved_chunks"]]
    assert result["unsupported_claims"] == []
    assert result["sub_queries"] == []
    assert result["retry_count"] == 0, "reducer 通道没被 None 清零"
    assert result["retrieval_attempts"] == 1
    assert result["error"] == ""
    assert node_sequence(result) == [
        "supervisor",
        "retriever",
        "synthesizer",
        "verifier",
    ]


# --------------------------------------------------------------------------- #
# Supervisor 的 LLM 层（阶段 5）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("retrieve", ROUTE_RETRIEVE),
        ("RETRIEVE", ROUTE_RETRIEVE),
        ("检索", ROUTE_RETRIEVE),
        ("知识库", ROUTE_RETRIEVE),
        ("direct", ROUTE_DIRECT),
        ("direct_answer", ROUTE_DIRECT),  # 蓝图里的原始命名
        ("寒暄", ROUTE_DIRECT),
        ("乱七八糟", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_intent(raw: Any, expected: str | None) -> None:
    """取值域按白名单收敛；无法识别时返回 None，由调用方整体回退到规则路由。

    返回 None 而不是硬猜一个值：连意图都说不清的响应，它的子问题也不值得采信。
    """
    assert normalize_intent(raw) == expected


def test_clean_sub_queries_strips_noise_and_dedups() -> None:
    """序号、引号、重复项、超长都是实测噪声，必须在送检索前清掉。"""
    raw = [
        "1. 2026Q2 销售政策相比 Q1 的变化",
        "2. 2026Q2 销售政策相比 Q1 的变化。",  # 只多一个句号
        "- 「2026Q2 对华东区的区域补贴调整」",
        "",
        "   ",
        "x" * (MAX_SUB_QUERY_CHARS + 20),
    ]

    cleaned = clean_sub_queries(raw)

    assert cleaned[0] == "2026Q2 销售政策相比 Q1 的变化"
    assert cleaned[1] == "2026Q2 对华东区的区域补贴调整"
    assert len(cleaned) == 3, "去重并丢弃空串后应剩 3 条"
    assert all(len(item) <= MAX_SUB_QUERY_CHARS for item in cleaned)


def test_clean_sub_queries_limits_count_and_accepts_single_string() -> None:
    """条数封顶；单条字符串（模型偶尔写成一段文字）按一条处理而不是按字符切。"""
    many = [f"子问题{i}" for i in range(MAX_SUB_QUERIES + 5)]
    assert len(clean_sub_queries(many)) == MAX_SUB_QUERIES
    assert clean_sub_queries("只有一个诉求") == ["只有一个诉求"]
    assert clean_sub_queries(None) == []


def test_supervisor_uses_structured_output() -> None:
    """结构化通道可用时走 json_schema，拆解结果进入 state 并随响应暴露。"""
    engine = StructuredSupervisorLLM(
        {
            "intent": "retrieve",
            "sub_queries": ["Q2 相比 Q1 的变化", "华东区补贴调整"],
            "reason": "复合问题",
        }
    )
    graph = build_graph(
        retriever=FakeSubQueryRetriever(default=[make_chunk()]),
        llm=FakeLLM(),
        supervisor_llm=engine,
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(COMPOUND_QUERY))

    assert engine.methods == ["json_schema"]
    assert result["route"] == ROUTE_RETRIEVE
    assert result["sub_queries"] == ["Q2 相比 Q1 的变化", "华东区补贴调整"]
    assert result["route_reason"] == "复合问题"


def test_supervisor_falls_back_to_text_json() -> None:
    """结构化通道报错时回退文本 JSON 解析 —— 不能整轮降级为规则路由。"""
    engine = StructuredSupervisorLLM(
        structured_error=RuntimeError("端点不支持 response_format"),
        text_payload={
            "intent": "retrieve",
            "sub_queries": ["子问题A", "子问题B"],
            "reason": "文本兜底",
        },
    )
    graph = build_graph(
        retriever=FakeSubQueryRetriever(default=[make_chunk()]),
        llm=FakeLLM(),
        supervisor_llm=engine,
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(COMPOUND_QUERY))

    assert engine.text_calls == 1
    assert result["sub_queries"] == ["子问题A", "子问题B"]


def test_supervisor_falls_back_to_rules_on_bad_response() -> None:
    """响应不是 JSON → 规则路由接管，且不产生任何子问题。"""
    graph = build_graph(
        retriever=FakeRetriever([make_chunk()]),
        llm=FakeLLM(),
        supervisor_llm=ScriptedSupervisorLLM(text="我不太确定该怎么回答你"),
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(QUERY))

    assert result["route"] == ROUTE_RETRIEVE
    assert result["sub_queries"] == []
    assert "关键词" in result["route_reason"]


def test_supervisor_falls_back_to_rules_on_llm_error() -> None:
    """模型调用异常也必须给出确定结论 —— 路由层不存在「失败」这种终态。"""
    graph = build_graph(
        retriever=FakeRetriever([]),
        llm=FakeLLM(),
        supervisor_llm=ScriptedSupervisorLLM(error=RuntimeError("模型网关 502")),
    )

    result = graph.invoke(initial_state("你好"))

    assert result["route"] == ROUTE_DIRECT


def test_hard_gate_blocks_llm_direct_on_business_question() -> None:
    """硬闸门：LLM 把业务问题判成 direct，必须被强制改判 retrieve。

    这是阶段 5 最重要的一条安全不变量 ——「不检索直接生成」等价于允许编造。
    """
    retriever = FakeRetriever([make_chunk()])
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        supervisor_llm=ScriptedSupervisorLLM(
            {"intent": "direct", "reason": "看起来是个简单问题"}
        ),
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(QUERY))

    assert result["route"] == ROUTE_RETRIEVE
    assert retriever.calls, "被拦截后必须真的去检索"
    assert "已拦截" in result["route_reason"]


def test_llm_intent_overrides_rule_for_short_followup() -> None:
    """反向不对称：规则把「≤3 字」当噪音，LLM 判 retrieve 时听 LLM 的。

    多轮场景下「Q2呢」这类追问本就需要检索；短句规则只是「没有模型时」的启发式。
    """
    retriever = FakeRetriever([make_chunk()])
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        supervisor_llm=ScriptedSupervisorLLM(
            {"intent": "retrieve", "reason": "上下文中的追问"}
        ),
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state("Q2呢"))

    assert result["route"] == ROUTE_RETRIEVE
    assert len(retriever.calls) == 1


def test_eval_mode_skips_supervisor_llm() -> None:
    """with_answer=false 必须完全不碰模型 —— 阶段 7 的评估零配额契约。"""
    supervisor = ScriptedSupervisorLLM({"intent": "direct"})
    llm = FakeLLM()
    retriever = FakeRetriever([make_chunk()])
    graph = build_graph(retriever=retriever, llm=llm, supervisor_llm=supervisor)

    result = graph.invoke(initial_state(QUERY, with_answer=False))

    assert supervisor.calls == [], "评估模式不得调用路由模型"
    assert llm.calls == []
    assert len(retriever.calls) == 1
    assert result["sub_queries"] == []


# --------------------------------------------------------------------------- #
# 并行子问题分发（阶段 5）
# --------------------------------------------------------------------------- #


def test_effective_sub_queries_drops_blanks_and_duplicates() -> None:
    assert effective_sub_queries({"sub_queries": ["a", "", "  ", "a", "b"]}) == ["a", "b"]
    assert effective_sub_queries({}) == []


def test_route_after_supervisor_single_query_uses_plain_edge() -> None:
    """只有 1 条子问题时不分发 —— 单路并行没有收益，只会多一层调度。"""
    route = route_after_supervisor(
        {"route": ROUTE_RETRIEVE, "sub_queries": ["only"], "with_answer": True}
    )
    assert route == "retriever"


def test_route_after_supervisor_fans_out_on_multiple_sub_queries() -> None:
    sends = route_after_supervisor(
        {
            "route": ROUTE_RETRIEVE,
            "query": COMPOUND_QUERY,
            "sub_queries": ["甲", "乙"],
            "with_answer": True,
        }
    )

    assert isinstance(sends, list), "多子问题必须返回 Send 列表而不是节点名"
    assert [send.node for send in sends] == ["retriever", "retriever"]
    assert [send.arg["search_query"] for send in sends] == ["甲", "乙"]
    assert all(send.arg["fanout"] for send in sends)


def test_build_fanout_payload_carries_everything_retriever_needs() -> None:
    """Send 的 payload 是整体替换，因此节点需要的一切都要显式搬运。"""
    chunk = make_chunk()
    payload = build_fanout_payload(
        {
            "query": COMPOUND_QUERY,
            "session_id": "s1",
            "top_k": 5,
            "include_expired": True,
            "retrieved_chunks": [chunk],
            "retrieval_attempts": 1,
            "retry_count": 0,
            "route": ROUTE_RETRIEVE,
            "sub_queries": ["甲", "乙"],
        },
        "甲",
    )

    assert payload["search_query"] == "甲"
    assert payload["query"] == COMPOUND_QUERY
    assert payload["top_k"] == 5
    assert payload["include_expired"] is True
    assert payload["retrieved_chunks"] == [chunk]
    assert payload["fanout"] is True


def test_compound_question_fans_out_and_merges() -> None:
    """蓝图阶段 5 的核心验收：复合问题被拆解、并行检索、合并成一份证据。

    替身按检索词返回**不同**的块，因此「两路都真的跑了」与「结果都汇合了」
    可以同时被断言。
    """
    chunk_a = make_chunk("sales_policy_2026q2_p1")
    chunk_b = make_chunk("sales_policy_2026q2_p7")
    retriever = FakeSubQueryRetriever({"Q1": [chunk_a], "华东": [chunk_b]})
    supervisor = ScriptedSupervisorLLM(
        {
            "intent": "retrieve",
            "sub_queries": ["2026Q2 相比 Q1 的变化", "华东区的区域补贴"],
            "reason": "复合问题",
        }
    )
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        supervisor_llm=supervisor,
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(COMPOUND_QUERY))

    assert len(retriever.calls) == 2, "两个子问题应各触发一次检索"
    assert {c.chunk_id for c in result["retrieved_chunks"]} == {
        chunk_a.chunk_id,
        chunk_b.chunk_id,
    }
    # 汇合之后下游只执行一次（不是每个分支各跑一遍）
    assert node_sequence(result).count("synthesizer") == 1
    assert node_sequence(result).count("verifier") == 1
    # 每支各写一次 retrieval_attempts，keep_max 兜住 → 仍是第 1 轮
    assert result["retrieval_attempts"] == 1
    assert result["sub_queries"] == ["2026Q2 相比 Q1 的变化", "华东区的区域补贴"]
    assert any("子问题" in event["msg"] for event in result["events"])


def test_parallel_branches_dedupe_identical_chunks() -> None:
    """两路都命中同一个块时，reducer 必须去重 —— 同一份证据不能进两次。

    这是 fan-out 最实际的风险：两条子问题问同一份政策的两个侧面时，检索结果
    高度重叠。重复块会浪费 prompt 预算，也会让 citations 出现重复条目。
    """
    shared = make_chunk("sales_policy_2026q2_p1")
    retriever = FakeSubQueryRetriever(default=[shared])
    supervisor = ScriptedSupervisorLLM(
        {"intent": "retrieve", "sub_queries": ["甲子问题", "乙子问题"], "reason": "复合"}
    )
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        supervisor_llm=supervisor,
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(COMPOUND_QUERY))

    assert len(retriever.calls) == 2, "两路都要真的检索过"
    assert [c.chunk_id for c in result["retrieved_chunks"]] == [shared.chunk_id]


def test_three_way_fanout_keeps_scalar_channels_mergeable() -> None:
    """3 路并行同时写 error / retrieval_attempts，必须靠 reducer 合法化。

    这条用例是探针结论的**回归测试**：若哪天有人把 ``retrieval_attempts`` 的
    reducer 去掉，这里会立刻抛
    ``InvalidUpdateError: Can receive only one value per step``，
    而不是等到生产环境真并发时才暴露。
    """
    retriever = FakeRetriever([])  # 三路全部落空 → 各支都往 error 通道写空串
    supervisor = ScriptedSupervisorLLM(
        {"intent": "retrieve", "sub_queries": ["甲", "乙", "丙"], "reason": "三诉求"}
    )
    graph = build_graph(
        retriever=retriever,
        llm=FakeLLM(),
        supervisor_llm=supervisor,
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )

    result = graph.invoke(initial_state(COMPOUND_QUERY))

    assert len(retriever.calls) == 3
    assert result["answer"] == REFUSE_TEXT
    assert result["error"] == ""
    assert result["retrieval_attempts"] == 1


def test_fanout_all_branches_failing_still_degrades() -> None:
    """3 路全部失败：不得抛并发写异常，仍要优雅降级为拒答。

    与阶段 4 的「关掉 chroma_server 后 /query 仍返回 200」是同一条不变量，
    只是失败从单路变成了三路并发。
    """
    retriever = FakeRetriever(error=RuntimeError("模拟向量库不可用"))
    supervisor = ScriptedSupervisorLLM(
        {"intent": "retrieve", "sub_queries": ["甲", "乙", "丙"]}
    )
    graph = build_graph(
        retriever=retriever, llm=FakeLLM(), supervisor_llm=supervisor
    )

    result = graph.invoke(initial_state(COMPOUND_QUERY))

    assert "模拟向量库不可用" in result["error"]
    assert result["answer"] == REFUSE_TEXT
    assert result["retrieved_chunks"] == []


# --------------------------------------------------------------------------- #
# 多轮会话隔离（阶段 5）
# --------------------------------------------------------------------------- #


def test_checkpointer_isolates_turns_and_sessions() -> None:
    """同一 session 的两轮互不污染，不同 session 之间完全隔离。

    断言的是「证据集合」而不是状态对象：只有上一轮的块真的被清掉，
    第二轮才可能只看到自己那一个。这类跨轮污染在阶段 2 就已埋下不变量
    （见 reset_turn 的两个用例），这里是它第一次在真实 checkpointer 下被验证。
    """
    chunk_q1 = make_chunk("sales_policy_2026q1_p1")
    chunk_hd = make_chunk("sales_policy_2026q2_p7")
    retriever = FakeSubQueryRetriever(
        {"返点比例": [chunk_q1], "华东": [chunk_hd]}, default=[chunk_q1]
    )
    graph = build_graph(
        checkpointer=MemorySaver(),
        retriever=retriever,
        llm=FakeLLM(),
        supervisor_llm=RuleOnlySupervisorLLM(),
        verifier_llm=ScriptedVerifierLLM([PASS_RESULT]),
    )
    session_a = {"configurable": {"thread_id": "sess-a"}}

    first = graph.invoke(initial_state("2026Q1 的返点比例是多少？"), config=session_a)
    second = graph.invoke(initial_state("华东区有什么补贴？"), config=session_a)
    other = graph.invoke(
        initial_state("华东区有什么补贴？"),
        config={"configurable": {"thread_id": "sess-b"}},
    )

    assert [c.chunk_id for c in first["retrieved_chunks"]] == [chunk_q1.chunk_id]
    assert [c.chunk_id for c in second["retrieved_chunks"]] == [chunk_hd.chunk_id], (
        "跨轮污染：第二轮仍能看到第一轮的块"
    )
    assert [c.chunk_id for c in other["retrieved_chunks"]] == [chunk_hd.chunk_id]
    # 事件也要清空，否则阶段 6 的 SSE 会把上一轮的轨迹重放一遍
    assert node_sequence(second) == ["supervisor", "retriever", "synthesizer", "verifier"]


def test_default_checkpointer_is_a_singleton() -> None:
    """checkpointer 必须是进程级单例，否则多轮上下文每次编译即丢失。"""
    from app.graph.builder import default_checkpointer, reset_checkpointer

    reset_checkpointer()
    try:
        assert default_checkpointer() is default_checkpointer()
    finally:
        reset_checkpointer()
