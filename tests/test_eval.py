"""评估层单元测试（阶段 7）。

覆盖三块：指标口径、judge 的四条路径、编排器的选题逻辑。
全部离线：不联网、不访问 Chroma、不 spawn MCP 子进程。

为什么这些用例值得写
--------------------
阶段 5 与阶段 6 各栽过一次「判据写错」的跟头（把期望值当成不变量），
两次都是**指标/断言本身错了**而不是产品错了。指标层的错误不会被任何
端到端测试抓到 —— 它只会让报告的数字变得好看或难看，不会让程序报错。
因此这里专门为口径的边界写用例：不适用返回 None 而不是 0、陷阱题不进
检索指标的分母、略微超满分不被读成百分制。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.schemas.query import RetrievedChunk
from eval import metrics as M
from eval import judge as J


# --------------------------------------------------------------------------- #
# 检索类指标
# --------------------------------------------------------------------------- #


def test_first_hit_rank_distinguishes_not_applicable_from_miss() -> None:
    """``None``（无标答）与 ``0``（有标答但未命中）必须可区分。

    这是本模块最核心的一条约定：把不适用算成 0 分会让陷阱题拉低检索均值，
    从而把一次「口径错误」伪装成「质量退化」。
    """
    assert M.first_hit_rank(["a"], []) is None
    assert M.first_hit_rank(["x", "y"], ["a"]) == 0
    assert M.first_hit_rank(["x", "a"], ["a"]) == 2


def test_hit_at_k_boundary_is_inclusive() -> None:
    """rank == k 算命中，rank == k+1 不算。"""
    assert M.hit_at_k(["a", "b", "c"], ["c"], 3) is True
    assert M.hit_at_k(["a", "b", "c", "d"], ["d"], 3) is False
    assert M.hit_at_k(["a"], [], 3) is None


def test_reciprocal_rank_values() -> None:
    assert M.reciprocal_rank(["a", "b"], ["b"]) == pytest.approx(0.5)
    assert M.reciprocal_rank(["x", "y"], ["a"]) == 0.0
    assert M.reciprocal_rank(["x"], []) is None


def test_version_accuracy_reads_first_chunk_only() -> None:
    """口径取首位：跨文档检索必然混入其它版本，全集比对永远不成立。"""
    assert M.version_accuracy(["2026Q3", "v2"], ["2026Q3"]) == 1.0
    assert M.version_accuracy(["v2", "2026Q3"], ["2026Q3"]) == 0.0
    assert M.version_accuracy([], ["2026Q3"]) == 0.0
    assert M.version_accuracy(["2026Q3"], []) is None


def test_count_stale_is_a_hard_invariant_style_metric() -> None:
    assert M.count_stale(["2026Q1", "v2", "2026Q2"], {"2026Q1", "2026Q2"}) == 2
    assert M.count_stale(["v2", "v3"], {"2026Q1"}) == 0
    assert M.count_stale([], {"2026Q1"}) == 0


# --------------------------------------------------------------------------- #
# 拒答判定
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer",
    [
        "根据现有资料无法确认2026年第一季度的销售返点政策。",
        "参考资料中未包含相关信息。",
        "现有资料无任何关于年假的规定。",
        "无法回答该问题。",
    ],
)
def test_looks_like_refusal_accepts_real_refusals(answer: str) -> None:
    assert M.looks_like_refusal(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "标准返点比例为 3.5%，不包含区域补贴。",
        "该政策适用于华北、华东、华南、西南四大区。",
        "",
    ],
)
def test_looks_like_refusal_ignores_ordinary_negation(answer: str) -> None:
    """正常否定句不得被误判为拒答。

    『不包含』『没有』这类词在正确陈述里也会出现，收进标记表会让拒答率虚高。
    宁可漏判（由 judge 语义兜底），不可误判。
    """
    assert M.looks_like_refusal(answer) is False


def test_clean_refusal_requires_empty_citations() -> None:
    """一边拒答一边挂引用不算干净拒答 —— 那是用引用为没把握的话背书。"""
    assert M.is_clean_refusal("根据现有资料无法确认。", []) is True
    assert M.is_clean_refusal("无法确认 [x]", ["x"]) is False
    assert M.is_clean_refusal("返点比例是 3.5%。", []) is False


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


def test_mean_skips_not_applicable_and_handles_empty() -> None:
    assert M.mean([1.0, None, 0.0]) == pytest.approx(0.5)
    assert M.mean([None, None]) is None
    assert M.mean([]) is None


def _case(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "x",
        "should_refuse": False,
        "relevant_chunk_ids": ["c1"],
        "required_doc_versions": ["v1"],
        "retrieved_chunk_ids": ["c1"],
        "retrieved_versions": ["v1"],
        "stale_versions": ["2026Q1"],
        "answer": "答案是 3.5% [c1]。",
        "citations": ["c1"],
        "judge": None,
        "retry_count": 0,
        "retrieval_attempts": 1,
        "route": "retrieve",
        "error": "",
    }
    base.update(overrides)
    return base


def test_summarize_excludes_traps_from_retrieval_denominator() -> None:
    """陷阱题不进检索指标的分母 —— 它们没有标答，算进去就是口径错误。"""
    cases = [
        _case(id="a"),
        _case(id="b", retrieved_chunk_ids=["x"], reciprocal_rank=None),
        _case(
            id="t1",
            should_refuse=True,
            relevant_chunk_ids=[],
            required_doc_versions=[],
            retrieved_chunk_ids=["c9"],
            retrieved_versions=["v1"],
            answer="根据现有资料无法确认。",
            citations=[],
        ),
    ]
    summary = M.summarize(cases)

    assert summary["n_cases"] == 3
    assert summary["n_answerable"] == 2
    assert summary["n_trap"] == 1
    assert summary["retrieval"]["hit_at_3"]["n"] == 2
    assert summary["retrieval"]["hit_at_3"]["value"] == pytest.approx(0.5)
    # 拒答率的分母是陷阱题
    assert summary["generation"]["refusal_rate"]["n"] == 1
    assert summary["generation"]["refusal_rate"]["value"] == pytest.approx(1.0)


def test_summarize_counts_stale_leakage() -> None:
    cases = [
        _case(id="ok"),
        _case(id="leak", retrieved_versions=["2026Q1", "v1"]),
    ]
    summary = M.summarize(cases)
    assert summary["retrieval"]["stale_hit_chunks"] == 1
    assert summary["retrieval"]["stale_hit_cases"] == 1


def test_summarize_separates_judge_failure_from_low_score() -> None:
    """judge 未执行不得计入评分均值 —— 它必须表现为分母变小，而不是分数变低。"""
    cases = [
        _case(id="a", judge={"faithfulness": 0.9, "citation_accuracy": 1.0,
                             "refusal_correct": None, "judge_failed": False}),
        _case(id="b", judge=J.failed_result("boom")),
    ]
    summary = M.summarize(cases)
    assert summary["generation"]["judged_cases"] == 2
    assert summary["generation"]["judge_failed_cases"] == 1
    assert summary["generation"]["faithfulness"]["n"] == 1
    assert summary["generation"]["faithfulness"]["value"] == pytest.approx(0.9)


def test_summarize_reports_verifier_trigger_rate_and_errors() -> None:
    cases = [
        _case(id="a"),
        _case(id="b", retry_count=1),
        _case(id="c", error="MCP检索不可用"),
    ]
    summary = M.summarize(cases)
    lifecycle = summary["lifecycle"]
    assert lifecycle["verifier_trigger_rate"]["value"] == pytest.approx(1 / 3)
    assert lifecycle["avg_retry_count"] == pytest.approx(1 / 3)
    assert [item["id"] for item in lifecycle["error_cases"]] == ["c"]


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #


def test_clamp01_handles_three_real_shapes() -> None:
    assert J.clamp01(0.85) == pytest.approx(0.85)
    assert J.clamp01("0.85（较高）") == pytest.approx(0.85)
    assert J.clamp01(85) == pytest.approx(0.85)
    assert J.clamp01("100") == pytest.approx(1.0)
    assert J.clamp01(-1) == 0.0


def test_clamp01_does_not_read_slight_overshoot_as_percent() -> None:
    """1.5 是「比满分还满」，不是 1.5%。

    若按百分制解释会得到 0.015 —— 把一条好评读成极差评，这是本模块
    最不能接受的错误方向。
    """
    assert J.clamp01(1.2) == 1.0
    assert J.clamp01(1.5) == 1.0
    assert J.clamp01(2.0) == 1.0
    assert J.clamp01(10) == pytest.approx(0.1)


def test_clamp01_rejects_non_numeric() -> None:
    assert J.clamp01(None) is None
    assert J.clamp01(True) is None
    assert J.clamp01("很高") is None


def test_clean_claims_strips_noise_dedups_and_limits() -> None:
    raw = ["1. 断言甲", "- 断言甲", "  断言乙  ", "", "断言丙"]
    assert J.clean_claims(raw) == ["断言甲", "断言乙", "断言丙"]
    long_text = "断" * (J.MAX_JUDGE_CLAIM_CHARS + 50)
    assert len(J.clean_claims([long_text])[0]) == J.MAX_JUDGE_CLAIM_CHARS
    assert len(J.clean_claims([f"断言{i}" for i in range(50)])) == J.MAX_JUDGE_CLAIMS


def test_failed_result_leaves_scores_none() -> None:
    """judge 失败必须留 None 并打失败标记，绝不返回 0 分。"""
    result = J.failed_result("boom")
    assert result["judge_failed"] is True
    assert result["faithfulness"] is None
    assert result["citation_accuracy"] is None
    assert result["refusal_correct"] is None


class _StructuredRunner:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def invoke(self, _messages: Any) -> Any:
        return self._payload


class StubJudgeLLM:
    """结构化通道可用（生产路径）。"""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.structured_calls = 0
        self.text_calls = 0

    def with_structured_output(self, _schema: Any, method: str | None = None) -> Any:
        self.structured_calls += 1
        return _StructuredRunner(self.payload)

    def invoke(self, _messages: Any) -> Any:
        self.text_calls += 1
        raise AssertionError("结构化通道可用时不应走文本兜底")


class BrokenStructuredJudgeLLM:
    """结构化通道构造即失败 -> 回退到文本 JSON。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.text_calls = 0

    def with_structured_output(self, _schema: Any, method: str | None = None) -> Any:
        raise RuntimeError("method json_schema not supported")

    def invoke(self, _messages: Any) -> Any:
        self.text_calls += 1
        return SimpleNamespace(content=self.text)


class DeadJudgeLLM:
    """两条路径都失败。"""

    def with_structured_output(self, _schema: Any, method: str | None = None) -> Any:
        raise RuntimeError("no structured output")

    def invoke(self, _messages: Any) -> Any:
        raise RuntimeError("network down")


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="测试文档",
            version="v1",
            heading="小节",
            text="返点比例为 3.5%。",
            score=1.0,
        )
    ]


def test_judge_answer_reads_structured_payload() -> None:
    engine = StubJudgeLLM(
        {
            "faithfulness": 0.86,
            "citation_accuracy": 1.0,
            "unsupported_claims": ["1. 编造的断言", "编造的断言"],
            "refusal_correct": True,
            "reason": "基本忠实",
        }
    )
    result = J.judge_answer(
        engine, question="问", answer="答 [c1]", chunks=_chunks()
    )
    assert engine.structured_calls == 1
    assert engine.text_calls == 0
    assert result["judge_failed"] is False
    assert result["faithfulness"] == pytest.approx(0.86)
    assert result["unsupported_claims"] == ["编造的断言"]
    assert result["refusal_correct"] is True


def test_judge_answer_accepts_pydantic_payload() -> None:
    """结构化通道也可能返回 pydantic 模型而非字典，两条都要吃下。"""
    payload = J.JudgeResult(
        faithfulness=0.5,
        citation_accuracy=0.25,
        unsupported_claims=[],
        refusal_correct=False,
        reason="部分无依据",
    )
    engine = StubJudgeLLM(payload)
    result = J.judge_answer(engine, question="问", answer="答", chunks=_chunks())
    assert result["faithfulness"] == pytest.approx(0.5)
    assert result["citation_accuracy"] == pytest.approx(0.25)
    assert result["refusal_correct"] is False


def test_judge_answer_falls_back_to_text_json() -> None:
    engine = BrokenStructuredJudgeLLM(
        '好的，结果如下：\n```json\n'
        '{"faithfulness": 0.75, "citation_accuracy": 0.5, '
        '"unsupported_claims": [], "refusal_correct": true, "reason": "ok"}\n```'
    )
    result = J.judge_answer(engine, question="问", answer="答", chunks=_chunks())
    assert engine.text_calls == 1
    assert result["judge_failed"] is False
    assert result["faithfulness"] == pytest.approx(0.75)


def test_judge_answer_marks_failure_without_raising() -> None:
    """judge 整体失败时返回空分并打标记，且**不得抛异常**。

    抛异常会让 35 题的整体评估因一题抖动而中断，剩下题目的数据全部作废。
    """
    result = J.judge_answer(DeadJudgeLLM(), question="问", answer="答", chunks=_chunks())
    assert result["judge_failed"] is True
    assert result["faithfulness"] is None
    assert result["refusal_correct"] is None


def test_judge_answer_marks_failure_on_non_json_reply() -> None:
    engine = BrokenStructuredJudgeLLM("抱歉，我无法给出评分。")
    result = J.judge_answer(engine, question="问", answer="答", chunks=_chunks())
    assert result["judge_failed"] is True
    assert "JSON" in result["reason"]


def test_build_context_flags_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    """截断必须显式暴露，否则「悄悄换了口径」无法被发现。"""
    monkeypatch.setattr(J, "MAX_JUDGE_CONTEXT_CHARS", 10)
    context, truncated = J.build_context(_chunks())
    assert truncated is True
    assert len(context) == 10


# --------------------------------------------------------------------------- #
# 第三条口径：纯检索档必须把生成类指标标为「未测量」
# --------------------------------------------------------------------------- #


def test_summarize_marks_generation_unmeasured_in_retrieval_only_mode() -> None:
    """``with_answer=False`` 时生成类指标必须全部为 ``None``，不能是 0。

    纯检索档下答案必然是空串、``retry_count`` 必然是 0。若按字面计算，
    报告会显示「拒答率 0」「Verifier 触发率 0」—— 把「没测量」报成「测得极差」，
    正好是本模块约定 1 要避免的错误。
    """
    cases = [
        _case(id="a"),
        _case(
            id="t1",
            should_refuse=True,
            relevant_chunk_ids=[],
            required_doc_versions=[],
            answer="",
            citations=[],
        ),
    ]
    measured = M.summarize(cases, with_answer=True)
    unmeasured = M.summarize(cases, with_answer=False)

    assert measured["generation"]["measured"] is True
    assert unmeasured["generation"]["measured"] is False
    assert sorted(unmeasured["generation"]) == sorted(measured["generation"])
    assert unmeasured["generation"]["refusal_rate"] == {"n": 0, "value": None}
    assert unmeasured["generation"]["refusal_rate_text"] == {"n": 0, "value": None}
    assert unmeasured["generation"]["faithfulness"] == {"n": 0, "value": None}
    assert unmeasured["lifecycle"]["verifier_trigger_rate"] == {"n": 0, "value": None}
    # 检索类指标不受影响，仍照常给出
    assert unmeasured["retrieval"]["hit_at_3"]["n"] == 1


def test_summarize_reports_three_refusal_tracks() -> None:
    """拒答率有三条口径：文本（上界）> 严格（下界），judge 为语义真值。

    实测 t2 就是这种情形：答案是拒答，但引用了一个块来支撑「资料中仅含 Q3」
    这一元事实 —— 该引用合法，因此严格口径记 0 而文本口径记 1。
    """
    cases = [
        _case(
            id="t2",
            should_refuse=True,
            relevant_chunk_ids=[],
            required_doc_versions=[],
            answer="根据现有资料无法确认。参考资料中仅有 Q3 政策 [c9]。",
            citations=["c9"],
            judge={"faithfulness": 1.0, "citation_accuracy": 1.0,
                   "refusal_correct": True, "judge_failed": False},
        ),
    ]
    generation = M.summarize(cases)["generation"]
    assert generation["refusal_rate"]["value"] == 0.0
    assert generation["refusal_rate_text"]["value"] == 1.0
    assert generation["judge_refusal_rate"]["value"] == 1.0


def test_refusal_diagnostics_lists_only_contested_cases() -> None:
    """一致通过的拒答题不进争议清单，否则报告会被无争议项淹掉。"""
    clean_refusal = _case(
        id="t1",
        should_refuse=True,
        relevant_chunk_ids=[],
        required_doc_versions=[],
        answer="根据现有资料无法确认。",
        citations=[],
        judge={"faithfulness": 1.0, "citation_accuracy": 1.0,
               "refusal_correct": True, "judge_failed": False},
    )
    meta_citation = _case(
        id="t2",
        should_refuse=True,
        relevant_chunk_ids=[],
        required_doc_versions=[],
        answer="根据现有资料无法确认。资料中仅有 Q3 [c9]。",
        citations=["c9"],
        judge={"faithfulness": 1.0, "citation_accuracy": 1.0,
               "refusal_correct": True, "judge_failed": False},
    )
    ordinary = _case(id="q1")

    diagnostics = M.refusal_diagnostics([clean_refusal, meta_citation, ordinary])
    assert [item["id"] for item in diagnostics] == ["t2"]
    assert diagnostics[0]["strict_refusal"] is False
    assert diagnostics[0]["text_refusal"] is True
    assert diagnostics[0]["judge_refusal_correct"] is True


def test_refusal_diagnostics_flags_a_failed_refusal() -> None:
    """该拒答却作答的题也必须进清单 —— 那是真实缺口，不能只报口径分歧。"""
    answered_instead = _case(
        id="t4",
        should_refuse=True,
        relevant_chunk_ids=[],
        required_doc_versions=[],
        answer="员工每年享有 10 天年假 [c9]。",
        citations=["c9"],
        judge={"faithfulness": 0.0, "citation_accuracy": 0.0,
               "refusal_correct": False, "judge_failed": False},
    )
    diagnostics = M.refusal_diagnostics([answered_instead])
    assert [item["id"] for item in diagnostics] == ["t4"]
    assert diagnostics[0]["text_refusal"] is False
    assert diagnostics[0]["judge_refusal_correct"] is False
