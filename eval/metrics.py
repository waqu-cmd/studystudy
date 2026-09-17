"""评估指标（纯函数模块）。

为什么单独成模块，而不写在 ``eval.py`` 里
----------------------------------------
指标是评估的**口径**。混在编排脚本里会导致「顺手改一行 print 就把口径改了」，
而且无法覆盖边界。本模块不读写文件、不联网、不 import 图，因此
``tests/test_eval.py`` 不需要任何 fixture 就能完整覆盖。

两条贯穿全模块的约定
--------------------
1. **``None`` = 不适用，不是 0 分。**
   陷阱题没有标准答案，它的 Hit@3 不是「0 分」而是「没有这项指标」。
   把不适用算成 0 会直接拖垮均值，并且会掩盖真实退化 —— 一个本该 1.0 的指标
   因为混入 5 道陷阱题的 0 变成 0.86，看起来像「质量下降」，
   实际上是统计口径错了。所有指标函数在输入不适用时返回 ``None``，
   :func:`mean` 跳过 ``None``。

2. **干净拒答 = 文本拒答 **且** 引用为空。**
   两个条件缺一不可。只按「引用数为 0」判定，会把阶段 6 实测到的
   「如实拒答且引用为 0」这种**正确**行为误记为忠实度缺口；
   只按文本关键词判定，又会漏掉「嘴上说无法确认、实际上编造了引用」的情况。

   但这条严格口径会把**合法拒答**也判掉：实测 t2 的答案是
   「根据现有资料无法确认。参考资料中仅有 2026 年第三季度的销售政策
   ……但未包含 2026 年第二季度的相关信息 [q3_p1]」—— 那句引用支撑的是
   「资料中仅含 Q3」这一**元事实**，是合法证据。因此严格口径只是**保守下界**。

拒答判定为什么是启发式、以及如何自我校验
----------------------------------------
文本拒答只能靠短语匹配，必然有误判。本模块因此**三轨并报**：
:func:`looks_like_refusal` 给宽松上界（只要文本拒答）、
:func:`is_clean_refusal` 给保守下界（文本拒答且零引用）、
judge（``judge.py``）给语义真值 ``refusal_correct``。
真值被上下界夹住；三者背离时会出现在
:func:`refusal_diagnostics` 的争议清单里，等人工一眼定论 ——
而不是悄悄采信其中某一个，也不是把「口径过严」记成「产品退化」。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

DEFAULT_K = 3
"""Hit@K 的默认 K。蓝图口径为 Hit@3。"""


# --------------------------------------------------------------------------- #
# 检索类指标
# --------------------------------------------------------------------------- #


def first_hit_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> int | None:
    """首个命中标答块的排名（1 起）。

    Returns:
        ``None`` —— 本题没有标答（陷阱题），指标不适用。
        ``0``    —— 有标答但整条召回列表都没命中。
        ``>=1``  —— 命中排名。

    ``None`` 与 ``0`` 必须区分：前者是「没有这项指标」，后者是「这项指标是 0 分」。
    把两者合并成 0 就是约定 1 要避免的错误。调用方应显式处理两个分支。
    """
    targets = {item for item in relevant if item}
    if not targets:
        return None
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in targets:
            return rank
    return 0


def hit_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int = DEFAULT_K
) -> bool | None:
    """top-k 内是否命中任一标答块。无标答时返回 ``None``。"""
    rank = first_hit_rank(retrieved, relevant)
    if rank is None:
        return None
    if rank == 0:
        return False
    return rank <= k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float | None:
    """MRR 的单题分量：首个命中排名的倒数。无标答 ``None``，未命中 ``0.0``。"""
    rank = first_hit_rank(retrieved, relevant)
    if rank is None:
        return None
    if rank == 0:
        return 0.0
    return 1.0 / rank


def version_accuracy(
    top_versions: Sequence[str], required: Iterable[str]
) -> float | None:
    """版本准确率：**首位**召回块的版本是否落在应有版本集合内。

    口径为什么取「首位」而不是「前 K 全部属于应有版本」
    --------------------------------------------------
    本项目每篇政策恰好切成 4 块，一次 top_k=10 的检索会返回 3 篇文档的块，
    要求「前 K 全部同版本」永远不可能成立（跨文档必然混入 v2 / v3）。
    而「首位是否命中正确版本」恰好就是时效过滤要保护的东西：关闭过滤后，
    实测 2026Q1/Q2 的块会直接顶到首位（见 ``baseline.json`` 的
    ``cases_without_expiry_filter``），该指标随之从 1.0 掉到 0.6。

    Returns:
        1.0 / 0.0；``required`` 为空（陷阱题）时 ``None``。
    """
    targets = {item for item in required if item}
    if not targets:
        return None
    if not top_versions:
        return 0.0
    return 1.0 if str(top_versions[0]) in targets else 0.0


def count_stale(retrieved_versions: Sequence[str], stale_versions: Iterable[str]) -> int:
    """召回结果中来自「已过期版本」的块数。

    启用时效过滤时该值必须恒为 0。这是**硬不变量**而非统计指标：
    只要出现一次，就说明过滤链路被绕过（向量下推、BM25 侧过滤、或 MCP 通道
    三者中任一失效），属于确定性缺陷，因此在回归测试里按 0 容差断言。
    """
    stale = {item for item in stale_versions if item}
    return sum(1 for version in retrieved_versions if str(version) in stale)


# --------------------------------------------------------------------------- #
# 生成类指标
# --------------------------------------------------------------------------- #

REFUSAL_MARKERS: tuple[str, ...] = (
    "无法确认",
    "无法回答",
    "无法判断",
    "无法提供",
    "无法核实",
    "无法从",
    "无法给出",
    "不能确认",
    "无可奉告",
    "没有相关",
    "无相关",
    "无任何",
    "未提及",
    "未包含",
    "未找到",
    "没有找到",
    "找不到",
    "不具备",
)
"""文本拒答的强标记。

只收「明确的否认/缺失表述」，不收「不包含」「没有」这类在正常否定句中
也会出现的词 —— 例如「标准返点不包含区域补贴」是一句正确陈述，
把它判成拒答会污染拒答率。宁可漏判（由 judge 兜底），不可误判。
"""


def looks_like_refusal(answer: str) -> bool:
    """纯文本判定：答案是否表达了「现有资料不足以回答」。"""
    text = (answer or "").strip()
    if not text:
        return False
    return any(marker in text for marker in REFUSAL_MARKERS)


def is_clean_refusal(answer: str, citations: Sequence[Any] | None) -> bool:
    """干净拒答：文本拒答 **且** 未给出任何引用。

    为什么必须同时成立：若答案一边说「无法确认」一边挂出引用，
    说明它在用引用为一句自己都没把握的话做背书 —— 这恰恰是忠实度风险，
    不能记成合格拒答。
    """
    return looks_like_refusal(answer) and not (citations or [])


def unsupported_claim_count(claims: Sequence[str] | None) -> int:
    """verifier 报出的无依据断言条数。"""
    return len([c for c in (claims or []) if str(c).strip()])


def refusal_diagnostics(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """列出**三条拒答口径互相矛盾**的陷阱题，供人工快速复核。

    为什么要单独列出：三条口径（严格 / 纯文本 / judge 语义）不一致时，
    单看汇总数字无法判断该改口径还是该改产品。把题号、引用内容与 judge 的
    语义判定并排放在一起，人一眼就能定论 —— 例如实测的 t2：
    答案是拒答，但引用了一个 Q3 的块来支撑「资料中仅含 Q3」这一**元事实**，
    该引用是合法证据而非错误背书，judge 判 refusal_correct=true。
    这种题不进「有争议」清单就会永远挂着一个看不懂的缺口。

    一致通过的题目不进清单，避免报告被无争议项淹掉。
    """
    contested: list[dict[str, Any]] = []
    for case in cases:
        if not case.get("should_refuse"):
            continue
        answer = case.get("answer") or ""
        citations = case.get("citations") or []
        judge = case.get("judge") if isinstance(case.get("judge"), dict) else None
        judge_verdict = judge.get("refusal_correct") if judge else None
        strict = is_clean_refusal(answer, citations)
        if strict and judge_verdict in (None, True):
            continue
        contested.append(
            {
                "id": case.get("id"),
                "text_refusal": looks_like_refusal(answer),
                "strict_refusal": strict,
                "judge_refusal_correct": judge_verdict,
                "citations": list(citations),
                "answer": answer,
            }
        )
    return contested


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


def mean(values: Iterable[float | int | None]) -> float | None:
    """求均值，跳过 ``None``（不适用）。

    全部为 ``None`` 时返回 ``None`` 而不是 0.0 —— 分母为空的均值没有定义，
    返回 0.0 会在报告里伪装成一个真实的 0 分。
    """
    kept = [float(v) for v in values if v is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _metric(name: str, values: Iterable[Any], *, boolean: bool = False) -> dict[str, Any]:
    """把一组单题分量收成一个带分母的指标块。

    统一形状 ``{"n": 有效题数, "value": 均值}``：分母必须随指标一起出现，
    否则「拒答率 0.8」在 n=5 与 n=50 下的可信度差异会被完全抹掉。
    """
    kept = [v for v in values if v is not None]
    if boolean:
        kept = [1.0 if v else 0.0 for v in kept]
    return {"n": len(kept), "value": None if not kept else mean(kept)}


def summarize(
    cases: Sequence[dict[str, Any]],
    *,
    k: int = DEFAULT_K,
    with_answer: bool = True,
) -> dict[str, Any]:
    """把逐题结果汇总成分层指标块。

    Args:
        cases: 每题一个字典，需含以下键（缺失按空值处理）：
            ``should_refuse`` / ``relevant_chunk_ids`` / ``retrieved_chunk_ids`` /
            ``required_doc_versions`` / ``retrieved_versions`` /
            ``answer`` / ``citations`` / ``judge`` / ``retry_count`` /
            ``retrieval_attempts`` / ``route`` / ``error``
        k: Hit@K 的 K。
        with_answer: 本轮是否真的执行了生成与核查。``False``（纯检索档）时
            ``generation`` 与 Verifier 相关指标**一律记 ``None``**，而不是 0 ——
            ``with_answer=false`` 下答案必然是空串、``retry_count`` 必然是 0，
            把它们算成「拒答率 0」「Verifier 触发率 0」等于把「没测量」
            报告成「测得极差」。这正是本模块约定 1 要避免的事。

    Returns:
        分三层：
        - ``retrieval``  —— 检索质量（可用 ``with_answer=false`` 零 LLM 配额测量）
        - ``generation`` —— 生成质量（拒答率、忠实度、引用准确率；需 judge）
        - ``lifecycle``  —— 自纠正与路由的可观测性（Verifier 触发率等）
    """

    def answerable(case: dict[str, Any]) -> bool:
        return bool(case.get("relevant_chunk_ids"))

    retrieval_cases = [c for c in cases if answerable(c)]
    trap_cases = [c for c in cases if c.get("should_refuse")]

    judged = [c for c in cases if isinstance(c.get("judge"), dict)]

    stale_total = sum(
        count_stale(c.get("retrieved_versions") or [], c.get("stale_versions") or [])
        for c in cases
    )
    stale_cases = sum(
        1
        for c in cases
        if count_stale(c.get("retrieved_versions") or [], c.get("stale_versions") or [])
        > 0
    )

    route_counts: dict[str, int] = {}
    for case in cases:
        route = str(case.get("route") or "")
        route_counts[route] = route_counts.get(route, 0) + 1

    if with_answer:
        generation: dict[str, Any] = {
            "measured": True,
            "refusal_rate": _metric(
                "refusal_rate",
                (is_clean_refusal(c.get("answer") or "", c.get("citations"))
                 for c in trap_cases),
                boolean=True,
            ),
            "refusal_rate_text": _metric(
                "refusal_rate_text",
                (looks_like_refusal(c.get("answer") or "") for c in trap_cases),
                boolean=True,
            ),
            "judge_refusal_rate": _metric(
                "judge_refusal_rate",
                (c["judge"].get("refusal_correct")
                 for c in trap_cases
                 if isinstance(c.get("judge"), dict)
                 and c["judge"].get("refusal_correct") is not None),
                boolean=True,
            ),
            "refusal_diagnostics": refusal_diagnostics(trap_cases),
            "faithfulness": _metric(
                "faithfulness",
                (c["judge"].get("faithfulness") for c in judged),
            ),
            "citation_accuracy": _metric(
                "citation_accuracy",
                (c["judge"].get("citation_accuracy") for c in judged),
            ),
            "unsupported_claims_total": sum(
                unsupported_claim_count(c.get("unsupported_claims")) for c in cases
            ),
            "judged_cases": len(judged),
            "judge_failed_cases": sum(
                1 for c in judged if c["judge"].get("judge_failed")
            ),
        }
        verifier_trigger = _metric(
            "verifier_trigger_rate",
            (int(c.get("retry_count") or 0) > 0 for c in cases),
            boolean=True,
        )
        avg_retry = mean(c.get("retry_count") for c in cases)
    else:
        empty = {"n": 0, "value": None}
        generation = {
            "measured": False,
            "refusal_rate": dict(empty),
            "refusal_rate_text": dict(empty),
            "judge_refusal_rate": dict(empty),
            "refusal_diagnostics": [],
            "faithfulness": dict(empty),
            "citation_accuracy": dict(empty),
            "unsupported_claims_total": 0,
            "judged_cases": 0,
            "judge_failed_cases": 0,
        }
        verifier_trigger = {"n": 0, "value": None}
        avg_retry = None

    return {
        "n_cases": len(cases),
        "n_answerable": len(retrieval_cases),
        "n_trap": len(trap_cases),
        "retrieval": {
            f"hit_at_{k}": _metric(
                f"hit_at_{k}",
                (hit_at_k(c.get("retrieved_chunk_ids") or [], c["relevant_chunk_ids"], k)
                 for c in retrieval_cases),
                boolean=True,
            ),
            "mrr": _metric(
                "mrr",
                (reciprocal_rank(c.get("retrieved_chunk_ids") or [], c["relevant_chunk_ids"])
                 for c in retrieval_cases),
            ),
            "version_accuracy": _metric(
                "version_accuracy",
                (version_accuracy(c.get("retrieved_versions") or [], c["required_doc_versions"])
                 for c in retrieval_cases),
            ),
            "stale_hit_chunks": stale_total,
            "stale_hit_cases": stale_cases,
        },
        "generation": generation,
        "lifecycle": {
            "measured_with_answer": with_answer,
            "verifier_trigger_rate": verifier_trigger,
            "avg_retrieval_attempts": mean(
                c.get("retrieval_attempts") for c in cases
            ),
            "avg_retry_count": avg_retry,
            "route_counts": route_counts,
            "error_cases": [
                {"id": c.get("id"), "error": c.get("error")}
                for c in cases
                if str(c.get("error") or "").strip()
            ],
        },
    }


__all__ = [
    "DEFAULT_K",
    "REFUSAL_MARKERS",
    "first_hit_rank",
    "hit_at_k",
    "reciprocal_rank",
    "version_accuracy",
    "count_stale",
    "looks_like_refusal",
    "is_clean_refusal",
    "unsupported_claim_count",
    "refusal_diagnostics",
    "mean",
    "summarize",
]
