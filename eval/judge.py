"""LLM-as-judge：忠实度与引用准确率（阶段 7）。

为什么需要 judge，而不是让 verifier 兼任
----------------------------------------
verifier 是**在线组件**，它的判定会直接驱动回退重检，因此必须便宜、快、保守
（fail-open）。judge 是**离线组件**：它只在评估时跑，可以给出更细的连续分数
（0~1）而不是一个 pass/fail，也不承担任何在线决策。
把两者合并会让「评估用的严格判定」泄漏进在线回路 —— 一次评估口径的调整
就会改变线上行为，这是必须避免的耦合。

三条实测得到的结构化输出约束（与 verifier 同源）
------------------------------------------------
langchain 0.3.63 + 百炼兼容端点 + deepseek-v4-flash 下：
1. ``method="json_schema"`` 可用且结果正确 —— 生产路径。
2. ``method="function_calling"`` **静默返回空对象**，不抛异常。最危险的失败形态：
   看起来调用成功，实则什么都没评。**不使用**该 method。
3. ``method="json_mode"`` 直接 400。因此兜底路径改为「纯文本 + 自己抠 JSON」，
   复用 :func:`synthesizer.parse_json_object`（与 supervisor、verifier 同一份实现）。

judge 自身失败时为什么必须留 ``None``，而不是记 0 分
----------------------------------------------------
0 分与「没测到」在报告里必须可区分。若把接口抖动记成 0 分，一次网络故障会表现为
「忠实度从 0.9 崩到 0.1」，从而触发一次完全错误的回滚决策。
本模块因此在失败时置 ``judge_failed=True`` 并把三个分数全部留 ``None``；
:func:`eval.metrics.mean` 会跳过 ``None``，报告里同时给出 ``judge_failed_cases``
计数，让「测得少」与「测得差」能被分开阅读。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_llm
from app.graph.nodes.synthesizer import (
    content_to_text,
    format_context,
    parse_json_object,
)

JUDGE_METHOD = "json_schema"
"""唯一实测可用的结构化输出方式，详见模块 docstring。"""

JUDGE_MAX_TOKENS = 1024
"""judge 输出只有一个分数三元组加几条断言摘录，1024 足够且能压住成本。"""

MAX_JUDGE_CONTEXT_CHARS = 12000
"""送入 judge 的参考资料字符上限。

理论上应当把**生成器看到的全部召回块**原样交给 judge —— 只给一部分会让 judge
把「被截掉的那部分里其实有依据」的断言误判为无依据，从而系统性压低忠实度。
本项目 top_k=10、单块上限 800 字，实测总量约 3000~5000 字，远低于上限；
该上限只是防止语料扩容后 prompt 无界膨胀，一旦触发会打 WARNING 并在报告里
通过 ``context_truncated`` 暴露出来，避免「悄悄换了口径」。
"""

MAX_JUDGE_CLAIMS = 8
"""judge 最多回报的无依据断言条数，与 verifier 的 MAX_CLAIMS 对齐。"""

MAX_JUDGE_CLAIM_CHARS = 120
"""单条断言摘录上限。judge 的摘录仅用于人工核查，不像 verifier 那样要进检索词，
因此可以比 verifier 宽松一些。"""

_NOISE_RE = re.compile(r"^[\s\-•*·–—]*?(?:\d+[.、)）]\s*)?")


class JudgeResult(BaseModel):
    """judge 的结构化契约（走 json_schema 时由服务端约束取值域）。"""

    faithfulness: float = Field(
        description="忠实度 0~1：答案中的事实性断言有多大比例能在参考资料中找到依据。"
        "1 表示全部有依据；0 表示全部无依据。纯拒答（不含任何事实性断言）应给 1。"
    )
    citation_accuracy: float = Field(
        description="引用准确率 0~1：答案标注的引用中，有多大比例确实支撑了它所标注的那句话。"
        "答案未使用任何引用时给 1（不存在错误引用）。"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="无依据的断言原文摘录，逐条列出，不得改写或补充解释",
    )
    refusal_correct: bool = Field(
        description="拒答行为是否正确：问题本应拒答（参考资料中没有相关内容）且答案确实拒答 → true；"
        "问题本有资料且答案正常作答 → true；该拒答却作答、或该作答却拒答 → false"
    )
    reason: str = Field(default="", description="一句话说明判定理由")


SYSTEM_PROMPT = """你是 RAG 答案质量评估员。给你「参考资料」「用户问题」与「系统答案」，\
请从三个维度打分。

维度一：忠实度 faithfulness（0~1）
- 逐句拆解答案中的**事实性断言**，检查每条能否在参考资料里找到依据。
- 只用参考资料判断，不要使用你自己的先验知识。参考资料里没写的，就是无依据。
- 分数 = 有依据的断言数 / 断言总数（四舍五入到两位小数）。
- 纯格式性表述（「根据以上资料」「综上所述」）不算断言。
- 答案若是**拒答**（说明资料不足、不予作答），它不含任何事实性断言，忠实度给 1，
  并在 reason 中注明这是拒答。

维度二：引用准确率 citation_accuracy（0~1）
- 答案中用 [chunk_id] 形式标注了引用。逐条检查：该引用指向的块是否**真的支撑**
  了它所在的那句话？标记存在不代表内容真的被支撑。
- 分数 = 正确引用数 / 引用总数。答案完全没有引用时给 1（不存在错误引用）。

维度三：拒答行为 refusal_correct（true/false）
- 若参考资料确实**没有**回答该问题所需的内容，答案应当拒答。此时答案拒答 → true，
  答案强行作答（编造或答非所问）→ false。
- 若参考资料**有**相关内容，答案应当作答。此时答案作答 → true，
  答案无端拒答 → false。

输出要求：
- faithfulness 与 citation_accuracy 必须是 0 到 1 之间的数字。
- unsupported_claims 只列真正无依据的断言原文，不要改写、不要补充解释、不要合并。
- 全部断言都有依据时，unsupported_claims 为空列表。"""


PERCENT_SCALE_THRESHOLD = 10.0
"""大于等于该值即判定模型用的是百分制，除以 100 后再夹紧。

阈值取 10 而不是 2 的理由：模型在 0~1 标尺上「略微超出满分」很常见
（1.2、1.5 都可能出现，语义是「比满分还满」，应夹到 1.0），
而把 1.5 按百分制解释成 1.5% 会把一条**好评**读成**极差评** ——
这是本模块最不能接受的一类错误。反之 ``85`` / ``100`` 不可能是 0~1 标尺上的
取值，只能是百分制。阈值落在 (1, 10) 之间时一律按「超出满分」处理：
夹紧是保守方向，它不会凭空制造一个差评。
"""


def clamp01(value: Any) -> float | None:
    """把模型给的分数夹到 [0, 1]；无法转成数字时返回 ``None``。

    处理三类实测会出现的输入：
    - 带说明的字符串（``"0.85（较高）"``）—— 抽出第一个数字；
    - 百分制（``85`` / ``100``）—— 除以 100，见 :data:`PERCENT_SCALE_THRESHOLD`；
    - 轻微超出满分（``1.2`` / ``1.5``）—— 夹到 1.0，**不**按百分制解释。

    直接 ``min(1, x)`` 会把 85 悄悄变成 1.0（把差评读成满分），
    而一律 ``x/100`` 又会把 1.5 变成 0.015（把好评读成极差），两个方向都要避开。
    布尔值显式返回 ``None``：``True`` 不是分数，是模型答错了字段类型。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        matched = re.search(r"-?\d+(?:\.\d+)?", value)
        if not matched:
            return None
        value = float(matched.group(0))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number >= PERCENT_SCALE_THRESHOLD:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _to_bool(value: Any) -> bool | None:
    """把模型给的拒答判定归一到布尔；无法识别时返回 ``None``。"""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "correct", "是", "正确", "对"}:
        return True
    if text in {"false", "no", "0", "incorrect", "否", "错误", "不对"}:
        return False
    return None


def clean_claims(raw: Any) -> list[str]:
    """清洗 judge 回报的无依据断言：去格式噪声、去重、限量、限长。"""
    if raw is None:
        return []
    items: list[Any] = [raw] if isinstance(raw, str) else list(raw) if isinstance(
        raw, (list, tuple, set)
    ) else [raw]

    claims: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _NOISE_RE.sub("", str(item or "").strip())
        text = text.strip().strip("\"'“”‘’「」《》").strip()
        if len(text) > MAX_JUDGE_CLAIM_CHARS:
            text = text[:MAX_JUDGE_CLAIM_CHARS].rstrip()
        if not text or text in seen:
            continue
        seen.add(text)
        claims.append(text)
        if len(claims) >= MAX_JUDGE_CLAIMS:
            break
    return claims


def failed_result(reason: str) -> dict[str, Any]:
    """构造「judge 未执行成功」的结果：三个分数留 ``None``，显式打失败标记。

    绝不返回 0 分 —— 见模块 docstring。
    """
    return {
        "faithfulness": None,
        "citation_accuracy": None,
        "unsupported_claims": [],
        "refusal_correct": None,
        "reason": reason,
        "judge_failed": True,
        "context_truncated": False,
    }


def build_context(chunks: list[Any]) -> tuple[str, bool]:
    """把召回块渲染成 judge 看的参考资料，返回 ``(文本, 是否被截断)``。"""
    text = format_context(chunks)
    if len(text) <= MAX_JUDGE_CONTEXT_CHARS:
        return text, False
    return text[:MAX_JUDGE_CONTEXT_CHARS], True


def _structured_runner(engine: Any) -> Callable[..., Any] | None:
    """拿到结构化输出通道；不支持时返回 None。"""
    builder = getattr(engine, "with_structured_output", None)
    if not callable(builder):
        return None
    try:
        return builder(JudgeResult, method=JUDGE_METHOD)
    except Exception:  # noqa: BLE001 - 老版本 langchain 不认 method 参数
        return None


def _payload_to_result(payload: dict[str, Any], *, truncated: bool) -> dict[str, Any]:
    """把结构化 / JSON 两种来源的字段统一成报告用的结果字典。"""
    return {
        "faithfulness": clamp01(payload.get("faithfulness")),
        "citation_accuracy": clamp01(payload.get("citation_accuracy")),
        "unsupported_claims": clean_claims(payload.get("unsupported_claims")),
        "refusal_correct": _to_bool(payload.get("refusal_correct")),
        "reason": str(payload.get("reason") or ""),
        "judge_failed": False,
        "context_truncated": truncated,
    }


def judge_answer(
    engine: Any,
    *,
    question: str,
    answer: str,
    chunks: list[Any],
) -> dict[str, Any]:
    """对一条答案打分。

    Args:
        engine: 具备 ``invoke`` / 可选 ``with_structured_output`` 的模型，可注入替身。
        question: 用户原问题。
        answer: 系统生成的答案。
        chunks: 生成时实际使用的召回块（必须与生成器看到的一致，见模块 docstring）。

    Returns:
        含 ``faithfulness`` / ``citation_accuracy`` / ``unsupported_claims`` /
        ``refusal_correct`` / ``reason`` / ``judge_failed`` / ``context_truncated``
        的字典。judge 失败时分数为 ``None`` 且 ``judge_failed=True``。

    两条路径：结构化输出（生产路径）与纯文本 JSON（兜底）。两者都失败时
    **不抛异常**，而是返回 :func:`failed_result` —— 评估脚本不该因为一条题目的
    judge 抖动而整体崩掉，剩下 34 条的分数仍然有效。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    context, truncated = build_context(chunks)
    user_content = (
        f"【参考资料】\n{context}\n\n【用户问题】\n{question}\n\n【系统答案】\n{answer}"
    )
    system = SystemMessage(content=SYSTEM_PROMPT)

    structured = _structured_runner(engine)
    if structured is not None:
        try:
            result = structured.invoke([system, HumanMessage(content=user_content)])
            if isinstance(result, dict):
                payload = result
            elif hasattr(result, "model_dump"):
                payload = result.model_dump()
            else:
                payload = {
                    key: getattr(result, key, None) for key in JudgeResult.model_fields
                }
            return _payload_to_result(payload, truncated=truncated)
        except Exception:  # noqa: BLE001 - 结构化失败不应放弃本次评估，落到底部文本解析
            pass

    try:
        reply = engine.invoke(
            [
                system,
                HumanMessage(
                    content=(
                        user_content
                        + "\n\n只输出一个 JSON 对象，键为 faithfulness / citation_accuracy / "
                        "unsupported_claims / refusal_correct / reason。"
                        "faithfulness 与 citation_accuracy 为 0~1 的数字，"
                        "refusal_correct 为 true 或 false，unsupported_claims 为字符串数组。"
                        "不要输出任何其它文字，不要用 markdown 代码块。"
                    )
                ),
            ]
        )
        payload = parse_json_object(content_to_text(reply.content))
    except Exception as exc:  # noqa: BLE001 - judge 失败不得中断整轮评估
        return failed_result(f"{type(exc).__name__}: {exc}")

    if payload is None:
        return failed_result("judge 响应不是合法 JSON")

    return _payload_to_result(payload, truncated=truncated)


def get_judge_llm() -> Any:
    """生产路径的 judge 模型。

    ``settings.judge_model_name`` 留空即复用主模型。temperature 固定 0：
    评分需要可复现，同一条答案两次跑出不同分数会让「指标变化」无法归因。
    """
    return get_llm(
        model=settings.judge_model_name,
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
    )


__all__ = [
    "JudgeResult",
    "SYSTEM_PROMPT",
    "JUDGE_METHOD",
    "JUDGE_MAX_TOKENS",
    "MAX_JUDGE_CONTEXT_CHARS",
    "MAX_JUDGE_CLAIMS",
    "MAX_JUDGE_CLAIM_CHARS",
    "PERCENT_SCALE_THRESHOLD",
    "clamp01",
    "clean_claims",
    "build_context",
    "failed_result",
    "judge_answer",
    "get_judge_llm",
]
