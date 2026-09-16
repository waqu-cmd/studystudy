"""Verifier 节点：答案忠实度核查与回退决策（阶段 3）。

职责
----
输入 ``answer + retrieved_chunks``，逐条判断答案中的断言能否在召回块里找到依据，
输出 ``verdict`` 与 ``unsupported_claims``；若判失败，则**顺手把缺口重组成新的
检索词写回 ``search_query``**，供 retriever 在下一轮瞄准缺口重检。

本节点是自纠正循环的**建议提出方**：给出 verdict 与缺口，并把缺口重组成
``search_query`` 写回状态。但「建议是否被采纳」由路由函数按 ``max_retry`` 闸门裁定，
``retry_count`` 的递增则在 retriever 里完成（只有它知道这次检索确实是重检）。
这样分工的原因是：``retry_count`` 会随响应暴露给调用方，语义必须是
**实际已执行的回退次数** —— 被闸门拦下的建议不该计入，否则 ``max_retry=0`` 时
会出现「回退 0 次却记 1 次」的歧义。

路由函数只读 state 选边、不修改状态：它可能被多次调用，任何副作用写在那儿都会
丢失或重复执行（见 ``graph/edges.py`` 的约定）。

为什么要 fail-open（核查失败不拦截）
------------------------------------
核查本身也要调 LLM，它同样会失败。若核查失败就判 fail，会触发一轮完全无意义的重检
（缺口都没识别出来，重检什么），还可能把 retry 配额白白烧光。因此核查未执行时
**放行**当前答案，并在 ``events`` 里留痕。理由：答案已受 ``SYSTEM_PROMPT_RAG``
的强约束（禁止编造、只采用覆盖今天的版本），核查是二次加固而非唯一防线 ——
拦下一个已受约束的答案，收益小于风险。

实测得到的三条结构化输出约束（langchain 0.3.63 + 百炼兼容端点 + deepseek-v4-flash）
-----------------------------------------------------------------------------------
1. ``method="json_schema"`` **可用且结果正确**：实测能准确区分有依据 / 无依据断言。
2. ``method="function_calling"`` **静默返回空对象** —— verdict 与 claims 全为空且
   不抛异常。这是最危险的一种失败：看起来调用成功，实则什么都没核查。
   因此本模块**不使用**该 method。
3. ``method="json_mode"`` 直接 400（``'messages' must contain the word 'json'``）。
4. 纯文本兜底时模型可能输出 ``verdict="partial"`` 这类**枚举外的值**，
   必须经 :func:`normalize_verdict` 归一化 —— 规则是「非 pass 即 fail」。

为什么缺口要由本节点写进 ``search_query``，而不是让 retriever 去读 ``unsupported_claims``
----------------------------------------------------------------------------------
两种做法等价，但前者把「重组检索词」这个策略集中在核查结果产生的地方：检索词怎么拼、
留多长、预算怎么分，都只有本模块知道。retriever 因此保持零改动，只认 ``search_query``
一个字段 —— 它至今不知道自纠正循环的存在。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_llm
from app.core.logging import logger
from app.graph.nodes.synthesizer import (
    content_to_text,
    format_context,
    parse_json_object,
)
from app.graph.state import (
    NODE_VERIFIER,
    VERDICT_FAIL,
    VERDICT_PASS,
    GraphState,
    make_event,
)

VERIFIER_MAX_TOKENS = 1024
"""核查输出很短（一个 verdict + 若干断言），1024 足够且能压住成本。"""

VERIFIER_METHOD = "json_schema"
"""唯一实测可用的结构化输出方式，详见模块 docstring。"""

MAX_CLAIMS = 8
"""单轮最多采纳的未支撑断言条数。防止模型贴回整段答案，把检索词淹没。"""

MAX_CLAIM_CHARS = 80
"""单条断言的长度上限。超出直接截断 —— 不加省略号，标点会污染检索词。"""

MAX_RETRY_QUERY_CHARS = 300
"""重组后检索词的长度上限。

首检的检索词是用户原话（通常 10~30 字），向量检索在这个长度上表现最好。
把缺口断言拼进去后长度会显著膨胀，而查询越长其向量越趋近「平均语义」，
与单个 chunk 的区分度反而下降。300 字是「补足缺口词汇」与「保持查询聚焦」
之间的折中，并保证原问题与缺口各占一半预算。
"""


class VerificationResult(BaseModel):
    """核查结果的结构化契约（走 json_schema 时由服务端约束取值域）。"""

    verdict: Literal["pass", "fail"] = Field(
        description="pass=全部断言均能在参考资料中找到依据；fail=存在无依据断言"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="无依据的断言原文摘录，逐条列出，不得改写或补充解释",
    )
    reason: str = Field(default="", description="一句话说明判定理由")


SYSTEM_PROMPT = """你是答案忠实度核查员。给你「参考资料」与「待核查答案」，\
你要逐句判断答案中的每一条事实性断言能否在参考资料里找到依据。

规则：
1. 只依据参考资料判断，不要使用你自己的先验知识。参考资料里没写的，就是无依据 —— \
即使你认为该说法在现实中成立。
2. 逐句拆解答案。带 [chunk_id] 标记的句子同样要核对：标记存在不代表内容真的被支撑。
3. 纯格式性表述（如「根据以上资料」「综上所述」「如需进一步了解」）不算断言，忽略。
4. 无依据的断言请**原文摘录**到 unsupported_claims，不要改写、不要补充解释、不要合并多条。
5. 全部断言都有依据 → verdict=pass，且 unsupported_claims 为空列表。
6. 只要存在任何一条无依据断言 → verdict=fail。
7. verdict 只能取 "pass" 或 "fail"，不要使用 "partial" 等其它取值。"""

# 断言列表里的序号、项目符号、包裹引号，都是模型常见的格式化噪声，清洗掉。
_CLAIM_NOISE_RE = re.compile(r"^[\s\-•*·–—]*?(?:\d+[.、)）]\s*)?")
_CLAIM_PUNCT_RE = re.compile(r"[\s,，。.、;；:：!！?？\"'“”‘’()（）\[\]【】—\-]+")

_PASS_TOKENS = frozenset(
    {
        "pass",
        "passed",
        "ok",
        "true",
        "yes",
        "supported",
        "faithful",
        "grounded",
        "通过",
        "有依据",
        "无问题",
    }
)
"""判定为「通过」的取值白名单。

采用白名单而非黑名单：核查是安全侧的判断题，未知取值一律视为 fail 更保守。
实测模型会输出 ``partial`` 这类中间态 —— 它必须落到 fail，因为「部分有依据」
恰恰意味着确实存在无依据的断言，正是回退要处理的场景。
"""


def normalize_verdict(raw: Any) -> str:
    """把模型的原始判定归一为 ``pass`` / ``fail``。

    「非 pass 即 fail」：未知取值（``partial``、空串、中文变体）全部归为 fail。
    fail 只会触发一次定向重检，代价远小于放过一条编造内容。
    """
    text = str(raw or "").strip().lower()
    return VERDICT_PASS if text in _PASS_TOKENS else VERDICT_FAIL


def clean_claims(raw: Any) -> list[str]:
    """清洗模型给出的无依据断言：去格式噪声、去重、限量、限长。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]

    claims: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _CLAIM_NOISE_RE.sub("", str(item or "").strip())
        text = text.strip().strip("\"'“”‘’「」《》").strip()
        if len(text) > MAX_CLAIM_CHARS:
            text = text[:MAX_CLAIM_CHARS].rstrip()
        if not text or text in seen:
            continue
        seen.add(text)
        claims.append(text)
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def claim_key(text: str) -> str:
    """断言的去重键：去掉空白与标点后的文本。

    实测第二轮核查会把同一句断言多写一个逗号，仅按字面比较会让重复项绕过差集
    （三轮下来 unsupported_claims 仍会堆到 4 条）。用归一化后的键比较，
    才能识别「语义相同、标点不同」的重复。

    这里只做标点归一，不做语义归一 —— 后者需要 embedding 或模型判断，
    代价高且会引入不确定性；标点差异恰好是实测中唯一的噪声来源。
    """
    return _CLAIM_PUNCT_RE.sub("", str(text or ""))


def parse_verification_payload(text: str) -> dict[str, Any] | None:
    """从纯文本响应里抠出 JSON 对象；解析不出来返回 None。

    阶段 5 起真正的实现在 :func:`synthesizer.parse_json_object` —— supervisor
    的路由决策需要同一套容错，两处各写一份必然漂移。这里保留原名作为薄包装：
    阶段 3 的测试与调用方依赖这个名字，改名的收益远小于风险。
    """
    return parse_json_object(text)


def build_retry_query(
    query: str,
    claims: list[str],
    *,
    max_chars: int = MAX_RETRY_QUERY_CHARS,
) -> str:
    """把「原问题 + 未支撑断言」重组成下一轮的检索词。

    为什么不是拿原问题重检：同一检索词只会召回同一批块，回退退化成空转。
    缺口断言本身就是「还缺什么信息」的自然语言描述，用它才能定向补齐首轮遗漏。

    预算分配：原问题与缺口各占一半上限。原问题过长时先截断 ——
    保留完整的缺口更有利于定向补齐。
    """
    head = (query or "").strip()[: max_chars // 2]

    tail: list[str] = []
    used = len(head)
    seen: set[str] = {head}
    for claim in claims:
        text = (claim or "").strip()
        if not text or text in seen:
            continue
        if used + len(text) + 1 > max_chars:
            break
        seen.add(text)
        tail.append(text)
        used += len(text) + 1

    return " ".join(part for part in [head, *tail] if part).strip()


def _structured_runner(engine: Any) -> Callable[..., Any] | None:
    """拿到模型的结构化输出通道；不支持时返回 None。

    这里**不**捕获 invoke 阶段的异常 —— ``with_structured_output`` 成功但请求被
    端点拒绝（例如不支持 response_format）只在 invoke 时暴露，那种情况由
    :func:`_verify` 回退到文本解析。
    """
    builder = getattr(engine, "with_structured_output", None)
    if not callable(builder):
        return None
    try:
        return builder(VerificationResult, method=VERIFIER_METHOD)
    except Exception as exc:  # noqa: BLE001 - 老版本 langchain 不认 method 参数
        logger.warning("结构化输出不可用，将回退到文本解析 | {}", exc)
        return None


def _extract(result: Any) -> tuple[Any, Any, Any]:
    """从结构化输出结果取出 verdict / claims / reason，兼容对象与字典两种形态。"""
    if isinstance(result, dict):
        return (
            result.get("verdict"),
            result.get("unsupported_claims"),
            result.get("reason"),
        )
    return (
        getattr(result, "verdict", None),
        getattr(result, "unsupported_claims", None),
        getattr(result, "reason", ""),
    )


def _verify(engine: Any, answer: str, chunks: list[Any]) -> tuple[str, list[str], str]:
    """执行一次核查，返回 ``(verdict, claims, reason)``。

    两条路径：结构化输出（生产路径，实测可靠）与纯文本 JSON 解析（兜底 ——
    结构化通道报错时不放弃核查）。两条都失败则抛异常，由调用方 fail-open。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(content=SYSTEM_PROMPT)
    user_content = (
        f"【参考资料】\n{format_context(chunks)}\n\n【待核查答案】\n{answer}"
    )

    structured = _structured_runner(engine)
    if structured is not None:
        try:
            result = structured.invoke([system, HumanMessage(content=user_content)])
            raw_verdict, raw_claims, raw_reason = _extract(result)
            return (
                normalize_verdict(raw_verdict),
                clean_claims(raw_claims),
                str(raw_reason or ""),
            )
        except Exception as exc:  # noqa: BLE001 - 结构化通道失败不应放弃核查
            logger.warning(
                "结构化核查调用失败，回退到文本 JSON 解析 | {}", type(exc).__name__
            )

    # 兜底：让模型直接吐 JSON。此路径下模型可能给出枚举外的 verdict，
    # normalize_verdict 会把它收敛到 fail。
    reply = engine.invoke(
        [
            system,
            HumanMessage(
                content=(
                    user_content
                    + "\n\n只输出一个 JSON 对象，键为 verdict / unsupported_claims / reason，"
                    'verdict 只能取 "pass" 或 "fail"。'
                    "不要输出任何其它文字，不要用 markdown 代码块。"
                )
            ),
        ]
    )
    payload = parse_verification_payload(content_to_text(reply.content))
    if payload is None:
        raise ValueError("核查响应不是合法 JSON")
    return (
        normalize_verdict(payload.get("verdict")),
        clean_claims(payload.get("unsupported_claims")),
        str(payload.get("reason") or ""),
    )


def verifier_node(
    state: GraphState,
    *,
    llm: Any | None = None,
    max_retry: int | None = None,
) -> dict:
    """核查节点。

    ``llm`` 允许注入替身，测试因此不需要网络。生产默认取
    ``settings.verifier_model_name``（留空即复用主模型），temperature 固定 0
    —— 核查是判断题，随机性只会让判定不稳定。

    ``max_retry`` **只用于日志与事件文案**，不参与任何判定（闸门在路由函数里）。
    builder 会把编译期实际生效的阈值传进来，保证文案与真实闸门一致 ——
    否则测试传 ``max_retry=2`` 而文案写着 ``settings.max_retry=3``，会误导排错。
    """
    limit = settings.max_retry if max_retry is None else max(0, int(max_retry))
    answer = (state.get("answer") or "").strip()
    chunks = list(state.get("retrieved_chunks") or [])
    query = state.get("query") or ""
    retry_count = int(state.get("retry_count") or 0)

    # 理论上不可达（路由已在 direct / 无资料 / 生成失败时拦下），保留为防御
    if not answer or not chunks:
        return {
            "verdict": VERDICT_PASS,
            "events": [make_event(NODE_VERIFIER, "无可核查内容，跳过核查")],
        }

    engine = (
        llm
        if llm is not None
        else get_llm(
            model=settings.verifier_model_name,
            temperature=0.0,
            max_tokens=VERIFIER_MAX_TOKENS,
        )
    )

    try:
        verdict, claims, reason = _verify(engine, answer, chunks)
    except Exception as exc:  # noqa: BLE001 - fail-open，见模块 docstring
        logger.exception("核查未执行 | query={!r}", query[:30])
        return {
            "verdict": VERDICT_PASS,
            "events": [
                make_event(
                    NODE_VERIFIER,
                    f"核查未执行（{type(exc).__name__}），已放行当前答案",
                )
            ],
        }

    if verdict == VERDICT_PASS:
        logger.info(
            "verifier | pass | query={!r} | chunks={} | reason={}",
            query[:30],
            len(chunks),
            reason[:60],
        )
        return {
            "verdict": VERDICT_PASS,
            "events": [make_event(NODE_VERIFIER, "核查通过：全部断言均有依据")],
        }

    # 判 fail 却拿不出具体缺口 → 没有可重检的方向，回退只会空转。
    # 此处收敛为 pass，并在事件里写明是「无缺口」而非「已通过」。
    if not claims:
        logger.warning(
            "verifier | fail 但无缺口明细，按通过处理 | query={!r} | reason={}",
            query[:30],
            reason[:60],
        )
        return {
            "verdict": VERDICT_PASS,
            "events": [
                make_event(NODE_VERIFIER, "判定不通过但未给出具体缺口，跳过回退")
            ],
        }

    retry_query = build_retry_query(query, claims)
    if not retry_query:
        return {
            "verdict": VERDICT_PASS,
            "events": [
                make_event(NODE_VERIFIER, "无法重组检索词，跳过回退")
            ],
        }

    # 回退判定用本轮检出的原始缺口，但写回状态时与历史记录做差集。
    # 多轮回退中同一批缺口常被反复检出 —— 实测三轮回退会往 unsupported_claims
    # 里塞进三份相同内容（8 条里大量重复），反而掩盖了「到底缺什么」这一核心信息。
    # 比较用 claim_key（忽略标点）而非字面，因为模型每轮的标点并不稳定。
    known_keys = {
        claim_key(item) for item in (state.get("unsupported_claims") or [])
    }
    fresh_claims = [
        claim for claim in claims if claim_key(claim) not in known_keys
    ]

    logger.info(
        "verifier | fail | query={!r} | 本轮缺口={} | 新增={} | 已回退={}/{} | 新检索词={!r}",
        query[:30],
        len(claims),
        len(fresh_claims),
        retry_count,
        limit,
        retry_query[:60],
    )

    return {
        "verdict": VERDICT_FAIL,
        # 增量语义（append_or_reset）：只写相对历史记录真正新增的缺口。
        # fresh_claims 为空是合法的 —— reducer 得到 [] 时状态保持不变。
        "unsupported_claims": fresh_claims,
        # 改写检索词是「换角度重检」的落点；retriever 节点读它，零改动。
        # 这里**不**写 retry_count —— 递增由 retriever 在实际执行重检时完成。
        "search_query": retry_query,
        "events": [
            make_event(
                NODE_VERIFIER,
                f"核查未通过：{len(claims)} 条断言无依据，已重组检索词准备重检"
                f"（第 {retry_count + 1}/{limit} 次）",
            )
        ],
    }


__all__ = [
    "VerificationResult",
    "SYSTEM_PROMPT",
    "VERIFIER_METHOD",
    "VERIFIER_MAX_TOKENS",
    "MAX_CLAIMS",
    "MAX_CLAIM_CHARS",
    "MAX_RETRY_QUERY_CHARS",
    "normalize_verdict",
    "clean_claims",
    "claim_key",
    "parse_verification_payload",
    "build_retry_query",
    "verifier_node",
]
