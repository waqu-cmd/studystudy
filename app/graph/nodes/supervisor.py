"""Supervisor 节点：意图识别 + 子问题拆解（阶段 5 由规则升级为 LLM 驱动）。

职责
----
决定一次提问怎么处理，产出三样东西：

1. ``route``：``retrieve``（要走知识库）或 ``direct``（知识库里没有答案可言）。
2. ``sub_queries``：拆解出的子问题。长度 ≥2 时由 ``edges.route_after_supervisor``
   用 ``Send`` 并行分发到 retriever，每个子问题一路。
3. ``route_reason``：人类可读的理由，直接进日志与响应。

三层兜底，逐层下沉
------------------
``结构化输出（json_schema）`` → ``文本 JSON 解析`` → ``阶段 2 的规则路由``。
任何一层失败都退回下一层，因此**永远有一个确定的结论**，不存在「路由不出来」
的中间态。规则层不只是兜底：它同时是下面那条硬闸门的判据。

实测得到的结构化输出约束（与 verifier 同源，langchain 0.3.63 + 百炼兼容端点）
--------------------------------------------------------------------------
``method="json_schema"`` 可用（本项目采用）；``function_calling`` 会**静默返回
空对象**（不抛异常，却什么都没决策）；``json_mode`` 直接 400。详见 verifier 模块
docstring —— 那三条结论对整个项目通用，这里不重复踩一遍。

为什么默认走检索，以及那条硬闸门
--------------------------------
本系统的核心指标是忠实度，而「不检索直接生成」等价于允许模型凭空作答。
因此：

- 规则层只在**能明确判定与知识库无关**时才判 direct（寒暄 / 元问题 / 空串）。
- LLM 层若判 direct，但规则层命中了 ``KB_HINTS`` 业务关键词，**强制改判 retrieve**
  并把拦截理由写进 ``route_reason``。代价是极少数真寒暄多跑一次检索（几十毫秒），
  收益是任何业务问题都不会被误放行给模型自由发挥。

这条闸门是刻意的**不对称**设计：LLM 判 retrieve 而规则判 direct 时听 LLM 的
（规则里的「长度 ≤3 字」只是没有模型时的启发式，例如「Q2 呢？」在多轮语境下
本就需要检索）；反向则不信任。

为什么 ``with_answer=False`` 时完全不调 LLM
-----------------------------------------
该模式是阶段 7 的评估入口，契约是「只测检索指标，不消耗 LLM 配额」。
若 supervisor 在此也发一次模型请求，评估成本会随题量线性增长，且结论不再可复现。
因此评估模式下直接走规则路由，``sub_queries`` 恒为空 —— 评估路径永远是单路检索。
（这也意味着：要评估「拆解」的收益，必须用 ``with_answer=true``。）
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.core.logging import logger
from app.graph.nodes.synthesizer import content_to_text, parse_json_object
from app.graph.state import (
    NODE_SUPERVISOR,
    ROUTE_DIRECT,
    ROUTE_RETRIEVE,
    GraphState,
    make_event,
)

SHORT_QUERY_CHARS = 3
"""去空白后不超过该长度、且不含业务关键词的问题视为寒暄噪音。"""

SUPERVISOR_METHOD = "json_schema"
"""唯一实测可用的结构化输出方式，详见 verifier 模块 docstring。"""

SUPERVISOR_MAX_TOKENS = 512
"""路由输出很短（一个枚举 + 最多 3 条子问题），512 足够且能压住成本。"""

MAX_SUB_QUERIES = 3
"""单轮最多采纳的子问题数。

不是「越多越好」：每多一路就多一次 embedding 请求与一次向量检索，召回块数量
线性增长，最终全部塞进同一个生成 prompt。3 路是「覆盖复合问题的多个诉求」
与「不把 prompt 撑爆」之间的折中，也覆盖了蓝图给出的示例（2 路）。
"""

MAX_SUB_QUERY_CHARS = 60
"""单条子问题的长度上限。

子问题需要补全主语与时间范围，因此比首检检索词（用户原话，通常 10~30 字）略长；
但继续膨胀会让它的向量趋近「平均语义」，与单个 chunk 的区分度下降。
超长直接截断 —— 不加省略号，否则标点会被当成检索词的一部分。
"""

KB_HINTS: frozenset[str] = frozenset(
    {
        # 文档类型
        "政策", "制度", "规定", "流程", "标准", "规范", "条款", "手册", "指南", "办法",
        # 报销与费用
        "报销", "发票", "差旅", "住宿", "招待", "礼品", "上限", "限额", "补贴",
        # 销售与考核
        "返点", "提成", "结算", "考核", "客户", "归属", "保护期", "合同", "签约", "续约",
        # 审批与权限
        "审批", "权限", "报备",
        # 信息与安全
        "密码", "账号", "设备", "数据", "外发", "网盘", "加密", "安全", "违规",
        # 时效相关（阶段 7 陷阱题大量使用）
        "有效期", "失效", "版本", "季度", "生效", "过期", "追溯",
        # 元信息
        "知识库", "文档", "资料",
    }
)
"""业务关键词。命中即判定为业务问题 —— 这是召回侧的强信号，
因此它优先于寒暄规则，避免「你好，请问…」这类礼貌前缀掩盖真实意图；
阶段 5 起它还兼任 LLM 判 direct 时的硬闸门判据。"""

META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(hi|hello|hey|yo|thanks|thank you)\b", re.IGNORECASE),
    re.compile(r"^(你好|您好|哈喽|嗨|在吗|在不在|早上好|中午好|下午好|晚上好)"),
    re.compile(r"^(谢谢|多谢|感谢|辛苦了|再见|拜拜|goodbye)"),
    re.compile(
        r"(你是谁|你叫什么|你是做什么的|你能做什么|你会什么|你有什么功能|"
        r"介绍.{0,3}你自己|自我介绍|怎么用你|你是什么模型)"
    ),
)
"""寒暄与「关于助手自身」的元问题。共性是答案不在企业知识库里。"""

DIRECT_TOKENS: frozenset[str] = frozenset(
    {
        "direct",
        "direct_answer",
        "directanswer",
        "no_retrieval",
        "chitchat",
        "chat",
        "greeting",
        "smalltalk",
        "none",
        "寒暄",
        "闲聊",
        "直接回答",
        "直接",
        "无需检索",
    }
)
"""判为 direct 的取值白名单。

采用白名单而非黑名单，方向与 verifier 相反：核查是安全侧的判断题，「非 pass
即 fail」；路由则「非 direct 即 retrieve」—— 漏检（该查没查）会直接导致编造，
多查一次只花几十毫秒。文本兜底路径下模型可能给出 ``direct_answer``
（蓝图里的原始命名）等变体，因此白名单要覆盖到。
"""

_SUB_QUERY_NOISE_RE = re.compile(r"^[\s\-•*·–—]*?(?:\d+[.、)）]\s*)?")
_SUB_QUERY_PUNCT_RE = re.compile(r"[\s,，。.、;；:：!！?？\"'“”‘’()（）\[\]【】—\-]+")

_WRAPPER_CHARS = "\"'“”‘’「」『』《》〈〉()（）[]【】〔〕"
"""子问题外层的包裹符号。

半角/全角引号与各种括号成对出现，模型把它们当「引用」用。实测最常见的两种是
``"..."`` 与 ``「...」``。按字符集合 strip 而不是配对匹配：模型经常只写一半，
按字符剥反而更鲁棒。
"""


def _strip_wrapping(text: str) -> str:
    """反复剥掉外层的序号与包裹符号，直到不再变化（最多 3 轮）。

    为什么需要循环：实测会出现 ``1. 「xxx」`` 这种双重包裹 ——
    只剥一次要么残留序号、要么残留半个引号，而残留符号会直接变成检索词的一部分，
    拉低向量召回质量。
    """
    for _ in range(3):
        before = text
        text = _SUB_QUERY_NOISE_RE.sub("", text.strip())
        text = text.strip(_WRAPPER_CHARS).strip()
        if text == before:
            break
    return text


class SupervisorDecision(BaseModel):
    """路由决策的结构化契约（走 json_schema 时由服务端约束取值域）。"""

    intent: Literal["retrieve", "direct"] = Field(
        description=(
            "retrieve=需要查企业知识库（任何与企业内部制度/政策/流程相关的输入，"
            "包括看似简单的事实型问题）；direct=与知识库完全无关的寒暄或关于助手自身的问题"
        )
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description=(
            "intent=retrieve 且问题确有多个独立诉求时，拆出的 2~3 条子问题，"
            "每条自带主语与时间范围、可独立检索；单一诉求时留空列表"
        ),
    )
    reason: str = Field(default="", description="一句话说明判定理由，用简体中文")


SYSTEM_PROMPT = """你是企业知识库问答系统的调度器（Supervisor）。\
你的唯一任务是决定这次输入要不要查企业知识库；如果要查，再判断是否需要拆成多条可并行检索的子问题。

【第一步：判定 intent】
- "direct"：与企业内部制度、政策、流程完全无关的输入 —— 寒暄（你好 / 谢谢）、
  告别、以及询问助手自身能力的问题（你是谁 / 你会做什么）。
- "retrieve"：其余全部。**只要有任何一处指向企业内部信息，就必须选 retrieve。**
  特别注意：看似简单的事实型问题（例如「差旅报销上限是多少」）依然属于 retrieve ——
  答案在企业文档里，不查就必然编造。

【第二步：拆解 sub_queries】（仅当 intent="retrieve" 时填写）
1. 只有一个信息诉求时，sub_queries 留空列表。**不要为了凑数而强行拆分。**
2. 确属复合问题（一个问句里有多个诉求，或多个主体 / 时间 / 维度需要分别查证）时，
   拆成 2~3 条**可独立检索**的子问题。每条都要补全主语与时间范围，
   保证脱离原句也能被检索系统理解。
   例：「2026Q2 政策相比 Q1 有哪些变化，对华东区的影响是什么」
   应拆为 ["2026Q2 销售政策相比 2026Q1 的具体变化", "2026Q2 销售政策对华东区的区域补贴调整"]
3. 子问题用简体中文，每条不超过 40 字，不要编号、不要引号、不要附加解释。
4. 拆解不得改变原意，不得引入原句没有的时间、区域或主体。"""


def decide_route(query: str) -> tuple[str, str]:
    """规则路由决策（阶段 2 保留至今），返回 ``(route, reason)``。

    两个用途：LLM 不可用 / 输出不合法时的**兜底**，以及 LLM 判 direct 时的
    **硬闸门**判据（见 :func:`supervisor_node`）。

    决策表（自上而下，首个命中者生效）
    ----------------------------------
    ====  ==================================  =========  ==============================
    序号  条件                                结果       理由
    ====  ==================================  =========  ==============================
    1     问题为空                             direct     无内容可检索
    2     命中 KB_HINTS 业务关键词             retrieve   业务问题，强信号（优先于寒暄）
    3     命中 META_PATTERNS 寒暄/元问题       direct     与知识库无关
    4     去空白后长度 <= SHORT_QUERY_CHARS    direct     过短且不含业务关键词
    5     其余                                 retrieve   默认：忠实度优先
    ====  ==================================  =========  ==============================

    规则 2 优先于规则 3 是刻意的：``"你好，请问差旅报销上限是多少"`` 同时命中两者，
    必须走检索，否则会把一个真业务问题当成寒暄放行。

    抽成纯函数（不读 state、不写日志）是为了让
    ``tests/test_graph_nodes.py::test_supervisor_routing`` 能直接对决策表
    做表驱动断言，不必构造整张图。
    """
    text = (query or "").strip()

    if not text:
        return ROUTE_DIRECT, "问题为空，无内容可检索"

    matched_hints = sorted(hint for hint in KB_HINTS if hint in text)
    if matched_hints:
        return ROUTE_RETRIEVE, f"命中业务关键词：{'/'.join(matched_hints[:3])}"

    for pattern in META_PATTERNS:
        if pattern.search(text):
            return ROUTE_DIRECT, "寒暄或关于助手自身的元问题，与企业知识库无关"

    if len(text) <= SHORT_QUERY_CHARS:
        return ROUTE_DIRECT, f"长度 {len(text)} ≤ {SHORT_QUERY_CHARS} 且不含业务关键词"

    return ROUTE_RETRIEVE, "默认走检索（忠实度优先）"


def normalize_intent(raw: Any) -> str | None:
    """把模型给的 intent 归一为 ``retrieve`` / ``direct``；无法识别返回 ``None``。

    返回 ``None`` 而不是硬猜一个值，是为了让调用方**整体放弃**这次 LLM 决策、
    回退到规则路由 —— 一个连意图都没说清的响应，它的子问题也不值得采信。
    """
    token = str(raw or "").strip().lower()
    if not token:
        return None
    if token in DIRECT_TOKENS:
        return ROUTE_DIRECT
    if token in {"retrieve", "retrieval", "search", "kb", "rag", "检索", "知识库", "查询"}:
        return ROUTE_RETRIEVE
    return None


def clean_sub_queries(
    raw: Any,
    *,
    max_count: int = MAX_SUB_QUERIES,
    max_chars: int = MAX_SUB_QUERY_CHARS,
) -> list[str]:
    """把模型给的子问题清洗成可直接送检索的文本列表。

    清洗四件事，都是实测会出现的噪声：

    1. **序号与项目符号**（``1.`` / ``-`` / ``•``）—— 留着会被当成检索词的一部分。
    2. **包裹符号** —— 模型很爱把每条子问题用引号或书名号括起来，
       半角与全角都出现过（``"..."`` / ``「...」`` / ``《...》``），
       而且经常只写半边，因此按字符集合剥而不是配对匹配。
    3. **重复项**：按「忽略标点后」的键去重，与 verifier 的 ``claim_key`` 同思路
       （模型常把同一句多写一个逗号）。
    4. **超长**：直接截断到上限，不补省略号（省略号会污染检索词）。

    单条字符串也接受（模型偶尔把列表写成一段文字），按条处理而不是按字符切。
    """
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = []

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _strip_wrapping(str(item or ""))
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        key = _SUB_QUERY_PUNCT_RE.sub("", text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_count:
            break
    return result


def _structured_runner(engine: Any) -> Callable[..., Any] | None:
    """拿到模型的结构化输出通道；不支持时返回 None。

    与 verifier 的同类函数保持一致：这里**不**捕获 invoke 阶段的异常 ——
    ``with_structured_output`` 构造成功但请求被端点拒绝只在 invoke 时暴露，
    那种情况由 :func:`_decide_with_llm` 回退到文本解析。
    """
    builder = getattr(engine, "with_structured_output", None)
    if not callable(builder):
        return None
    try:
        return builder(SupervisorDecision, method=SUPERVISOR_METHOD)
    except Exception as exc:  # noqa: BLE001 - 老版本 langchain 不认 method 参数
        logger.warning("结构化路由通道不可用，将回退到文本 JSON 解析 | {}", exc)
        return None


def _field(result: Any, name: str) -> Any:
    """从结构化输出结果里取字段，兼容 pydantic 对象与字典两种形态。"""
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def _decide_with_llm(
    engine: Any, text: str
) -> tuple[str | None, list[str], str] | None:
    """让 LLM 做一次路由决策。

    Returns:
        ``(intent, sub_queries, reason)``；``intent`` 为 ``None`` 表示这次决策
        不可用（调用方回退到规则）。返回 ``None`` 表示连响应都没拿到。
    """
    system = SystemMessage(content=SYSTEM_PROMPT)
    user = HumanMessage(content=f"【用户输入】\n{text}")

    structured = _structured_runner(engine)
    if structured is not None:
        try:
            result = structured.invoke([system, user])
            return (
                normalize_intent(_field(result, "intent")),
                clean_sub_queries(_field(result, "sub_queries")),
                str(_field(result, "reason") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - 结构化通道失败不应放弃路由
            logger.warning(
                "结构化路由调用失败，回退到文本 JSON 解析 | {}", type(exc).__name__
            )

    # 兜底：让模型直接吐 JSON
    try:
        reply = engine.invoke(
            [
                system,
                HumanMessage(
                    content=(
                        f"{text}\n\n只输出一个 JSON 对象，键为 intent / sub_queries / reason，"
                        'intent 只能取 "retrieve" 或 "direct"。不要输出其它文字。'
                    )
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - 模型不可用时交给规则路由
        logger.warning(
            "supervisor 模型调用失败，回退到规则路由 | {}", type(exc).__name__
        )
        return None

    payload = parse_json_object(content_to_text(getattr(reply, "content", reply)))
    if payload is None:
        logger.info("supervisor 响应不是合法 JSON，回退到规则路由")
        return None

    return (
        normalize_intent(payload.get("intent")),
        clean_sub_queries(payload.get("sub_queries")),
        str(payload.get("reason") or ""),
    )


def supervisor_node(state: GraphState, *, llm: Any | None = None) -> dict:
    """路由节点：判定意图并按需拆解子问题。

    ``llm`` 允许注入替身，测试因此不需要网络。生产默认取主模型（temperature=0
    —— 路由是判断题，随机性只会让同一问题时而走检索时而走 direct）。
    """
    text = (state.get("query") or "").strip()

    # 空输入连 LLM 都不必调用：没有内容可供检索，也没有可拆解的诉求
    if not text:
        logger.info("supervisor | route=direct | reason=问题为空")
        return _payload(text, ROUTE_DIRECT, [], "问题为空，无内容可检索")

    rule_route, rule_reason = decide_route(text)
    route: str | None = None
    sub_queries: list[str] = []
    reason = ""

    if state.get("with_answer", True):
        engine = (
            llm
            if llm is not None
            else get_llm(temperature=0.0, max_tokens=SUPERVISOR_MAX_TOKENS)
        )
        decision = _decide_with_llm(engine, text)
        if decision is not None:
            route, sub_queries, reason = decision
    else:
        # 评估模式：保住「不消耗 LLM 配额」的契约，详见模块 docstring
        logger.debug("评估模式：supervisor 走规则路由，不调用 LLM")

    if route is None:
        route, sub_queries, reason = rule_route, [], rule_reason

    # ---------------- 硬闸门：业务问题不得被判为 direct ----------------
    if route == ROUTE_DIRECT and rule_route == ROUTE_RETRIEVE:
        logger.warning(
            "supervisor 判 direct 但命中业务关键词，已强制改判 retrieve | query={!r}",
            text[:30],
        )
        route = ROUTE_RETRIEVE
        reason = f"{reason}（已拦截：{rule_reason}）"

    if route != ROUTE_RETRIEVE:
        sub_queries = []

    return _payload(text, route, sub_queries, reason)


def _payload(
    text: str, route: str, sub_queries: list[str], reason: str
) -> dict:
    """组装 supervisor 的返回值，顺带把决策写进执行事件。

    ``search_query`` 的取值规则：

    - 拆出 ≥2 条子问题 → 写用户原话。此时真正送去检索的是各分支子问题
      （由 ``Send`` 的 payload 承载），``search_query`` 只作为「本轮检索意图」的
      代表值留下，供 verifier 回退时作为参照。
    - 恰好 1 条子问题 → 写该子问题。它是模型补全主语/时间范围后的版本，
      比用户原话更适合检索（相当于一次免费的查询改写）。
    - direct → 写原话，仅为字段完整性，检索不会发生。
    """
    search_query = sub_queries[0] if (route == ROUTE_RETRIEVE and len(sub_queries) == 1) else text

    if sub_queries:
        detail = f"；拆解为 {len(sub_queries)} 个子问题，{'并行检索' if len(sub_queries) >= 2 else '单路检索'}"
    else:
        detail = ""

    logger.info(
        "supervisor | route={} | sub_queries={} | reason={} | query={!r}",
        route,
        len(sub_queries),
        reason,
        text[:30],
    )

    return {
        "query": text,
        # search_query 在此初始化；阶段 3 的 verifier 会在回退前改写它
        "search_query": search_query,
        "route": route,
        "route_reason": reason,
        "sub_queries": sub_queries,
        "events": [
            make_event(NODE_SUPERVISOR, f"路由决策：{route}（{reason}）{detail}")
        ],
    }


def reset_turn_node(state: GraphState) -> dict:
    """每轮入口的清零节点。

    为什么必须是一个独立节点
    ------------------------
    节点的返回值是一个 dict，同一通道只能写一次 —— 既要清空 ``events``（写 None）
    又要记录本轮自己的事件，两者写在同一个 dict 里会互相覆盖。因此把「清零」
    单独成节点，supervisor 才能在清零之后正常追加自己的事件。

    为什么阶段 2 不省略
    ------------------
    阶段 2 没有 checkpointer，每轮状态天然是干净的，这个节点看似多余；但阶段 5
    挂上 MemorySaver 后，累加字段会跨轮保留（实测第二轮 events 变成上一轮的两倍），
    而「跨轮污染」是极难定位的一类 bug。阶段 5 已真实挂上 checkpointer，
    本节点从此是**必需品**。

    阶段 5 的补充
    ------------
    ``route`` / ``route_reason`` / ``sub_queries`` 也在此清零：上一轮若是复合问题，
    残留的子问题列表会让下一轮的 supervisor 判断失真（更糟的是它带 reducer，
    不清就会与本轮的拆解结果累加）。
    """
    return {
        # None 即清空（见 state.append_or_reset），不是「无更新」
        "events": None,
        "retrieved_chunks": None,
        "unsupported_claims": None,
        "sub_queries": None,
        # 非累加字段在此显式归零，避免上一轮的残留值被下游误读
        "retry_count": None,
        "retrieval_attempts": None,
        "verdict": "",
        "answer": "",
        "citations": [],
        "llm_model": "",
        "error": None,
        "route": "",
        "route_reason": "",
        "search_query": "",
    }


__all__ = [
    "SHORT_QUERY_CHARS",
    "KB_HINTS",
    "META_PATTERNS",
    "DIRECT_TOKENS",
    "SUPERVISOR_METHOD",
    "SUPERVISOR_MAX_TOKENS",
    "MAX_SUB_QUERIES",
    "MAX_SUB_QUERY_CHARS",
    "SupervisorDecision",
    "SYSTEM_PROMPT",
    "decide_route",
    "normalize_intent",
    "clean_sub_queries",
    "supervisor_node",
    "reset_turn_node",
]
