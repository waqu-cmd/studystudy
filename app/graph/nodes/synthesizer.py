"""Synthesizer 节点：整合生成 + 引用绑定（阶段 2）。

本模块同时承载三样东西，它们都围绕同一件事 —— 「把召回块变成带引用的答案」：

1. 提示词（``SYSTEM_PROMPT_*``）
2. 上下文与答案文本的拼装/解析（``format_context`` / ``content_to_text`` /
   ``extract_citations``）
3. 节点本体（``synthesizer_node``）

为什么这些辅助函数不单独成模块：蓝图的 ``graph/`` 目录清单里只有
``state.py`` / ``builder.py`` / ``edges.py`` / ``nodes/``，没有 ``prompts.py``
一类的文件。为了不偏离既定结构，把它们与唯一的使用方放在一起。
阶段 3 的 verifier 需要复用 ``content_to_text`` 与 ``extract_citations``，
从本模块 import 即可（同级节点间复用，不构成循环依赖）。

引用为什么按 chunk_id 的格式提取，而不是匹配方括号
------------------------------------------------
阶段 1 实测：模型会把 ``[chunk_id]`` 写成 ``【来源：chunk_id】`` 等变体。
与其反复调 prompt 赌模型听话，不如利用 ``chunk_id`` 的确定性格式
（``{doc_id}_p{index}``，见 rag/chunker.py）直接扫描答案文本 —— 任何包裹符号
都能识别。同时，只有出现在本轮召回集合里的 id 才会被采纳，
模型编造的引用会被丢弃（这是阶段 3 引用准确率指标的前提）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.graph.state import NODE_SYNTHESIZER, ROUTE_DIRECT, GraphState, make_event
from app.schemas.query import Citation, RetrievedChunk

REFUSE_TEXT = "根据现有资料无法确认。"
"""检索无结果时的固定答复。不走 LLM —— 没有资料就没有可生成的内容，
调用模型只会给它编造的机会，同时白付一次 token。"""

SYSTEM_PROMPT_RAG = """你是企业知识库问答助手，回答必须严格基于给定的参考资料。

规则：
1. 引用格式严格为「半角方括号 + chunk_id」，紧跟结论，例如：新客户返点为 3.5% [sales_policy_2026q3_p1]。
   不要使用【来源：xxx】、脚注、全角括号或其它变体。
2. 参考资料中没有的信息一律不得编造。若资料不足以回答，直接说明「根据现有资料无法确认」。
3. 同一政策可能存在多个季度版本。只采用有效期覆盖今天的版本，不得把不同版本的内容混在一起。
4. 回答用简体中文，先给结论，再给依据。"""

SYSTEM_PROMPT_DIRECT = """你是企业知识库问答助手。当前问题被判定为寒暄或关于助手自身的元问题，
因此没有提供任何参考资料。

规则：
1. 直接、简短地回答，一到两句话即可。
2. 绝对不要提及任何具体制度条款、数字、金额、比例或日期 —— 你此刻没有任何资料依据。
3. 如果用户其实想问企业制度类问题，只回复「请补充你想了解的具体制度或政策名称」。
4. 回答用简体中文。"""

# chunk_id 的确定性格式是 f"{doc_id}_p{index}"（见 rag/chunker.py）。
# 直接按该模式扫描答案，任何包裹符号下的引用都能识别。
_CITATION_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*_p\d+")

# 从自然语言响应里抠 JSON 对象用的粗匹配。三种形态都要能命中：
# 裸 JSON、```json 围栏包裹、JSON 前后带解释性文字。
# verifier 的核查结论与阶段 5 supervisor 的路由决策共用同一个匹配器。
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def format_context(hits: list[RetrievedChunk]) -> str:
    """把召回结果拼成带元信息的上下文块。

    显式带上《文档名》与版本号，是为了让模型能分辨同一政策的多个季度版本 ——
    这正是阶段 7 陷阱题要考察的能力。有效期也一并给出，使「只采用覆盖今天的
    版本」这条规则有据可依，而不是让模型猜。
    """
    blocks = [
        (
            f"[{hit.chunk_id}] 文档《{hit.doc_title}》版本={hit.version} "
            f"小节={hit.heading} 有效期={hit.effective_date}~{hit.expire_date}\n{hit.text}"
        )
        for hit in hits
    ]
    return "\n\n".join(blocks)


def build_user_prompt(query: str, hits: list[RetrievedChunk]) -> str:
    """RAG 分支的用户消息：问题 + 参考资料。"""
    return f"【用户问题】\n{query}\n\n【参考资料】\n{format_context(hits)}"


def content_to_text(content: Any) -> str:
    """把 AIMessage.content 归一成字符串。

    langchain 的 content 类型是 ``str | list[dict | str]``，多模态模型会返回列表。
    阶段 3 的 verifier 解析结构化输出时也要用同一个归一化逻辑。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型响应里抠出一个 JSON 对象；解析不出来返回 ``None``。

    容错三种实测都会出现的形态：裸 JSON、```json 围栏包裹、JSON 前后带一句
    解释性中文。**本函数绝不抛异常** —— 调用方一律把 ``None`` 解读为
    「模型没按约定格式回答」，据此回退到确定性路径（规则路由 / 文本兜底），
    而不是让一次格式抖动把整轮问答打断。

    为什么放在 synthesizer 模块：与 ``content_to_text`` 同理，本模块是本项目
    共享解析工具的所在地。verifier（阶段 3 的核查结论）与 supervisor
    （阶段 5 的路由决策）都从这里取用，避免两份实现各自演化。
    """
    if not text or not text.strip():
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        matched = _JSON_BLOCK_RE.search(candidate)
        if not matched:
            return None
        try:
            loaded = json.loads(matched.group(0))
        except json.JSONDecodeError:
            return None

    return loaded if isinstance(loaded, dict) else None


def extract_citations(answer: str, hits: list[RetrievedChunk]) -> list[Citation]:
    """从答案文本中提取 chunk_id，并与召回结果对齐。

    只保留确实来自本轮召回集合的 id —— 模型编造的引用会被丢弃。
    ``Citation.claim`` 留空：逐句断言归阶段 3 的 verifier 填写。
    """
    index = {hit.chunk_id: hit for hit in hits}
    seen: set[str] = set()
    citations: list[Citation] = []
    for matched in _CITATION_RE.finditer(answer):
        chunk_id = matched.group(0)
        if chunk_id in seen or chunk_id not in index:
            continue
        seen.add(chunk_id)
        hit = index[chunk_id]
        citations.append(
            Citation(
                chunk_id=chunk_id,
                doc_id=hit.doc_id,
                doc_title=hit.doc_title,
                doc_version=hit.version,
            )
        )
    return citations


def _invoke(llm: Any, system_prompt: str, user_prompt: str) -> str:
    """调用 LLM 并返回纯文本答案。"""
    message = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return content_to_text(message.content).strip()


def synthesizer_node(state: GraphState, *, llm: Any | None = None) -> dict:
    """生成节点：按路由分支产出答案与引用。

    三条分支：
    - ``direct``：寒暄/元问题，无参考资料，用 ``SYSTEM_PROMPT_DIRECT`` 约束
      不许编造制度细节。
    - ``retrieve`` 且无命中块：直接拒答，**不调用 LLM**，省一次 token 也
      堵住编造路径。
    - ``retrieve`` 且有命中块：走 RAG 生成 + 引用绑定。

    ``llm`` 允许注入，测试时传替身即可覆盖全部分支而不打网络。
    """
    route = state.get("route") or ""
    query = state.get("query") or ""
    chunks = list(state.get("retrieved_chunks") or [])
    attempts = int(state.get("retrieval_attempts") or 0)

    if route == ROUTE_DIRECT:
        system_prompt, user_prompt = SYSTEM_PROMPT_DIRECT, query
    elif not chunks:
        return {
            "answer": REFUSE_TEXT,
            "citations": [],
            "llm_model": "",
            "events": [
                make_event(NODE_SYNTHESIZER, f"无参考资料（检索 {attempts} 轮），已拒答")
            ],
        }
    else:
        system_prompt = SYSTEM_PROMPT_RAG
        user_prompt = build_user_prompt(query, chunks)

    engine = llm if llm is not None else get_llm()

    try:
        answer = _invoke(engine, system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 - 生成失败不应丢掉已检索到的块
        return {
            "answer": "",
            "citations": [],
            "llm_model": "",
            "error": f"生成失败：{type(exc).__name__}: {exc}",
            "events": [
                make_event(
                    NODE_SYNTHESIZER, f"生成失败：{type(exc).__name__}"
                )
            ],
        }

    citations = extract_citations(answer, chunks)


    return {
        "answer": answer,
        "citations": citations,
        # ChatOpenAI 暴露 model_name 字段；注入替身时退回类名，便于测试断言。
        "llm_model": getattr(engine, "model_name", "") or type(engine).__name__,
        "events": [
            make_event(
                NODE_SYNTHESIZER,
                f"已生成答案（{len(answer)} 字，引用 {len(citations)} 处）",
            )
        ],
    }


__all__ = [
    "REFUSE_TEXT",
    "SYSTEM_PROMPT_RAG",
    "SYSTEM_PROMPT_DIRECT",
    "format_context",
    "build_user_prompt",
    "content_to_text",
    "parse_json_object",
    "extract_citations",
    "synthesizer_node",
]
