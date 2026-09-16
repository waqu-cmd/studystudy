"""query 请求 / 响应模型（阶段 1）。

阶段 1 的 /query 为同步返回；阶段 6 会在同一请求体上补 SSE 流式端点，
因此这里的数据契约要足够稳定，避免后续改动调用方。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """一条召回结果。对应 GraphState["retrieved_chunks"] 的元素形态。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    version: str
    heading: str = ""
    effective_date: str = ""
    expire_date: str = ""
    text: str
    score: float = Field(description="RRF 融合分数，越大越相关")
    vector_rank: int | None = Field(default=None, description="稠密一路的排名，1 起")
    bm25_rank: int | None = Field(default=None, description="稀疏一路的排名，1 起")
    vector_distance: float | None = Field(
        default=None, description="余弦距离，0 表示完全同向"
    )


class Citation(BaseModel):
    """答案中的一条引用。对应 GraphState["citations"] 的元素形态。

    阶段 1 只做「从答案文本中提取 [chunk_id] 标记」这一步；
    claim 字段留给阶段 3 由 Synthesizer / Verifier 填充。
    """

    claim: str = ""
    chunk_id: str
    doc_id: str = ""
    doc_title: str = ""
    doc_version: str = ""


class QueryRequest(BaseModel):
    """POST /query 请求体。"""

    query: str = Field(min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(
        default=None, description="会话 ID，阶段 5 接入 checkpointer 后用于多轮状态保留"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="覆盖 .env 的 TOP_K，评估时常用"
    )
    include_expired: bool = Field(
        default=False,
        description="是否包含已过期文档。默认 false，即只检索未过期版本",
    )
    with_answer: bool = Field(
        default=True,
        description=(
            "是否调用 LLM 生成答案。置 false 时只返回检索结果，"
            "用于在不消耗 LLM 配额的前提下评估检索质量"
        ),
    )


class QueryResponse(BaseModel):
    """POST /query 响应体。

    阶段 2 追加三个只读字段（route / route_reason / retrieval_attempts）用于
    解释「这一轮为什么这样走」—— 路由可解释性是多 Agent 架构最常被追问的点，
    而这三个值在 state 里本来就有，暴露出来零成本。

    阶段 3 再追加 verdict / unsupported_claims / retry_count：自纠正循环的效果
    必须可观测，否则「提升了忠实度」只是一句无法验证的话。这三项正是阶段 7
    计算忠实度指标的数据来源，也让调用方能自行决定是否采信一个 verdict=fail
    的答案（系统选择诚实输出而非隐瞒）。
    """

    query: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    llm_model: str = Field(default="", description="生成答案所用模型；with_answer=false 时为空")
    include_expired: bool = False
    elapsed_ms: int = 0
    route: str = Field(
        default="", description="supervisor 的路由结果：retrieve | direct"
    )
    route_reason: str = Field(default="", description="路由理由，人类可读")
    sub_queries: list[str] = Field(
        default_factory=list,
        description=(
            "阶段 5：复合问题被拆解出的子问题。长度 ≥2 时图内用 Send 并行检索，"
            "因此该字段非空即说明本轮走的是多路检索；空或单元素表示单路检索"
        ),
    )
    retrieval_attempts: int = Field(
        default=0, description="实际检索轮次：1 表示首检即够，2 表示回退重检过一次"
    )
    verdict: str = Field(
        default="",
        description=(
            "核查结论：pass | fail。空串表示本轮未执行核查"
            "（direct 路由、无参考资料或生成失败）"
        ),
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="核查中未能在召回块找到依据的断言；非空即忠实度风险的直接证据",
    )
    retry_count: int = Field(
        default=0, description="已回退重检次数，上限为 .env 的 MAX_RETRY"
    )
    error: str = Field(
        default="", description="检索或生成阶段的错误；为空表示无错误，此时已有结果仍有效"
    )


__all__ = ["RetrievedChunk", "Citation", "QueryRequest", "QueryResponse"]
