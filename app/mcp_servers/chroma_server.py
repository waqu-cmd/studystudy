"""MCP Server：向量检索工具（stdio 传输）。

暴露工具
--------
``search_documents(query, top_k, include_expired) -> list[dict]``
    向量 + BM25 混合检索（RRF 融合），返回 ``RetrievedChunk`` 的字典列表。

``collection_stats() -> dict``
    索引统计（总块数 / 未过期块数 / 集合名 / 向量维度）。

本 Server 不实现任何检索算法，只把 ``app.rag.retriever.HybridRetriever``
原样暴露成 MCP 工具 —— 剥离的是**进程边界**，不是逻辑。算法仍只有一份实现，
阶段 1 的评估结论（Hit@3 / MRR）对本 Server 依然成立。

======================================================================
!! 四条硬约束，违反任意一条都会导致极难排查的故障 !!
======================================================================

约束 1：本文件**不能**出现 ``from __future__ import annotations``。
    实测（mcp 1.9.2）：
        mcp/server/fastmcp/tools/base.py:64
            if issubclass(param.annotation, Context):
        TypeError: issubclass() arg 1 must be a class
    future import 会把注解延迟为字符串，而 ``Tool.from_function`` 未做
    ``get_type_hints`` 解析就直接 ``issubclass``。该报错完全不指向真因，
    表现为「子进程启动即退出 + 客户端只看到 Connection closed」。

约束 2：本文件**不能**向 stdout 写任何非协议内容。
    stdio 传输下 stdout 被 JSON-RPC 独占，一行 ``print`` 就会破坏握手。
    本 Server 因此不产生任何 stdout 输出。

约束 3（最隐蔽的一条）：**工具函数内部不得执行「首次 import」**。
    实测：一个只在工具函数里 ``importlib.import_module("app.rag.retriever")`` 的
    探针 Server，该调用**永久挂起**（180s 超时也不返回），而同类探针中
    纯返回、同步 ``time.sleep(2)``、同步 HTTP 请求全部正常。
    FastMCP 把同步工具交给工作线程执行，而 import 需要获取全局 import lock ——
    与事件循环线程的 import 需求互锁，形成死锁。表现极具误导性：**首次调用挂起、
    第二次起正常**（模块已被加载），极易被误判为「首次连接慢」。
    因此本模块把 ``app.rag.*`` 全部 import 提到**模块顶层**，并用 ``warmup()``
    在 ``mcp.run()`` 之前完成初始化 —— 那时还是单线程，不存在锁竞争。

约束 4（性能）：``_retriever`` 必须是模块级单例。
    Server 进程长驻（见 mcp_client.py 的连接模型），单例让 Chroma 连接与
    BM25 语料缓存跨多次调用复用；每调用重建会让每次检索多付数百毫秒。
"""

from mcp.server.fastmcp import FastMCP

# 顶层 import（约束 3）：绝不能在工具函数内首次 import
from app.core.config import settings
from app.rag.indexer import get_collection
from app.rag.retriever import HybridRetriever, load_corpus

mcp = FastMCP("enterprise-kb-chroma")

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """进程内检索器单例。构造本身很轻，重活在 ``warmup()`` 里做。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def warmup() -> None:
    """在 ``mcp.run()`` 之前的单线程阶段完成重初始化。

    做三件事：建立 Chroma 连接、加载全量语料（供 BM25 用）、构造检索器。
    全部在单线程阶段完成，既避开约束 3 的 import 锁竞争，也让第一次真实调用
    不必承担冷启动成本（实测冷启动约 3 秒，会直接被调用方误判为超时）。

    失败不阻止启动：Chroma 目录损坏时，Server 仍应能起来让 ``/health``
    反映出问题，而不是连握手都完成不了。真正的错误会在首次调用时再次暴露。
    """
    try:
        corpus = load_corpus()
        _get_retriever()
    except Exception:  # noqa: BLE001 - 预热失败不应阻止 Server 启动
        pass


@mcp.tool(
    description=(
        "在企业知识库中检索与问题最相关的文档片段。"
        "采用向量 + BM25 混合检索与 RRF 融合排序，默认过滤已过期文档。"
        "返回字段：chunk_id / doc_id / doc_title / version / heading / "
        "effective_date / expire_date / text / score / vector_rank / "
        "bm25_rank / vector_distance。"
    )
)
def search_documents(
    query: str, top_k: int = 0, include_expired: bool = False
) -> list[dict]:
    """混合检索。

    Args:
        query: 检索词。空串直接返回空列表，不抛异常。
        top_k: 返回条数上限。0（默认）表示沿用服务端配置 ``TOP_K``。
        include_expired: 是否纳入已过期文档。默认 False —— 这是阶段 7
            「版本准确率」指标成立的前提，仅在对照实验中置 True。

    Returns:
        命中片段字典列表，按 RRF 分数降序。检索失败时**不抛异常**而是返回
        带 ``error`` 字段的诊断条目 —— MCP 工具抛异常会让客户端只收到
        ``ToolException`` 文本，丢失结构化上下文，也不利于优雅降级。
    """
    text = (query or "").strip()
    if not text:
        return []

    limit = int(top_k) if top_k and int(top_k) > 0 else settings.top_k
    try:
        hits = _get_retriever().search(
            text, top_k=limit, include_expired=bool(include_expired)
        )
    except Exception as exc:  # noqa: BLE001 - 工具层不抛异常，转成诊断条目
        return [{"error": f"{type(exc).__name__}: {exc}", "chunk_id": ""}]

    return [hit.model_dump() for hit in hits]


@mcp.tool(
    description=(
        "返回知识库的索引统计：已索引块数、未过期块数、集合名与向量维度。"
        "用于健康检查与索引状态确认。"
    )
)
def collection_stats() -> dict:
    """索引统计。读取失败返回带 ``error`` 字段的字典而非抛异常。"""
    try:
        collection = get_collection()
        total = int(collection.count())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "total": -1}

    try:
        active = len(load_corpus().active_positions)
    except Exception:  # noqa: BLE001 - 未过期计数失败时用 -1 表示不可用
        active = -1

    return {
        "collection": settings.chroma_collection,
        "total": total,
        "active": active,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
    }


def main() -> None:
    """入口：预热 → 进 stdio 事件循环。

    ``warmup()`` 必须在 ``mcp.run()`` 之前 —— 那时还是单线程，
    不存在约束 3 描述的 import 锁竞争。
    """
    warmup()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
