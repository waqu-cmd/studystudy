"""MCP Server：外部 Web 搜索工具（可选，stdio 传输）。

暴露工具
--------
``web_search(query, max_results) -> dict``
    调用外部搜索 API，返回 ``{"results": [...], "provider": ..., "error": ...}``。

本 Server 的设计取向：**不虚构能力**
------------------------------------
蓝图把本 Server 标为「可选」。项目当前没有外部检索服务的账号，因此这里
既不写死某个厂商、也不假装能搜 —— 而是做成「按配置启用」：

- 未配置 ``SEARCH_API_BASE`` 时，``web_search`` 返回带 ``error`` 字段的
  结构化失败结果，明确说明缺少哪一项配置，**绝不返回空列表冒充「没搜到」**。
  这两者在调用方看来完全不同：前者是配置问题，后者是内容问题。
- 配置后按 Tavily 风格的 ``POST {base}`` 调用（``api_key`` + ``query`` +
  ``max_results``）。换厂商只需改 ``SEARCH_API_BASE`` 与响应映射。

把可选能力做成「显式未启用」而不是「悄悄缺失」，是工具层该有的诚实：
调用方（Supervisor）能据此决定是否降级，而不是拿到一个无法归因的空结果。

======================================================================
!! 两条硬约束（同 chroma_server.py，改动前请读完）!!
======================================================================

约束 1：禁止 ``from __future__ import annotations``。
    mcp 1.9.2 的 ``Tool.from_function`` 对参数注解直接调 ``issubclass()``，
    注解被延迟成字符串会抛 ``TypeError: issubclass() arg 1 must be a class``。

约束 2：禁止向 stdout 输出。stdio 传输下 stdout 是 JSON-RPC 独占通道。
"""

import httpx
from mcp.server.fastmcp import FastMCP

from app.core.config import settings
from app.core.logging import logger, setup_mcp_logging

mcp = FastMCP("enterprise-kb-search")


def is_configured() -> bool:
    """外部搜索是否已配置。未配置时工具返回显式错误而非静默空结果。"""
    return bool(settings.search_api_base.strip() and _api_key())


def _api_key() -> str:
    return settings.search_api_key.get_secret_value().strip()


@mcp.tool(
    description=(
        "在外部互联网上搜索信息，用于补充企业知识库未覆盖的内容。"
        "返回 {results: [{title, url, content, score}], provider, error}。"
        "注意：本工具默认未启用，未配置时 error 字段会说明缺失的配置项；"
        "此时应改用企业知识库检索，不要据此认为「网上没有相关信息」。"
    )
)
def web_search(query: str, max_results: int = 5) -> dict:
    """外部搜索。

    所有失败路径都返回带 ``error`` 的字典，不抛异常 —— 外部服务的可用性
    不受本服务控制，让它以「降级」而非「中断」的方式呈现给调用方。
    """
    text = (query or "").strip()
    if not text:
        return {
            "results": [],
            "provider": settings.search_provider,
            "error": "query 不能为空",
        }

    if not is_configured():
        logger.warning("web_search 被调用但未配置外部搜索服务")
        return {
            "results": [],
            "provider": settings.search_provider,
            "error": (
                "外部搜索未启用：未配置 SEARCH_API_BASE / SEARCH_API_KEY。"
                "请改用企业知识库检索，或补齐配置后重启本 Server。"
            ),
        }

    limit = max(1, min(int(max_results or 5), 20))
    payload = {"api_key": _api_key(), "query": text, "max_results": limit}
    base = settings.search_api_base.rstrip("/")

    try:
        with httpx.Client(timeout=settings.search_timeout) as client:
            response = client.post(base, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("web_search HTTP 错误 | status={}", exc.response.status_code)
        return {
            "results": [],
            "provider": settings.search_provider,
            "error": f"搜索服务返回 HTTP {exc.response.status_code}",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("web_search 调用失败")
        return {
            "results": [],
            "provider": settings.search_provider,
            "error": f"{type(exc).__name__}: {exc}",
        }

    raw_items = data.get("results") if isinstance(data, dict) else None
    results: list[dict] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "content": str(item.get("content") or item.get("snippet") or ""),
                "score": float(item.get("score") or 0.0),
            }
        )

    logger.info("web_search | query={!r} | results={}", text[:40], len(results))
    return {
        "results": results,
        "provider": settings.search_provider,
        "error": "",
    }


def main() -> None:
    """入口：先配 stderr 日志，再进 stdio 事件循环。"""
    setup_mcp_logging()
    logger.info(
        "search_server 启动 | provider={} | configured={}",
        settings.search_provider or "(未设置)",
        is_configured(),
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
