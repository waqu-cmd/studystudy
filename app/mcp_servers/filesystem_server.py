"""MCP Server：文档库文件访问工具（stdio 传输）。

暴露工具
--------
``list_docs() -> list[dict]``
    扫描 ``DOCS_DIR``，解析每篇文档的 front-matter，返回清单
    （doc_id / doc_title / version / effective_date / expire_date / 文件名 / 字数）。

``read_file(filename) -> dict``
    读取单篇文档全文，同时回传它的时效元数据。

为什么单独做成一个 Server 而不是挂在 chroma_server 上
----------------------------------------------------
两者的能力边界不同：chroma_server 只需读向量库，filesystem_server 只需读文件系统。
拆开后各自的最小权限更小（前者不碰文件系统、后者不连数据库），任一崩溃不影响另一个
—— 这正是阶段 4「可独立部署」的验证点：关掉 chroma_server，``list_docs`` 仍可用。

======================================================================
!! 三条硬约束（同 chroma_server.py，改动前请读完）!!
======================================================================

约束 1：禁止 ``from __future__ import annotations``。
    mcp 1.9.2 的 ``Tool.from_function`` 对参数注解直接调 ``issubclass()``，
    注解被延迟成字符串会抛 ``TypeError: issubclass() arg 1 must be a class``。

约束 2：禁止向 stdout 输出。stdio 传输下 stdout 是 JSON-RPC 独占通道。

约束 3：**工具函数内部不得执行「首次 import」**。
    实测：只含 ``importlib.import_module(...)`` 的探针工具会**永久挂起**，
    而同步 sleep、同步 HTTP、stderr 日志均正常 —— import 需要全局 import lock，
    与事件循环线程互锁。故 ``app.rag.chunker`` 的 import 已在模块顶层完成。

路径安全
--------
``read_file`` 接受客户端传来的任意字符串，必须视为不可信输入。
所有路径先 ``resolve()`` 再校验是否落在 ``DOCS_DIR`` 内，可同时挡住
``../`` 穿越、绝对路径越界与符号链接逃逸（resolve 会展开符号链接）。
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app.core.config import settings
from app.core.logging import logger, setup_mcp_logging

# 顶层 import（约束 3）：绝不能在工具函数内首次 import
from app.rag.chunker import chunk_document, parse_front_matter

mcp = FastMCP("enterprise-kb-filesystem")

# settings.docs_dir 在 config 层已解析为绝对路径，此处直接用
DOCS_DIR: Path = settings.docs_dir

# 只允许读取文本类文档。白名单而非黑名单：MCP 工具面向外部调用方，
# 黑名单总有遗漏（例如 .env、.pem 之类既非文档也非二进制）。
ALLOWED_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".txt"})


def _resolve_within_docs(filename: str) -> Path:
    """把入参解析成 DOCS_DIR 内的真实路径，越界即抛 ValueError。

    校验顺序有意为之：先解析再判断，才能同时覆盖 ``..`` 穿越与符号链接。
    只做字符串 ``startswith`` 判断不够 —— ``docs_evil/`` 会通过 ``docs`` 前缀检查。
    """
    raw = (filename or "").strip()
    if not raw:
        raise ValueError("filename 不能为空")

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = DOCS_DIR / candidate

    resolved = candidate.resolve()
    base = DOCS_DIR.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"路径越界：{raw} 不在文档目录 {base} 内")
    if resolved.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"不支持的文件类型 {resolved.suffix!r}，仅允许 {sorted(ALLOWED_SUFFIXES)}"
        )
    if not resolved.is_file():
        raise ValueError(f"文件不存在：{raw}")
    return resolved


def _describe(path: Path) -> dict:
    """解析单篇文档的 front-matter，返回清单项。

    解析失败不抛异常 —— 一篇 front-matter 写坏的文件不应让整个 ``list_docs``
    失败，而是把错误作为该条目的一部分返回，便于定位。
    """
    item = {
        "filename": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "doc_id": path.stem,
        "doc_title": "",
        "version": "",
        "effective_date": "",
        "expire_date": "",
        "source": "",
        "characters": 0,
        "chunks": 0,
        "error": "",
    }

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        item["error"] = f"读取失败：{type(exc).__name__}: {exc}"
        return item

    try:
        meta, body = parse_front_matter(raw)
        item["doc_title"] = str(meta.get("doc_title") or path.stem)
        item["version"] = str(meta.get("version") or "")
        item["effective_date"] = str(meta.get("effective_date") or "")
        item["expire_date"] = str(meta.get("expire_date") or "9999-12-31")
        item["source"] = str(meta.get("source") or "local")
        item["doc_id"] = str(meta.get("doc_id") or path.stem)
        item["characters"] = len(body)
        item["chunks"] = len(chunk_document(raw, fallback_doc_id=path.stem))
    except Exception as exc:  # noqa: BLE001
        item["error"] = f"解析失败：{type(exc).__name__}: {exc}"
    return item


@mcp.tool(
    description=(
        "列出知识库中的全部文档，返回每篇文档的 doc_id、标题、版本号、"
        "生效日期、失效日期、字节数与切块数量。"
        "用于回答「知识库有哪些文档」「某政策现在生效的是哪一版」这类问题。"
    )
)
def list_docs() -> list[dict]:
    """文档清单。目录不存在时返回空列表（不抛异常）。"""
    if not DOCS_DIR.is_dir():
        logger.warning("文档目录不存在：{}", DOCS_DIR)
        return []

    items: list[dict] = []
    for path in sorted(DOCS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        items.append(_describe(path))

    logger.info("list_docs | dir={} | count={}", DOCS_DIR, len(items))
    return items


@mcp.tool(
    description=(
        "读取知识库中指定文档的全文，同时返回其版本与时效元数据。"
        "filename 为文件名（含扩展名）或相对路径，例如 "
        "'sales_policy_2026q3.md'。仅允许读取文档目录下的 .md/.markdown/.txt 文件。"
    )
)
def read_file(filename: str) -> dict:
    """读取单篇文档。

    输入非法（越界、类型不允许、不存在）时**抛出** ValueError —— 这类错误是
    调用方参数问题，必须让调用方看到明确原因，静默返回空内容反而更难排查。
    MCP 会把异常转成 ``isError=True`` 的结果，客户端据此收到结构化错误提示。
    """
    path = _resolve_within_docs(filename)
    content = path.read_text(encoding="utf-8")
    item = _describe(path)
    item["content"] = content
    item["lines"] = content.count("\n") + 1
    logger.info("read_file | file={} | chars={}", path.name, len(content))
    return item


def main() -> None:
    """入口：先配 stderr 日志，再进 stdio 事件循环。"""
    setup_mcp_logging()
    logger.info("filesystem_server 启动 | docs_dir={}", DOCS_DIR)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
