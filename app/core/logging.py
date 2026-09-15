"""loguru 日志配置：接管标准库 logging，统一控制台与文件输出。

重要约束：
MCP Server 通过 stdio 与客户端通信，stdout 就是 JSON-RPC 通道。
任何写进 stdout 的日志都会破坏协议握手，客户端会报 JSON 解析错误或直接卡死。
因此 MCP Server 侧必须使用 setup_mcp_logging()（只写 stderr），绝不能用 setup_logging()。
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

from loguru import logger

from app.core.config import PROJECT_ROOT, settings

LOG_DIR: Path = PROJECT_ROOT / "logs"

CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)

# 需要从标准库转交到 loguru 的第三方 logger
_INTERCEPTED_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langchain_core",
    "langgraph",
    "chromadb",
    "mcp",
)

_configured = False


class InterceptHandler(logging.Handler):
    """把标准库 logging 的记录转发给 loguru，保证格式与 sink 完全统一。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _wire_stdlib_logging() -> None:
    """让 uvicorn / httpx / langchain 等第三方库的日志也走 loguru。"""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in _INTERCEPTED_LOGGERS:
        std = logging.getLogger(name)
        std.handlers = [InterceptHandler()]
        std.propagate = False


def setup_logging(*, log_file: bool = True, force: bool = False) -> None:
    """初始化日志（幂等）。

    Args:
        log_file: 是否额外落盘到 logs/app.log。
        force: 置 True 可强制重新初始化（多 worker 或测试场景）。
    """
    global _configured
    if _configured and not force:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=CONSOLE_FORMAT,
        colorize=sys.stderr.isatty(),
        backtrace=True,
        diagnose=False,  # 置 True 会在 traceback 中打印局部变量，有泄露密钥的风险
    )

    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            LOG_DIR / "app.log",
            level=settings.log_level,
            format=FILE_FORMAT,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            enqueue=True,  # 异步写盘，避免日志 I/O 阻塞事件循环
            backtrace=True,
            diagnose=False,
        )

    _wire_stdlib_logging()
    _configured = True


def setup_mcp_logging() -> None:
    """MCP Server 专用：只写 stderr，绝不触碰 stdout。

    stdio 传输下 stdout 被 JSON-RPC 独占，任何日志输出都会破坏协议。
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=CONSOLE_FORMAT,
        colorize=False,  # stderr 会被客户端捕获，不做 ANSI 着色
        backtrace=True,
        diagnose=False,
    )
    _wire_stdlib_logging()


__all__ = ["logger", "setup_logging", "setup_mcp_logging", "LOG_DIR"]

