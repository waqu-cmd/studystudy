"""配置层测试：LangSmith 追踪的彻底移除。

本文件存在的唯一理由：LangSmith 一旦被激活，会把检索到的文档正文与用户原始
问题一并上传到外部服务，属于数据外泄面，且会给每次图执行叠加一次同步网络开销。
这不是「提醒开发者注意」能守住的，必须由测试锁死。

对应实现：``Settings._disable_langsmith_tracing``（进程启动时摘除环境变量）。
"""

from __future__ import annotations

import os

from app.core.config import PROJECT_ROOT, settings

# 两个命名空间都要覆盖：langchain 老式前缀 LANGCHAIN_* 与 langsmith 官方
# 前缀 LANGSMITH_*。只清一边等于没清。
_TRACING_ENV_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
)


def test_langsmith_tracing_env_vars_are_stripped(monkeypatch) -> None:
    """外部环境（或别的项目）留下的追踪变量必须被清掉。

    场景：机器上另一个项目 export 了 LANGCHAIN_TRACING_V2=true，
    langchain-core 直接读 os.environ 而不是读本项目 settings，
    不主动摘除就会静默产生追踪流量。
    """
    for name in _TRACING_ENV_VARS:
        monkeypatch.setenv(name, "true" if name.endswith("TRACING_V2") else "leak")

    settings._disable_langsmith_tracing()

    remaining = [name for name in _TRACING_ENV_VARS if name in os.environ]
    assert remaining == [], f"未被清除的追踪变量：{remaining}"


def test_langsmith_tracing_disabled_is_idempotent() -> None:
    """变量不存在时不能抛异常 —— 正常环境里这些变量本就不存在。"""
    settings._disable_langsmith_tracing()
    settings._disable_langsmith_tracing()


def test_langsmith_settings_fields_are_gone() -> None:
    """三个 LangSmith 配置字段已从 Settings 中删除，防止被重新引入。"""
    for name in ("langchain_tracing_v2", "langchain_api_key", "langchain_project"):
        assert not hasattr(settings, name), f"Settings 仍暴露 LangSmith 字段：{name}"


def test_env_example_has_no_tracing_keys() -> None:
    """`.env.example` 不得包含追踪相关键 —— 否则等于给使用者留一个开关。"""
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    leaked = [
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if "=" in line
        and not line.lstrip().startswith("#")
        and line.split("=", 1)[0].strip().startswith(("LANGCHAIN_", "LANGSMITH_"))
    ]
    assert leaked == [], f".env.example 仍含追踪键：{leaked}"
