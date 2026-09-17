"""全局配置：pydantic-settings 加载 .env，导出单例 settings。

字段命名契约
------------
本模块的字段名（小写）与 .env 的变量名（大写）严格一一对应。当前 .env 采用
「PROVIDER + API_BASE」命名，因此字段是 llm_api_base / embedding_api_base，
而不是 base_url。改字段名 = 改 .env，两边必须同步。

两组密钥的复用规则
------------------
Embedding 与 LLM 常常共用同一个平台的 Key（例如本项目用阿里云百炼的
OpenAI 兼容端点同时承载两者）。EMBEDDING_API_KEY 留空时自动复用
LLM_API_KEY，避免在两处维护同一个值。

路径类配置
----------
在此解析为绝对路径，与进程工作目录解耦 —— 从任意目录启动 uvicorn 都成立。
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[0]=core, parents[1]=app, parents[2]=项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = PROJECT_ROOT / ".env"

DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _resolve_version() -> str:
    """版本号的唯一来源：pyproject.toml 注册的包元数据。"""
    try:
        return _pkg_version("enterprise-kb-agent")
    except PackageNotFoundError:
        return "0.0.0+local"


APP_VERSION: str = _resolve_version()


class Settings(BaseSettings):
    """项目全部可调参数。改这里 = 改 .env，两边必须同步。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- LLM：Agent 推理、路由、核查、生成 ----------

    llm_provider: str = "dashscope"
    llm_api_key: SecretStr = SecretStr("")
    llm_api_base: str = DASHSCOPE_COMPAT_BASE
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_timeout: float = Field(default=120.0, gt=0)

    # ---------- Embedding：text-embedding-v4，输出 1024 维 ----------
    embedding_provider: str = "dashscope"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_api_base: str = DASHSCOPE_COMPAT_BASE
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=10, gt=0, le=64)

    # ---------- 向量库 ----------
    chroma_path: Path = Path("./data/chroma_db")
    chroma_collection: str = "enterprise_kb"

    # ---------- 检索 ----------
    top_k: int = Field(default=10, gt=0)
    rrf_k: int = Field(default=60, gt=0)

    # ---------- 文档库 ----------
    docs_dir: Path = Path("./data/docs")

    # ---------- Agent 自纠正循环 ----------
    max_retry: int = Field(default=3, ge=0)
    verifier_model: str = "qwen3.6-flash"

    # ---------- MCP 工具层 ----------
    mcp_enabled: bool = True
    mcp_servers: str = "chroma,filesystem"
    mcp_timeout: float = Field(default=30.0, gt=0)
    mcp_start_timeout: float = Field(default=60.0, gt=0)
    mcp_ping_timeout: float = Field(default=5.0, gt=0)

    # ---------- 外部搜索（未配置即显式停用） ----------
    search_provider: str = "tavily"
    search_api_base: str = ""
    search_api_key: SecretStr = SecretStr("")
    search_timeout: float = Field(default=20.0, gt=0)

    # ---------- LLM-as-judge ----------
    judge_model: str = "deepseek-v4-pro"

    # ---------- 多轮会话 ----------
    memory_enabled: bool = True

    # ---------------- 校验 ----------------

    @field_validator("llm_api_base", "embedding_api_base", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """去掉末尾斜杠。openai SDK 会把 base_url 与 /chat/completions 拼接，
        多一个斜杠会得到 //chat/completions 并可能被网关拒绝。"""
        return value.rstrip("/")

    @field_validator("chroma_path", "docs_dir", mode="after")
    @classmethod
    def _absolutize(cls, value: Path) -> Path:
        """相对路径一律相对项目根解析，与进程工作目录解耦。"""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # ---------------- 初始化钩子 ----------------

    def model_post_init(self, __context: Any) -> None:
        self._fallback_embedding_key()
        self._disable_langsmith_tracing()

    def _fallback_embedding_key(self) -> None:
        if not self.embedding_api_key.get_secret_value().strip():
            self.embedding_api_key = self.llm_api_key

    def _disable_langsmith_tracing(self) -> None:
        for name in (
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_TRACING",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_ENDPOINT",
            "LANGCHAIN_PROJECT",
            "LANGSMITH_TRACING",
            "LANGSMITH_API_KEY",
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_PROJECT",
        ):
            os.environ.pop(name, None)

    # ---------------- 只读派生属性 ----------------

    @property
    def judge_model_name(self) -> str:
        """LLM-as-judge 实际使用的模型名，留空则复用主模型。"""
        return self.judge_model.strip() or self.llm_model

    @property
    def verifier_model_name(self) -> str:
        """阶段 3 核查器实际使用的模型名，留空则复用主模型。"""
        return self.verifier_model.strip() or self.llm_model

    # ---------------- 就绪校验（供 core/llm.py 调用） ----------------

    def require_llm_ready(self) -> None:
        if not self.llm_api_key.get_secret_value().strip():
            raise RuntimeError(f"LLM_API_KEY 未配置 —— 请在 {ENV_FILE} 中填写")
        if not self.llm_model.strip():
            raise RuntimeError(f"LLM_MODEL 未配置 —— 请在 {ENV_FILE} 中填写")

    def require_embedding_ready(self) -> None:
        if not self.embedding_api_key.get_secret_value().strip():
            raise RuntimeError(
                "EMBEDDING_API_KEY 与 LLM_API_KEY 均为空 —— 请在 "
                f"{ENV_FILE} 中至少填写 LLM_API_KEY"
            )
        if not self.embedding_model.strip():
            raise RuntimeError(f"EMBEDDING_MODEL 未配置 —— 请在 {ENV_FILE} 中填写")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()

__all__ = [
    "PROJECT_ROOT",
    "ENV_FILE",
    "APP_VERSION",
    "DASHSCOPE_COMPAT_BASE",
    "Settings",
    "get_settings",
    "settings",
]
