"""全局配置：pydantic-settings 加载 .env，导出单例 settings。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from importlib.metadata import version as _pkg_version
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from importlib.metadata import PackageNotFoundError

# app/core/config.py -> parents[0]=core, parents[1]=app, parents[2]=项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = PROJECT_ROOT / ".env"



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

    # ---------- LLM：推理、路由、核查、生成 ----------
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, gt=0)

    # ---------- Embedding：输出 1024 维 ----------
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)

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

    # ---------- LLM-as-judge，留空则复用 LLM_MODEL ----------
    judge_model: str = ""

    # ---------- 日志 ----------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ---------- LangSmith（可选） ----------
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr = SecretStr("")
    langchain_project: str = "enterprise-kb-agent"

    # ---------------- 校验 ----------------

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("chroma_path", "docs_dir", mode="after")
    @classmethod
    def _absolutize(cls, value: Path) -> Path:
        """相对路径一律相对项目根解析，与进程工作目录解耦。"""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # ---------------- 初始化钩子 ----------------

    def model_post_init(self, __context: Any) -> None:
        self._bridge_tracing_env()

    def _bridge_tracing_env(self) -> None:
        """把 LangSmith 开关同步进 os.environ。

        LangChain / LangSmith 是直接读 os.environ 的，而 pydantic-settings 解析
        .env 文件并不会写入 os.environ。不做这一步桥接，.env 里开了追踪也不生效。
        """
        if not self.langchain_tracing_v2:
            return
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        key = self.langchain_api_key.get_secret_value().strip()
        if key:
            os.environ["LANGCHAIN_API_KEY"] = key

    # ---------------- 就绪校验（供 core/llm.py 调用） ----------------

    def require_llm_ready(self) -> None:
        if not self.llm_api_key.get_secret_value().strip():
            raise RuntimeError("LLM_API_KEY 未配置 —— 请在 %s 中填写" % ENV_FILE)
        if not self.llm_model.strip():
            raise RuntimeError("LLM_MODEL 未配置 —— 请在 %s 中填写" % ENV_FILE)

    def require_embedding_ready(self) -> None:
        if not self.embedding_api_key.get_secret_value().strip():
            raise RuntimeError("EMBEDDING_API_KEY 未配置 —— 请在 %s 中填写" % ENV_FILE)
        if not self.embedding_model.strip():
            raise RuntimeError("EMBEDDING_MODEL 未配置 —— 请在 %s 中填写" % ENV_FILE)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """供 FastAPI Depends 使用；进程内只解析一次。"""
    return Settings()


settings: Settings = get_settings()

__all__ = ["PROJECT_ROOT", "ENV_FILE", "APP_VERSION", "Settings", "get_settings", "settings"]
