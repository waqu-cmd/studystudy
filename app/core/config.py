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
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[0]=core, parents[1]=app, parents[2]=项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = PROJECT_ROOT / ".env"

# 阿里云百炼（DashScope）的 OpenAI 兼容端点。
# LLM 与 Embedding 共用同一个 base_url，仅 model 名不同。
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
    # 注意：deepseek-chat / deepseek-reasoner 已于 2026-07-24 弃用。
    llm_provider: str = "dashscope"
    llm_api_key: SecretStr = SecretStr("")
    llm_api_base: str = DASHSCOPE_COMPAT_BASE
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_timeout: float = Field(default=120.0, gt=0)

    # ---------- Embedding：text-embedding-v4，输出 1024 维 ----------
    embedding_provider: str = "dashscope"
    # 留空则自动复用 llm_api_key，见 _fallback_embedding_key
    embedding_api_key: SecretStr = SecretStr("")
    embedding_api_base: str = DASHSCOPE_COMPAT_BASE
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = Field(default=1024, gt=0)
    # text-embedding-v4 单次请求最多 10 条文本，超过会报 400，必须分批
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
    # 阶段 3：核查器专用模型。留空则复用 llm_model。
    # 核查是「判断题 + 结构化输出」，比生成任务简单，可换成更便宜的小模型。
    verifier_model: str = ""

    # ---------- MCP 工具层（阶段 4） ----------
    # 总开关。置 false 时 retriever 节点退回阶段 3 的进程内直连检索 ——
    # 用于 A/B 对比「工具层解耦」的收益与开销，检索算法本身不变。
    mcp_enabled: bool = True
    # 启用的 Server 列表（逗号分隔）。search 需外部检索凭据，默认不启动，
    # 避免白付一个子进程的启动成本。
    mcp_servers: str = "chroma,filesystem"
    # 单次工具调用超时。检索内含一次 embedding 网络请求，不宜过短。
    mcp_timeout: float = Field(default=30.0, gt=0)
    # 启动时与全部 Server 握手的总超时。真实 Server 需 import chromadb，
    # 冷启动明显慢于探针脚本，故给足余量。
    mcp_start_timeout: float = Field(default=60.0, gt=0)
    # 健康检查单次 ping 超时。
    mcp_ping_timeout: float = Field(default=5.0, gt=0)

    # ---------- 外部搜索（阶段 4 可选项，未配置即显式停用） ----------
    search_provider: str = "tavily"
    search_api_base: str = ""
    search_api_key: SecretStr = SecretStr("")
    search_timeout: float = Field(default=20.0, gt=0)

    # ---------- 阶段 7：LLM-as-judge，留空则复用 LLM_MODEL ----------
    judge_model: str = ""

    # ---------- 日志 ----------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ---------- 多轮会话（阶段 5） ----------
    # 置 true 时编译出的图挂 MemorySaver，session_id 作为 thread_id 保留本轮状态。
    # 置 false 退回无状态（每轮独立），供回归对照与「完全无状态」部署使用。
    memory_enabled: bool = True

    # ---------------- 校验 ----------------

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

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
        """EMBEDDING_API_KEY 未配置时复用 LLM_API_KEY。"""
        if not self.embedding_api_key.get_secret_value().strip():
            self.embedding_api_key = self.llm_api_key

    def _disable_langsmith_tracing(self) -> None:
        """本项目不使用 LangSmith，进程启动时强制摘除追踪相关环境变量。

        为什么必须显式摘除，而不是「不设置就没事」：
        langchain-core 直接读 os.environ，而不是读本模块的 settings 对象。
        若机器上因其他项目留下了 LANGCHAIN_TRACING_V2=true / LANGSMITH_TRACING=true，
        LangChain 会把每次图的执行（含检索到的文档正文与用户原始问题）
        上传到 LangSmith —— 既是数据外泄面，也会给每次调用叠加一次同步网络开销。
        这里主动摘除，保证本进程在任何环境下都不会产生追踪流量。
        """
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
    """供 FastAPI Depends 使用；进程内只解析一次。"""
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
