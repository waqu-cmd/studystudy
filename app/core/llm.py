"""LLM 与 Embedding 客户端单例（阶段 1）。

为什么 Embedding 自己实现，而不用 langchain 的 OpenAIEmbeddings
------------------------------------------------------------------
阿里云百炼 text-embedding-v4 有两个硬约束，langchain 的默认行为两条都会踩：

1. 单次请求最多 10 条文本，langchain 默认按 1000 条一批发 -> 接口报 400
2. 只接受原始字符串，langchain 默认先用 tiktoken 按 token 截断再发 -> 接口报 400
   （须显式 check_embedding_ctx_length=False）

虽然这两个参数都可配，但直接用 openai SDK 更可控：分批、重试、维度自检、
超长预检集中在一处，不引入 langchain 的隐式行为。

LLM 侧使用 ChatOpenAI：阶段 2 起 LangGraph 节点依赖 langchain 的 message 协议与
tool calling，自己包一层没有收益。

客户端生命周期
--------------
均为进程内懒加载单例。密钥校验推迟到首次取用（require_*_ready），
因此只要不真正调用 LLM/Embedding，服务仍可启动、/health 仍可访问。
"""

from __future__ import annotations

import time
from functools import lru_cache

import openai
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import logger

# Embedding 请求参数
MAX_EMBED_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.5

# 单条文本字符上限。模型上限 8192 token，中文约 1 字 ≈ 0.6~1 token，
# 这里取 6000 字符作保守护栏；本项目 chunk 上限 800 字符，正常不会触发。
MAX_EMBED_CHARS = 6000

# 可重试错误：限流、网络、服务端 5xx。400 属配置错误，重试无意义。
_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class EmbeddingError(RuntimeError):
    """Embedding 调用失败（含维度不符、空文本、批量超限等）。"""


class ChatModelError(RuntimeError):
    """LLM 调用失败。"""


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


class DashScopeEmbeddings:
    """OpenAI 兼容的 Embedding 客户端，按 batch_size 自动分批。

    方法名与 langchain Embeddings 接口保持一致（embed_documents / embed_query），
    便于将来需要时替换实现而不改动调用方。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError("Embedding 密钥为空，请检查 .env 中的 LLM_API_KEY")
        self._model = model
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout
        )

    # ---------------- 公开接口 ----------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，返回顺序与入参严格一致。"""
        if not texts:
            return []

        prepared: list[str] = []
        for index, raw in enumerate(texts):
            text = (raw or "").strip()
            if not text:
                raise EmbeddingError(f"第 {index} 条文本为空，Embedding 无法处理空串")
            if len(text) > MAX_EMBED_CHARS:
                logger.warning(
                    "文本超长已截断 | index={} | chars={} -> {}",
                    index,
                    len(text),
                    MAX_EMBED_CHARS,
                )
                text = text[:MAX_EMBED_CHARS]
            prepared.append(text)

        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self._batch_size):
            batch = prepared[start:start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    # ---------------- 内部实现 ----------------

    def _embed_batch(self, batch: list[str], *, attempt: int = 0) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dim,
                encoding_format="float",
            )
        except _RETRYABLE as exc:
            if attempt + 1 >= MAX_EMBED_ATTEMPTS:
                raise EmbeddingError(
                    f"Embedding 调用重试 {MAX_EMBED_ATTEMPTS} 次仍失败：{exc}"
                ) from exc
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Embedding 调用失败，{:.1f}s 后重试 | attempt={} | {}",
                delay,
                attempt + 1,
                type(exc).__name__,
            )
            time.sleep(delay)
            return self._embed_batch(batch, attempt=attempt + 1)
        except openai.BadRequestError as exc:
            # 400 多为配置问题：批量超限 / 模型名错误 / dimensions 不被支持
            raise EmbeddingError(
                f"Embedding 请求被拒绝（400）：{exc}。请检查 "
                f"EMBEDDING_MODEL={self._model}、EMBEDDING_DIM={self._dim}、"
                f"EMBEDDING_BATCH_SIZE={self._batch_size}"
            ) from exc

        # 按 index 排序，保证与入参顺序一致（网关不保证返回顺序）
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingError(
                f"Embedding 返回条数不符：请求 {len(batch)} 条，返回 {len(items)} 条"
            )

        vectors: list[list[float]] = []
        for item in items:
            vector = list(item.embedding)
            if len(vector) != self._dim:
                raise EmbeddingError(
                    f"Embedding 维度不符：期望 {self._dim}，实际 {len(vector)}。"
                    "若刚更换过 EMBEDDING_MODEL，必须先删除 data/chroma_db/ 重建索引，"
                    "因为 Chroma 的维度在首次写入时即已锁定。"
                )
            vectors.append(vector)
        return vectors


# --------------------------------------------------------------------------- #
# 单例与便捷函数
# --------------------------------------------------------------------------- #

_embedder: DashScopeEmbeddings | None = None


def get_embedder() -> DashScopeEmbeddings:
    """Embedding 单例（懒加载，首次调用时校验密钥）。"""
    global _embedder
    if _embedder is None:
        settings.require_embedding_ready()
        _embedder = DashScopeEmbeddings(
            api_key=settings.embedding_api_key.get_secret_value().strip(),
            base_url=settings.embedding_api_base,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
        )
        logger.info(
            "Embedding 客户端就绪 | provider={} | model={} | dim={} | batch={}",
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_dim,
            settings.embedding_batch_size,
        )
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """便捷函数：批量向量化。"""
    return get_embedder().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """便捷函数：单条查询向量化。"""
    return get_embedder().embed_query(text)


@lru_cache(maxsize=8)
def _build_llm(
    model: str, temperature: float, max_tokens: int, timeout: float
) -> ChatOpenAI:
    settings.require_llm_ready()
    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key.get_secret_value().strip(),
        base_url=settings.llm_api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
    )


def get_llm(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """LLM 实例（按参数缓存）。

    阶段 3 的 verifier 会用更小的模型与更低的 max_tokens 控成本，
    因此保留三个可覆盖参数；不传则全部取 .env 配置。
    """
    return _build_llm(
        model or settings.llm_model,
        settings.llm_temperature if temperature is None else temperature,
        max_tokens or settings.llm_max_tokens,
        settings.llm_timeout,
    )


def reset_clients() -> None:
    """清空客户端缓存。供测试与「改了 .env 后热重载」使用。"""
    global _embedder
    _embedder = None
    _build_llm.cache_clear()


__all__ = [
    "EmbeddingError",
    "ChatModelError",
    "DashScopeEmbeddings",
    "get_embedder",
    "embed_texts",
    "embed_query",
    "get_llm",
    "reset_clients",
    "MAX_EMBED_CHARS",
]
