"""RRF（Reciprocal Rank Fusion）融合算法（阶段 1，纯函数模块）。

公式
----
    score(d) = Σ_r  weight_r / (k + rank_r(d))

rank 从 1 开始，k 为平滑常数（默认 60）。

为什么用 RRF 而不是加权求和
---------------------------
稠密检索给出的是余弦距离（越小越相似），稀疏检索给出的是 BM25 分数
（越大越相关，且量纲与语料长度强相关）。两者不可直接相加，做归一化又会
引入对分数分布的假设。RRF 只使用「排名」，天然规避了量纲问题 ——
这也是它成为混合检索默认融合策略的原因。

本模块是纯函数：输入若干有序 id 列表，输出融合排序，不依赖任何外部状态，
因此 test_rrf.py 可直接覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_K = 60


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """融合多路有序召回。

    Args:
        rankings: 每路召回的有序 id 列表，越靠前排名越高（自然序，非逆序）。
            同一路内的重复 id 只计首次出现，不重复累加分数。
        k: 平滑常数。越大则头部与尾部的分数差异越小，通常取 60。
        weights: 每路的权重，默认全部为 1.0。

    Returns:
        [(id, score), ...]，按 score 降序；分数相同时按首次出现顺序稳定排序
        （保证结果可复现，便于测试与回归对比）。

    Raises:
        ValueError: k <= 0，或 weights 长度与 rankings 不一致。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights 长度({len(weights)})与 rankings 路数({len(rankings)})不一致"
        )

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0

    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        rank = 0
        for doc_id in ranking:
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1

    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))


def rrf_scores(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """与 rrf_fuse 相同，但返回 {id: score} 映射。

    对应 GraphState["rrf_scores"] 字段的形态（阶段 2 使用）。
    """
    return dict(rrf_fuse(rankings, k=k, weights=weights))


def ranks_of(ranking: Sequence[str]) -> dict[str, int]:
    """把有序 id 列表转成 {id: rank}，rank 从 1 开始，重复 id 取首次。"""
    ranks: dict[str, int] = {}
    for doc_id in ranking:
        if doc_id and doc_id not in ranks:
            ranks[doc_id] = len(ranks) + 1
    return ranks


__all__ = ["DEFAULT_K", "rrf_fuse", "rrf_scores", "ranks_of"]
