"""RRF 融合单元测试（给定两组排序，验证融合顺序正确）。"""

from __future__ import annotations

import pytest

from app.rag.rrf import DEFAULT_K, ranks_of, rrf_fuse, rrf_scores


class TestRrfFuse:
    def test_document_in_both_lists_outranks_single_list(self) -> None:
        """这是 RRF 的核心价值：在多数路里都出现的候选，优先于只在单路排得高的候选。

        路1: A, B, C, D
        路2: A, C
        分数：A=2/61, C=1/63+1/62, B=1/62, D=1/64
        -> C(两路共现) 超过 B(仅单路第 2)
        """
        fused = rrf_fuse([["A", "B", "C", "D"], ["A", "C"]])
        assert [doc_id for doc_id, _ in fused] == ["A", "C", "B", "D"]

    def test_scores_are_descending(self) -> None:
        fused = rrf_fuse([["A", "B", "C"], ["B", "C", "A"]])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_ids_within_one_ranking_counted_once(self) -> None:
        fused = rrf_fuse([["A", "A", "B"]])
        scores = dict(fused)
        assert scores["A"] == pytest.approx(1 / (DEFAULT_K + 1))
        assert scores["B"] == pytest.approx(1 / (DEFAULT_K + 2))

    def test_ties_keep_first_seen_order(self) -> None:
        """分数相同时结果必须稳定，否则回归对比会出现随机波动。"""
        fused = rrf_fuse([["A", "B", "C"], ["B", "A", "C"]])
        # A 与 B 两路名次互换，分数完全相同
        assert dict(fused)["A"] == pytest.approx(dict(fused)["B"])
        assert [doc_id for doc_id, _ in fused[:2]] == ["A", "B"]

    def test_weights_shift_ranking(self) -> None:
        rankings = [["A", "B", "C", "D"], ["A", "C"]]
        fused = rrf_fuse(rankings, weights=[1.0, 0.0])
        assert [doc_id for doc_id, _ in fused] == ["A", "B", "C", "D"]

    def test_weighted_scores_scale_linearly(self) -> None:
        one = dict(rrf_fuse([["A", "B"]], weights=[1.0]))
        half = dict(rrf_fuse([["A", "B"]], weights=[0.5]))
        assert half["A"] == pytest.approx(one["A"] / 2)

    def test_empty_input_returns_empty(self) -> None:
        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []

    def test_raises_on_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse([["A"]], k=0)

    def test_raises_on_mismatched_weights(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse([["A"], ["B"]], weights=[1.0])


class TestRrfScores:
    def test_matches_fuse_output(self) -> None:
        rankings = [["A", "B"], ["B", "C"]]
        assert rrf_scores(rankings) == dict(rrf_fuse(rankings))

    def test_returns_plain_dict(self) -> None:
        """对应 GraphState["rrf_scores"] 的形态。"""
        scores = rrf_scores([["A"]])
        assert isinstance(scores, dict)
        assert set(scores) == {"A"}


class TestRanksOf:
    def test_assigns_one_based_ranks(self) -> None:
        assert ranks_of(["A", "B", "C"]) == {"A": 1, "B": 2, "C": 3}

    def test_keeps_first_occurrence(self) -> None:
        assert ranks_of(["A", "B", "A", "C"]) == {"A": 1, "B": 2, "C": 3}

    def test_skips_blank_ids(self) -> None:
        assert ranks_of(["", "A"]) == {"A": 1}

    def test_empty_input(self) -> None:
        assert ranks_of([]) == {}
