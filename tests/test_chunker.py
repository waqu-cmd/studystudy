"""chunker 单元测试（对应蓝图「五、单元测试清单」：按标题切块、metadata 完整、块大小上限）。

本模块是纯函数，因此不需要任何 fixture。
"""

from __future__ import annotations

import pytest

from app.rag.chunker import (
    MAX_CHARS,
    NO_EXPIRE_DATE,
    NO_EXPIRE_ORD,
    NO_HEADING,
    build_doc_metadata,
    chunk_document,
    date_to_ord,
    parse_front_matter,
)

DOC = """---
doc_id: demo_doc
doc_title: 演示文档
version: 2026Q3
effective_date: 2026-07-01
expire_date: 2026-09-30
source: confluence
---

# 演示文档

## 返点政策

新客户签约金额的 3.5%。

## 考核与结算

返点按月结算。
"""


class TestParseFrontMatter:
    def test_splits_meta_and_body(self) -> None:
        meta, body = parse_front_matter(DOC)
        assert meta["doc_id"] == "demo_doc"
        assert meta["version"] == "2026Q3"
        assert body.lstrip().startswith("# 演示文档")

    def test_returns_empty_meta_without_front_matter(self) -> None:
        meta, body = parse_front_matter("# 标题\n\n正文")
        assert meta == {}
        assert body == "# 标题\n\n正文"

    def test_rejects_non_mapping_front_matter(self) -> None:
        with pytest.raises(ValueError):
            parse_front_matter("---\n- a\n- b\n---\n正文")


class TestDateToOrd:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2026-09-30", 20260930),
            ("2026-9-30", 20260930),  # 容错未补零写法
            ("9999-12-31", NO_EXPIRE_ORD),
            ("2026/09/30", 20260930),
            ("", 0),
            ("not-a-date", 0),
        ],
    )
    def test_converts_to_yyyymmdd(self, value: str, expected: int) -> None:
        assert date_to_ord(value) == expected

    def test_ord_preserves_chronological_order(self) -> None:
        """Chroma 的范围过滤依赖整数序 == 日期序这一性质。"""
        dates = ["2026-01-01", "2026-03-31", "2026-07-01", "2026-09-30", "9999-12-31"]
        ords = [date_to_ord(d) for d in dates]
        assert ords == sorted(ords)


class TestBuildDocMetadata:
    def test_normalizes_dates_to_strings(self) -> None:
        meta = build_doc_metadata(
            {
                "doc_id": "d1",
                "version": "v1",
                "effective_date": "2026-01-01",
                "expire_date": "2026-03-31",
            }
        )
        assert meta["effective_date"] == "2026-01-01"
        assert isinstance(meta["effective_date"], str)
        assert meta["expire_ord"] == 20260331

    def test_missing_expire_date_uses_sentinel(self) -> None:
        """长期有效文档必须写哨兵值，否则会被时效过滤误杀。"""
        meta = build_doc_metadata(
            {"doc_id": "d2", "version": "v1", "effective_date": "2025-01-01"}
        )
        assert meta["expire_date"] == NO_EXPIRE_DATE
        assert meta["expire_ord"] == NO_EXPIRE_ORD

    @pytest.mark.parametrize("missing", ["version", "effective_date"])
    def test_requires_mandatory_fields(self, missing: str) -> None:
        payload = {"doc_id": "d3", "version": "v1", "effective_date": "2026-01-01"}
        payload.pop(missing)
        with pytest.raises(ValueError, match=missing):
            build_doc_metadata(payload)

    def test_falls_back_to_file_stem(self) -> None:
        meta = build_doc_metadata(
            {"version": "v1", "effective_date": "2026-01-01"},
            fallback_doc_id="from_filename",
        )
        assert meta["doc_id"] == "from_filename"

    def test_raises_without_any_doc_id(self) -> None:
        with pytest.raises(ValueError):
            build_doc_metadata({"version": "v1", "effective_date": "2026-01-01"})


class TestChunkDocument:
    def test_splits_by_h2(self) -> None:
        chunks = chunk_document(DOC)
        assert [c.metadata["heading"] for c in chunks] == ["返点政策", "考核与结算"]

    def test_chunk_ids_are_deterministic(self) -> None:
        chunks = chunk_document(DOC)
        assert [c.chunk_id for c in chunks] == ["demo_doc_p0", "demo_doc_p1"]
        # 同一输入两次切块结果必须一致，否则 upsert 无法覆盖旧块
        assert [c.chunk_id for c in chunk_document(DOC)] == [
            c.chunk_id for c in chunks
        ]

    def test_metadata_is_complete(self) -> None:
        meta = chunk_document(DOC)[0].metadata
        for key in (
            "doc_id",
            "doc_title",
            "version",
            "effective_date",
            "expire_date",
            "effective_ord",
            "expire_ord",
            "source",
            "chunk_id",
            "heading",
            "chunk_index",
        ):
            assert key in meta, f"metadata 缺少 {key}"

    def test_all_metadata_values_are_chroma_compatible(self) -> None:
        """Chroma 的 metadata 只接受 str / int / float / bool。"""
        for chunk in chunk_document(DOC):
            for key, value in chunk.metadata.items():
                assert isinstance(value, (str, int, float, bool)), (
                    f"{key} 的类型 {type(value)} 无法写入 Chroma"
                )

    def test_h1_is_removed_from_body(self) -> None:
        assert all("演示文档" not in chunk.text for chunk in chunk_document(DOC))

    def test_content_before_first_h2_lands_in_overview(self) -> None:
        raw = (
            "---\ndoc_id: d\nversion: v1\neffective_date: 2026-01-01\n---\n\n"
            "# 标题\n\n这段在第一个 H2 之前。\n\n## 小节\n\n小节内容。\n"
        )
        chunks = chunk_document(raw)
        assert chunks[0].metadata["heading"] == NO_HEADING
        assert "第一个 H2 之前" in chunks[0].text

    def test_raises_on_empty_body(self) -> None:
        raw = (
            "---\ndoc_id: d\nversion: v1\neffective_date: 2026-01-01\n---\n\n"
            "# 只有标题\n"
        )
        with pytest.raises(ValueError):
            chunk_document(raw)

    def test_meta_defaults_fill_missing_fields(self) -> None:
        """PDF 场景：文件本身无 front-matter，由 indexer 合成 meta_defaults 补齐。"""
        raw = "# 无元数据文档\n\n## 小节\n\n正文内容。\n"
        chunks = chunk_document(
            raw,
            fallback_doc_id="pdf_stem",
            meta_defaults={
                "doc_id": "pdf_stem",
                "doc_title": "PDF 标题",
                "version": "v1",
                "effective_date": "2026-01-01",
            },
        )
        meta = chunks[0].metadata
        assert meta["doc_id"] == "pdf_stem"
        assert meta["doc_title"] == "PDF 标题"
        assert meta["expire_date"] == NO_EXPIRE_DATE
        assert meta["source"] == "local"

    def test_front_matter_wins_over_meta_defaults(self) -> None:
        payload = dict(
            fallback_doc_id="ignored",
            meta_defaults={"doc_id": "ignored", "version": "fallback"},
        )
        assert chunk_document(DOC, **payload)[0].metadata["doc_id"] == "demo_doc"
        assert chunk_document(DOC, **payload)[0].metadata["version"] == "2026Q3"


class TestChunkSizeLimit:
    def test_single_oversized_paragraph_is_hard_split(self) -> None:
        para = "段落内容。" * 400  # 2000 字符，且无空行 -> 走硬切分支
        raw = (
            "---\ndoc_id: long\nversion: v1\neffective_date: 2026-01-01\n---\n\n"
            f"# 标题\n\n## 大节\n\n{para}\n"
        )
        chunks = chunk_document(raw)
        assert len(chunks) > 1
        assert all(len(c.text) <= MAX_CHARS for c in chunks)

    def test_multi_paragraph_split_respects_limit(self) -> None:
        body = "\n\n".join(part * 300 for part in ["甲", "乙", "丙"])
        raw = (
            "---\ndoc_id: long2\nversion: v1\neffective_date: 2026-01-01\n---\n\n"
            f"# 标题\n\n## 大节\n\n{body}\n"
        )
        chunks = chunk_document(raw)
        assert all(len(c.text) <= MAX_CHARS for c in chunks)

    def test_long_section_is_split_by_h3_first(self) -> None:
        part = "小节正文。" * 120  # 600 字符，两个 H3 共 1200+，超过上限
        raw = (
            "---\ndoc_id: long3\nversion: v1\neffective_date: 2026-01-01\n---\n\n"
            f"# 标题\n\n## 大节\n\n### 甲\n\n{part}\n\n### 乙\n\n{part}\n"
        )
        headings = [c.metadata["heading"] for c in chunk_document(raw)]
        assert headings == ["甲", "乙"]

    def test_short_document_stays_in_one_chunk(self) -> None:
        raw = (
            "---\ndoc_id: tiny\nversion: v1\neffective_date: 2026-01-01\n---\n\n"
            "# 标题\n\n## 短节\n\n很短。\n"
        )
        assert len(chunk_document(raw)) == 1
