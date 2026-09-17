"""评估回归测试（阶段 7）：防止语料/题目/指标三者之间悄悄失配。

为什么需要一个专门的回归测试，而不是只看 ``eval/report.json``
------------------------------------------------------------
本项目的评估结果**依赖三件会独立变化的东西**：语料内容、日期、题目期望。
它们任何一项漂移都不会让代码报错，只会让报告里的数字静默改变：

1. 有人编辑了 ``data/docs/`` 里的政策文本 → 切块边界变了 → ``chunk_id``
   位移（``sales_policy_2026q3_p1`` 变成另一小节）→ 标答指向错误的块，
   指标照常算出来，但已经不对了。
2. 日期跨过某个有效期 → 文档从「有效」变「过期」→ 标答根本检索不到。
3. 有人手改 golden set 却写错 id 或漏填 ``should_refuse``。

本文件把这三类都变成**离线可执行、失败信息可操作**的断言。它不跑图、不联网、
不访问 Chroma（语料直接从 ``data/docs/*.md`` 现场切块），因此可以放进 CI 的
每次提交。

时间窗为什么要**双向**守护
--------------------------
单向不够。既要确认「正常题引用的文档还没过期」（否则题目失去意义），
也要确认「陷阱题依赖的过期文档确实过期了」（否则『应当拒答』这个前提不成立，
拒答率会莫名其妙地掉下来）。两个方向都会随时间自然失效，必须显式失败。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.core.config import settings
from app.rag.chunker import NO_EXPIRE_ORD, chunk_document
from eval.eval import GOLDEN_SET_PATH

REPORT_PATH = Path(__file__).resolve().parents[1] / "eval" / "report.json"

EXPECTED_NORMAL = 30
EXPECTED_TRAP = 5

#: 指标下限。低于此值视为退化，测试失败。
#:
#: 取值原则：比实测值低一档留出正常波动空间，但要高到能抓住真实退化。
#: 全部为「语义上必须成立」的下限，不是「当前跑出来的数字」——
#: 否则每次跑完把数字抄进来，测试就退化成了自我确认。
#:
#: ``refusal_rate``（严格口径：文本拒答 **且** 零引用）的下限刻意设得低：
#: 它是**保守下界**，按设计就会因「合法拒答里带了一句元证据引用」而扣分
#: （实测 t2 即如此：引用 Q3 的块来支撑『资料中仅含 Q3』，judge 判其正确）。
#: 把关卡设在该指标的实测值上，会让一次正常的措辞变化被误报成退化。
#: 真正的语义闸门是 ``judge_refusal_rate``。
FLOORS: dict[str, float] = {
    "hit_at_3": 0.90,
    "mrr": 0.90,
    "version_accuracy": 0.90,
    "refusal_rate": 0.60,
    "judge_refusal_rate": 0.90,
    "faithfulness": 0.85,
    "citation_accuracy": 0.80,
}


# --------------------------------------------------------------------------- #
# 语料现场切块（离线，不碰 Chroma）
# --------------------------------------------------------------------------- #


def _today_ord() -> int:
    return int(date.today().strftime("%Y%m%d"))


def _live_chunk_index() -> dict[str, dict[str, Any]]:
    """把 ``data/docs/`` 现场切块，得到 ``{chunk_id: metadata}``。

    刻意**不**读 Chroma：索引是派生产物，可能没建、可能过期；
    而 golden set 的期望是针对**语料文件**写的，比对对象应当是文件本身。
    """
    index: dict[str, dict[str, Any]] = {}
    docs_dir = Path(settings.docs_dir)
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        for chunk in chunk_document(raw, fallback_doc_id=path.stem):
            index[chunk.chunk_id] = dict(chunk.metadata)
    return index


def _load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return _load_golden()


@pytest.fixture(scope="module")
def chunk_index() -> dict[str, dict[str, Any]]:
    return _live_chunk_index()


# --------------------------------------------------------------------------- #
# 一、golden set 结构
# --------------------------------------------------------------------------- #


def test_golden_set_case_counts(golden: dict[str, Any]) -> None:
    cases = golden["cases"]
    normal = [c for c in cases if c["category"] == "normal"]
    trap = [c for c in cases if c["category"] == "trap"]
    assert len(normal) == EXPECTED_NORMAL, f"正常题应为 {EXPECTED_NORMAL} 条"
    assert len(trap) == EXPECTED_TRAP, f"陷阱题应为 {EXPECTED_TRAP} 条"


def test_golden_set_ids_are_unique(golden: dict[str, Any]) -> None:
    ids = [c["id"] for c in golden["cases"]]
    duplicates = {item for item in ids if ids.count(item) > 1}
    assert not duplicates, f"id 重复：{sorted(duplicates)}"


def test_normal_cases_have_targets(golden: dict[str, Any]) -> None:
    """正常题必须有标答与应有版本，且不得标成应拒答。"""
    for case in golden["cases"]:
        if case["category"] != "normal":
            continue
        assert case["should_refuse"] is False, f"{case['id']} 不应标 should_refuse"
        assert case["relevant_chunk_ids"], f"{case['id']} 缺 relevant_chunk_ids"
        assert case["required_doc_versions"], f"{case['id']} 缺 required_doc_versions"
        assert case["trap_type"] is None, f"{case['id']} 正常题不应有 trap_type"


def test_trap_cases_are_refusals_without_targets(golden: dict[str, Any]) -> None:
    """陷阱题必须拒答、无标答、且有明确类型。

    陷阱题若带了标答，它就不再是「资料不足」测试，而会混进检索指标的分母 ——
    正是 metrics 层刻意避免的口径错误。
    """
    for case in golden["cases"]:
        if case["category"] != "trap":
            continue
        assert case["should_refuse"] is True, f"{case['id']} 陷阱题必须 should_refuse"
        assert case["relevant_chunk_ids"] == [], f"{case['id']} 陷阱题不应有标答"
        assert case["required_doc_versions"] == [], f"{case['id']} 陷阱题不应有应有版本"
        assert case["trap_type"], f"{case['id']} 缺 trap_type"


# --------------------------------------------------------------------------- #
# 二、题目 ↔ 语料一致性
# --------------------------------------------------------------------------- #


def test_every_relevant_chunk_id_exists_in_corpus(
    golden: dict[str, Any], chunk_index: dict[str, dict[str, Any]]
) -> None:
    """标答块 id 必须真实存在于语料切块结果中。

    这条抓的是「改了文档正文导致 chunk_id 位移」—— 那是最隐蔽的一类失效：
    指标照常算出来，只是全都算错了。失败信息直接给出可用 id，便于修正。
    """
    known = set(chunk_index)
    stale_refs: dict[str, list[str]] = {}
    for case in golden["cases"]:
        missing = [cid for cid in case["relevant_chunk_ids"] if cid not in known]
        if missing:
            stale_refs[case["id"]] = missing
    assert not stale_refs, (
        f"标答块在语料中不存在：{stale_refs}。"
        f" 语料现有 id 共 {len(known)} 个，可能是文档被编辑后切块边界变了。"
    )


def test_golden_set_corpus_block_matches_live_corpus(
    golden: dict[str, Any], chunk_index: dict[str, dict[str, Any]]
) -> None:
    """golden set 记录的语料规模必须与现场一致。

    语料扩容本身是允许的，但必须同步更新 golden set 的 ``corpus`` 说明块 ——
    否则文件在描述一个已经不存在的世界。
    """
    declared = golden["corpus"]
    assert declared["chunks_total"] == len(chunk_index), (
        f"golden set 声明 {declared['chunks_total']} 块，实际切出 {len(chunk_index)} 块。"
        " 若是有意扩容语料，请同步更新 golden_set.json 的 corpus 说明块。"
    )
    live_docs = {meta["doc_id"] for meta in chunk_index.values()}
    assert declared["documents"] == len(live_docs)


def test_required_versions_match_target_chunks(
    golden: dict[str, Any], chunk_index: dict[str, dict[str, Any]]
) -> None:
    """``required_doc_versions`` 必须与标答块自身的 version 一致。

    这两处填的是同一件事，写不一致时指标会自相矛盾：
    version_accuracy 用 required 判对错，而 Hit@3 用标答块判对错。
    """
    for case in golden["cases"]:
        if not case["relevant_chunk_ids"]:
            continue
        versions = {
            chunk_index[cid]["version"]
            for cid in case["relevant_chunk_ids"]
            if cid in chunk_index
        }
        assert versions == set(case["required_doc_versions"]), (
            f"{case['id']}：标答块版本 {sorted(versions)} 与 "
            f"required_doc_versions {case['required_doc_versions']} 不一致"
        )


# --------------------------------------------------------------------------- #
# 三、时间窗双向守护
# --------------------------------------------------------------------------- #


def test_answerable_cases_are_still_retrievable(
    golden: dict[str, Any], chunk_index: dict[str, dict[str, Any]]
) -> None:
    """正常题的标答必须仍处于有效期内。

    过期后检索会按设计过滤掉这些块，题目从「有标答」变成「答不出来」，
    指标会整体下滑但**不是**质量退化 —— 必须在语义层提前拦下。
    """
    today = _today_ord()
    expired: dict[str, str] = {}
    for case in golden["cases"]:
        for cid in case["relevant_chunk_ids"]:
            meta = chunk_index.get(cid)
            if meta is None:
                continue
            if int(meta.get("expire_ord", NO_EXPIRE_ORD)) < today:
                expired[case["id"]] = f"{cid} 已于 {meta.get('expire_date')} 过期"
    assert not expired, (
        f"golden set 前提已失效：{expired}。"
        " 处理方式见 golden_set.json 的 validity_window.action_on_expiry："
        "新增一份更新季度的政策文档，并把相关题目的 required_doc_versions 改为新版本。"
    )


def test_trap_premise_still_holds(
    golden: dict[str, Any], chunk_index: dict[str, dict[str, Any]]
) -> None:
    """过期版本陷阱所依赖的文档必须确实已过期。

    这是另一个方向：若这些文档又变成有效（例如有人延长了有效期、或测试机上
    时间被回拨到 2026 年第一季度之前），「应当拒答」的前提就不成立了，
    拒答率会无解释地下滑。
    """
    today = _today_ord()
    premised = {"expired_version", "comparison_expired"}
    broken: dict[str, list[str]] = {}
    for case in golden["cases"]:
        if case.get("trap_type") not in premised:
            continue
        for version in case.get("forbidden_versions") or []:
            doc_chunks = [
                meta
                for meta in chunk_index.values()
                if meta.get("version") == version
            ]
            if not doc_chunks:
                broken.setdefault(case["id"], []).append(f"{version} 在语料中不存在")
                continue
            still_active = int(doc_chunks[0].get("expire_ord", NO_EXPIRE_ORD)) >= today
            if still_active:
                broken.setdefault(case["id"], []).append(
                    f"{version} 仍未过期（expire={doc_chunks[0].get('expire_date')}）"
                )
    assert not broken, (
        f"陷阱题前提不成立：{broken}。"
        " 这些题目依赖『该版本已失效』，前提变了就必须换成新的过期版本或改题。"
    )


# --------------------------------------------------------------------------- #
# 四、报告指标下限
# --------------------------------------------------------------------------- #


def _load_report() -> dict[str, Any] | None:
    if not REPORT_PATH.exists():
        return None
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_metrics_meet_floors() -> None:
    """对最近一次全量报告施加指标下限。

    ``report.json`` 是**采集产物**（需要真实 API 才能生成），因此不在每次提交时
    重新生成。这里的作用是：只要仓库里带着这份报告，它就必须通过下限校验 ——
    防止有人把一份退化的报告提交进来当作最新结果。
    未生成报告的新克隆上会跳过，而不是失败。
    """
    report = _load_report()
    if report is None:
        pytest.skip(f"未找到 {REPORT_PATH.name}，跳过指标下限校验")

    if report.get("partial"):
        pytest.skip("报告来自部分题目（--ids/--limit），不下结论")

    summary = report["summary"]
    failures: list[str] = []

    def check(label: str, metric: dict[str, Any] | None, floor: float) -> None:
        if not metric or metric.get("value") is None:
            failures.append(f"{label}：报告中没有该指标（n={metric.get('n') if metric else 0}）")
            return
        if metric["value"] + 1e-9 < floor:
            failures.append(
                f"{label}：{metric['value']:.4f} < 下限 {floor}（n={metric['n']}）"
            )

    retrieval = summary["retrieval"]
    check(f"hit_at_{3}", retrieval.get("hit_at_3"), FLOORS["hit_at_3"])
    check("mrr", retrieval.get("mrr"), FLOORS["mrr"])
    check("version_accuracy", retrieval.get("version_accuracy"), FLOORS["version_accuracy"])

    if report.get("mode") == "retrieval-only":
        assert not failures, "检索类指标低于下限：" + "；".join(failures)
        return

    generation = summary["generation"]
    check("refusal_rate", generation.get("refusal_rate"), FLOORS["refusal_rate"])
    check(
        "judge_refusal_rate",
        generation.get("judge_refusal_rate"),
        FLOORS["judge_refusal_rate"],
    )
    check("faithfulness", generation.get("faithfulness"), FLOORS["faithfulness"])
    check("citation_accuracy", generation.get("citation_accuracy"), FLOORS["citation_accuracy"])
    assert not failures, "指标低于下限：" + "；".join(failures)


def test_report_has_no_stale_leakage() -> None:
    """启用时效过滤时，召回结果不得含任何已过期版本的块。

    这是硬不变量（0 容差），不是统计指标：出现一次就说明过滤链路被绕过
    （向量下推 / BM25 侧过滤 / MCP 通道三者中任一失效）。
    """
    report = _load_report()
    if report is None:
        pytest.skip(f"未找到 {REPORT_PATH.name}，跳过过期泄漏校验")
    if report.get("partial"):
        pytest.skip("报告来自部分题目，不下结论")
    if report.get("include_expired"):
        pytest.skip("本次评测刻意纳入过期文档，该不变量不适用")

    assert report["summary"]["retrieval"]["stale_hit_chunks"] == 0, (
        "召回结果中出现了已过期版本的块 —— 时效过滤被绕过。"
        f" 逐题明细见 {REPORT_PATH.name} 的 cases[].stale_hits。"
    )
