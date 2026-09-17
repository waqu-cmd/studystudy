"""评估主脚本（阶段 7）：跑全量 golden set，输出 JSON 报告 + 控制台表格。

用法
----
::

    # 纯检索档：不调 LLM，零生成配额，只测检索类指标
    python -m eval.eval --retrieval-only

    # 全量档：检索 + 生成 + LLM-as-judge
    python -m eval.eval

    # 只跑部分题目 / 不送 judge（省配额）
    python -m eval.eval --ids q1,t1 --no-judge
    python -m eval.eval --limit 5

    # 量化「时效过滤」的独立贡献：额外跑一遍 include_expired=true 的纯检索对照
    python -m eval.eval --retrieval-only --contrast

    # 与阶段 1 的 baseline.json 逐题对比
    python -m eval.eval --compare

两档评测为什么都要保留
----------------------
检索质量与生成质量是**两个可独立退化**的东西：换 embedding 模型只影响前者，
改 prompt 只影响后者。混在一档里跑，一旦指标掉了无法归因。
纯检索档还有个实际好处 —— ``with_answer=false`` 时 supervisor 会跳过 LLM 调用
（见 ``nodes/supervisor.py``），因此整档评测**不消耗任何生成配额**，
可以随便跑、放进 CI。

报告为什么写成与 ``baseline.json`` 同构的 JSON
---------------------------------------------
蓝图要求「对比阶段 1 baseline 和阶段 5 最终版，记录提升幅度」。同构才能逐题对比：
本文件产出的 ``cases[].hit_at_k`` / ``reciprocal_rank`` 与 baseline 的
``cases[].hit_at_3`` / ``reciprocal_rank`` 一一对应，``--compare`` 据此算差。
字段名从 ``hit_at_3`` 换成 ``hit_at_k`` 是因为 K 现在是可配参数
（``DEFAULT_K``），把它写死进字段名会在改 K 时变成谎言。

``as_of_date`` 为什么必须进报告
-------------------------------
本项目的检索结果**依赖当天日期**：``sales_policy_2026q3`` 在 2026-09-30 之后
转为过期，12 个有效块降到 8 个，全部销售类题目的标答随之不可检索。
因此任何一份报告只有在「采集日期」这一前提下才可比较 —— 把日期写进报告、
并让 ``test_regression.py`` 在时间窗失效时直接失败，比事后对着一份过期数字
反复排查要省事得多。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import APP_VERSION, settings  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.state import initial_state  # noqa: E402
from eval import metrics  # noqa: E402
from eval import judge as judge_mod  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
BASELINE_PATH = EVAL_DIR / "baseline.json"
DEFAULT_REPORT_PATH = EVAL_DIR / "report.json"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """从 dict 或 pydantic 模型里取字段。

    召回块在两条通道里形态不同：MCP 通道把块以 JSON 字典送回，直连通道给的是
    ``RetrievedChunk`` 模型。报告层不应关心这个差异。
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> dict[str, Any]:
    """读取 golden set。解析失败直接抛 —— 评估的前提文件坏掉必须立刻暴露。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("cases"):
        raise ValueError(f"golden set 结构异常：{path}")
    return payload


def select_cases(
    payload: dict[str, Any], *, ids: str = "", limit: int = 0
) -> list[dict[str, Any]]:
    """按 ``--ids`` / ``--limit`` 筛选题目。顺序始终以文件顺序为准。"""
    cases = list(payload["cases"])
    if ids.strip():
        wanted = [item.strip() for item in ids.split(",") if item.strip()]
        index = {str(c.get("id")): c for c in cases}
        missing = [item for item in wanted if item not in index]
        if missing:
            raise ValueError(f"golden set 中不存在这些 id：{missing}")
        cases = [index[item] for item in wanted]
    if limit and limit > 0:
        cases = cases[:limit]
    return cases


def corpus_facts() -> dict[str, Any]:
    """从 Chroma 读出语料现状与「已过期版本」集合。

    ``stale_versions`` 不是写死的常量，而是**按当天日期现算**的：语料扩容或
    日期推进后它必须自动跟上，否则回归测试里的「零过期泄漏」不变量会失效。
    注意某个版本可能同时存在有效与过期文档（同名版本的两篇），只有**完全过期**
    的版本才计入 ``stale_versions``。
    """
    from app.rag.chunker import NO_EXPIRE_ORD  # noqa: PLC0415
    from app.rag.indexer import get_collection  # noqa: PLC0415
    from app.rag.retriever import today_ord  # noqa: PLC0415

    collection = get_collection()
    data = collection.get(include=["metadatas"])
    metadatas = [m or {} for m in (data.get("metadatas") or [])]
    today = today_ord()

    docs_all: set[str] = set()
    docs_active: set[str] = set()
    versions_active: set[str] = set()
    versions_stale: set[str] = set()
    chunks_active = 0

    for meta in metadatas:
        doc_id = str(meta.get("doc_id") or "")
        version = str(meta.get("version") or "")
        expire = int(meta.get("expire_ord", NO_EXPIRE_ORD))
        docs_all.add(doc_id)
        if expire >= today:
            chunks_active += 1
            docs_active.add(doc_id)
            versions_active.add(version)
        else:
            versions_stale.add(version)

    return {
        "collection": collection.name,
        "documents": len(docs_all),
        "documents_active": len(docs_active),
        "chunks_total": len(metadatas),
        "chunks_active": chunks_active,
        "chunks_expired": len(metadatas) - chunks_active,
        "stale_versions": sorted(versions_stale - versions_active),
        "as_of_date": date.today().isoformat(),
    }


# --------------------------------------------------------------------------- #
# 单题执行
# --------------------------------------------------------------------------- #


def _invoke_once(
    graph: Any,
    case: dict[str, Any],
    *,
    with_answer: bool,
    top_k: int | None,
    include_expired: bool,
    stale_versions: set[str],
) -> tuple[dict[str, Any], list[Any]]:
    """跑一题（单次尝试），返回 ``(报告用字典, 原始召回块列表)``。

    原始召回块要单独返回：judge 必须看到**与生成器完全相同**的参考资料才公平，
    而报告 JSON 里只保留 chunk_id 与 version，不带正文。

    图抛异常时**不中断整轮评测** —— 35 题里有一题因为网络抖动挂掉，
    剩下 34 题的数据仍然有效。异常收敛进 ``error`` 字段，并在摘要里计数。
    """
    relevant = list(case.get("relevant_chunk_ids") or [])
    required = list(case.get("required_doc_versions") or [])

    state = initial_state(
        case["query"],
        top_k=top_k,
        include_expired=include_expired,
        with_answer=with_answer,
    )

    started = time.perf_counter()
    error = ""
    try:
        result: dict[str, Any] = graph.invoke(state)
    except Exception as exc:  # noqa: BLE001 - 单题失败不应中断整批
        result = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    chunks = list(result.get("retrieved_chunks") or [])
    chunk_ids = [str(_get(item, "chunk_id", "")) for item in chunks]
    chunk_versions = [str(_get(item, "version", "")) for item in chunks]
    citations = list(result.get("citations") or [])
    citation_ids = [str(_get(item, "chunk_id", "")) for item in citations]

    report_case = {
        "id": case.get("id"),
        "query": case.get("query"),
        "category": case.get("category"),
        "trap_type": case.get("trap_type"),
        "should_refuse": bool(case.get("should_refuse")),
        "relevant_chunk_ids": relevant,
        "required_doc_versions": required,
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_versions": chunk_versions,
        "stale_versions": sorted(stale_versions),
        "hit_at_k": metrics.hit_at_k(chunk_ids, relevant, metrics.DEFAULT_K),
        "reciprocal_rank": metrics.reciprocal_rank(chunk_ids, relevant),
        "version_accuracy": metrics.version_accuracy(chunk_versions, required),
        "stale_hits": metrics.count_stale(chunk_versions, stale_versions),
        "answer": str(result.get("answer") or ""),
        "citations": citation_ids,
        "route": str(result.get("route") or ""),
        "route_reason": str(result.get("route_reason") or ""),
        "sub_queries": [str(q) for q in (result.get("sub_queries") or [])],
        "verdict": str(result.get("verdict") or ""),
        "unsupported_claims": [str(c) for c in (result.get("unsupported_claims") or [])],
        "retry_count": int(result.get("retry_count") or 0),
        "retrieval_attempts": int(result.get("retrieval_attempts") or 0),
        "llm_model": str(result.get("llm_model") or ""),
        "elapsed_ms": elapsed_ms,
        "error": error or str(result.get("error") or ""),
    }
    return report_case, chunks


def run_case(
    graph: Any,
    case: dict[str, Any],
    *,
    with_answer: bool,
    top_k: int | None,
    include_expired: bool,
    stale_versions: set[str],
    retries: int = 0,
) -> tuple[dict[str, Any], list[Any]]:
    """跑一题，对**报错的题**至多重试 ``retries`` 次。

    为什么需要重试
    --------------
    实测出现过一次 ``search_documents 调用超时（>30.0s）``—— 与一次 embedding
    接口重试同时发生，属外部服务瞬时抖动，不是代码缺陷。但它的后果很重：
    该题召回为空，于是 Hit@3 / MRR / 版本准确率**三个指标同时**把这一题记成 0 分，
    一次 13 分钟的整体评估就被一次 30 秒的网络抖动污染了。

    重试为什么必须留下痕迹
    ----------------------
    每次重试都把**历次失败原因**累计进 ``error_history``、
    重试次数写进 ``error_retries``。这样「用重试把真实故障盖住」做不到 ——
    报告里能看见重试了几次、每次原本错在哪；若同一题每次都被重试，
    那本身就是需要修的信号。

    只对**报错**的题重试，不对「召回为空但没报错」的题重试：后者是检索质量
    本身的结论，重试等于用随机性粉饰指标。
    """
    attempt = 0
    history: list[str] = []
    while True:
        report_case, chunks = _invoke_once(
            graph,
            case,
            with_answer=with_answer,
            top_k=top_k,
            include_expired=include_expired,
            stale_versions=stale_versions,
        )
        error = str(report_case.get("error") or "")
        if not error or attempt >= max(0, retries):
            report_case["error_retries"] = attempt
            report_case["error_history"] = history
            return report_case, chunks
        history.append(error)
        attempt += 1


def attach_judgments(
    cases: list[dict[str, Any]],
    chunks_by_id: dict[str, list[Any]],
    engine: Any,
) -> None:
    """就地给每题挂上 ``judge`` 结果。

    空答案不送评：它多半来自检索失败或生成失败，送评只会消耗配额，
    而且 judge 对一个空答案给出的分数没有解释力（真正的原因在 ``error`` 里）。
    """
    for case in cases:
        answer = str(case.get("answer") or "")
        if not answer.strip():
            case["judge"] = judge_mod.failed_result("答案为空，未送评")
            continue
        case["judge"] = judge_mod.judge_answer(
            engine,
            question=str(case.get("query") or ""),
            answer=answer,
            chunks=chunks_by_id.get(str(case.get("id")), []),
        )


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def build_report(
    cases: list[dict[str, Any]],
    *,
    facts: dict[str, Any],
    args: argparse.Namespace,
    contrast: list[dict[str, Any]] | None = None,
    elapsed_s: float = 0.0,
) -> dict[str, Any]:
    """组装报告。字段骨架与 ``baseline.json`` 对齐，便于 ``--compare``。"""
    with_answer = not args.retrieval_only
    summary = metrics.summarize(
        cases, k=metrics.DEFAULT_K, with_answer=with_answer
    )

    report: dict[str, Any] = {
        "stage": "stage7",
        "app_version": APP_VERSION,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": facts["as_of_date"],
        "mode": "retrieval-only" if args.retrieval_only else "full",
        "partial": bool(args.ids.strip() or args.limit),
        "include_expired": bool(args.include_expired),
        "pipeline": (
            "Supervisor(LLM 路由 + Send 并行) -> MCP 混合检索(向量+BM25,RRF k={}) "
            "-> Synthesizer -> Verifier(自纠正, max_retry={})"
        ).format(settings.rrf_k, settings.max_retry),
        "config": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "judge_model": settings.judge_model_name,
            "verifier_model": settings.verifier_model_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "top_k": args.top_k or settings.top_k,
            "rrf_k": settings.rrf_k,
            "max_retry": settings.max_retry,
            "memory_enabled": settings.memory_enabled,
            "mcp_enabled": settings.mcp_enabled,
            "mcp_servers": settings.mcp_servers,
        },
        "corpus": {
            "documents": facts["documents"],
            "documents_active": facts["documents_active"],
            "chunks_total": facts["chunks_total"],
            "chunks_active": facts["chunks_active"],
            "chunks_expired": facts["chunks_expired"],
            "stale_versions": facts["stale_versions"],
        },
        "golden_set": {
            "path": str(GOLDEN_SET_PATH.name),
            "version": args.golden_version,
            "n_cases": len(cases),
        },
        "summary": summary,
        "elapsed_s": round(elapsed_s, 1),
        "cases": cases,
    }

    if contrast is not None:
        report["cases_include_expired"] = [
            {
                "id": c["id"],
                "retrieved_chunk_ids": c["retrieved_chunk_ids"],
                "retrieved_versions": c["retrieved_versions"],
                "hit_at_k": c["hit_at_k"],
                "reciprocal_rank": c["reciprocal_rank"],
                "version_accuracy": c["version_accuracy"],
                "stale_hits": c["stale_hits"],
            }
            for c in contrast
        ]
        report["summary_include_expired"] = metrics.summarize(
            contrast, k=metrics.DEFAULT_K, with_answer=False
        )

    return report


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "  n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_line(label: str, block: dict[str, Any]) -> str:
    return f"  {label:<22} {_fmt(block.get('value')):>8}   (n={block.get('n', 0)})"


def print_report(report: dict[str, Any]) -> None:
    """控制台表格。只打印可读摘要，完整数据在 JSON 里。"""
    summary = report["summary"]
    corpus = report["corpus"]
    print()
    print("=" * 72)
    print(f"阶段 7 评估报告 | 模式={report['mode']} | 采集={report['captured_at']}")
    print("=" * 72)
    print(
        f"语料：{corpus['documents']} 篇（有效 {corpus['documents_active']}） | "
        f"{corpus['chunks_total']} 块（有效 {corpus['chunks_active']} / "
        f"过期 {corpus['chunks_expired']}） | 过期版本={corpus['stale_versions']}"
    )
    print(
        f"题目：{summary['n_cases']} 题 = 有标答 {summary['n_answerable']} + "
        f"应拒答 {summary['n_trap']} | 耗时 {report['elapsed_s']}s"
    )

    retrieval = summary["retrieval"]
    print()
    print(f"检索指标（分母=有标答题 {summary['n_answerable']}）")
    print(_metric_line(f"Hit@{metrics.DEFAULT_K}", retrieval[f"hit_at_{metrics.DEFAULT_K}"]))
    print(_metric_line("MRR", retrieval["mrr"]))
    print(_metric_line("版本准确率", retrieval["version_accuracy"]))
    print(
        f"  {'过期版本泄漏':<20} {retrieval['stale_hit_chunks']:>8}   "
        f"(涉及 {retrieval['stale_hit_cases']} 题；启用过滤时应恒为 0)"
    )

    generation = summary["generation"]
    print()
    if generation.get("measured"):
        print(
            f"生成指标（拒答分母={summary['n_trap']}；评分分母={generation['judged_cases']}）"
        )
    else:
        print("生成指标：未测量（--retrieval-only 下不生成答案，故拒答率与忠实度无定义）")
    print(_metric_line("拒答率(严格:零引用)", generation["refusal_rate"]))
    print(_metric_line("拒答率(纯文本)", generation["refusal_rate_text"]))
    print(_metric_line("拒答率(judge 语义)", generation["judge_refusal_rate"]))
    print(_metric_line("忠实度(judge)", generation["faithfulness"]))
    print(_metric_line("引用准确率(judge)", generation["citation_accuracy"]))
    print(
        f"  {'无依据断言总数':<20} {generation['unsupported_claims_total']:>8}   "
        f"(judge 未执行 {generation['judge_failed_cases']} 题)"
    )

    diagnostics = generation.get("refusal_diagnostics") or []
    if diagnostics:
        print()
        print(
            f"拒答口径存在分歧的题（{len(diagnostics)}）"
            "—— 严格口径与 judge 语义判定不一致，需人工一眼定论："
        )
        for item in diagnostics:
            print(
                f"  {item['id']}: 严格={item['strict_refusal']} "
                f"纯文本={item['text_refusal']} judge={item['judge_refusal_correct']} "
                f"引用={item['citations']}"
            )
            print(f"      {item['answer'][:110]}")

    lifecycle = summary["lifecycle"]
    print()
    print(
        "自纠正与路由"
        + ("" if lifecycle["measured_with_answer"] else "（Verifier 相关未测量）")
    )
    print(_metric_line("Verifier 触发率", lifecycle["verifier_trigger_rate"]))
    print(
        f"  {'平均检索轮次':<20} {_fmt(lifecycle['avg_retrieval_attempts']):>8}   "
        f"平均回退次数 {_fmt(lifecycle['avg_retry_count'])}"
    )
    print(f"  {'路由分布':<20} {lifecycle['route_counts']}")

    errors = lifecycle["error_cases"]
    if errors:
        print()
        print(f"仍报错的题目（{len(errors)}）—— 这些题目的检索类指标已受影响：")
        for item in errors:
            print(f"  [{item['id']}] {(item['error'] or '')[:100]}")

    retried = [c for c in report["cases"] if int(c.get("error_retries") or 0) > 0]
    if retried:
        print()
        print(f"经重试后通过的题目（{len(retried)}）—— 原始错误保留在 error_history：")
        for case in retried:
            history = case.get("error_history") or [""]
            print(
                f"  [{case['id']}] 重试 {case['error_retries']} 次 | "
                f"{str(history[-1])[:88]}"
            )

    traps = [c for c in report["cases"] if c.get("should_refuse")]
    if traps and report["summary"]["generation"].get("measured"):
        print()
        print("陷阱题逐条")
        for case in traps:
            clean = metrics.is_clean_refusal(case["answer"], case["citations"])
            flag = "OK " if clean else "!! "
            print(
                f"  {flag}{case['id']:<4} {case['trap_type']:<18} "
                f"引用={len(case['citations'])} 召回={len(case['retrieved_chunk_ids'])} "
                f"{(case['answer'] or '')[:44]}"
            )
    print()


def compare_with_baseline(report: dict[str, Any], path: Path = BASELINE_PATH) -> None:
    """与阶段 1 的 ``baseline.json`` 逐题对比。

    只对**两边都有的 id** 比 —— 阶段 1 只有 6 题（q1~q5 + t1），阶段 7 有 35 题。
    其余 29 题没有可比对象，它们本身就是阶段 7 的增量（版本准确率、拒答率、
    忠实度、Verifier 触发率在阶段 1 都还不存在）。
    """
    if not path.exists():
        print(f"未找到基线文件：{path}")
        return

    baseline = json.loads(path.read_text(encoding="utf-8"))
    old_by_id = {str(c.get("id")): c for c in (baseline.get("cases") or [])}
    new_by_id = {str(c.get("id")): c for c in report["cases"]}

    shared = [cid for cid in old_by_id if cid in new_by_id]
    print()
    print("=" * 72)
    print(f"与阶段 1 基线对比（baseline.json 共 {len(old_by_id)} 题，可比 {len(shared)} 题）")
    print("=" * 72)
    print(f"  {'id':<5}{'阶1 Hit@3':>10}{'阶7 Hit@3':>12}{'阶1 MRR':>10}{'阶7 MRR':>10}  判定")
    print("  " + "-" * 62)

    regressed = 0
    for cid in sorted(shared):
        old, new = old_by_id[cid], new_by_id[cid]
        old_hit = old.get("hit_at_3")
        old_mrr = old.get("reciprocal_rank")
        new_hit = new.get("hit_at_k")
        new_mrr = new.get("reciprocal_rank")

        verdict = "一致"
        if old_mrr is not None and new_mrr is not None:
            if new_mrr + 1e-9 < old_mrr:
                verdict = "退化 !!"
                regressed += 1
            elif new_mrr > old_mrr + 1e-9:
                verdict = "提升"
        print(
            f"  {cid:<5}{str(old_hit):>10}{str(new_hit):>12}"
            f"{_fmt(old_mrr, 3):>10}{_fmt(new_mrr, 3):>10}  {verdict}"
        )

    print()
    print("阶段 1 未测量的指标（阶段 7 新增，其数值本身即为增量）")
    generation = report["summary"]["generation"]
    lifecycle = report["summary"]["lifecycle"]
    print(
        f"  版本准确率 {_fmt(report['summary']['retrieval']['version_accuracy'].get('value'))}"
        f" | 拒答率 {_fmt(generation['refusal_rate'].get('value'))}"
        f" | 忠实度 {_fmt(generation['faithfulness'].get('value'))}"
        f" | 引用准确率 {_fmt(generation['citation_accuracy'].get('value'))}"
        f" | Verifier 触发率 {_fmt(lifecycle['verifier_trigger_rate'].get('value'))}"
    )
    if regressed:
        print(f"  !! 有 {regressed} 题出现退化，需人工确认。")
    print()


def print_contrast(report: dict[str, Any]) -> None:
    """打印时效过滤的对照结果，量化该设计的独立贡献。"""
    if "summary_include_expired" not in report:
        return
    with_filter = report["summary"]["retrieval"]
    without = report["summary_include_expired"]["retrieval"]
    print("=" * 72)
    print("时效过滤的独立贡献对照（同一批题、同一检索算法，只切换 expire_ord 过滤）")
    print("=" * 72)
    print(f"  {'指标':<16}{'启用过滤':>12}{'关闭过滤':>12}")
    print("  " + "-" * 40)
    for key, label in (
        (f"hit_at_{metrics.DEFAULT_K}", f"Hit@{metrics.DEFAULT_K}"),
        ("mrr", "MRR"),
        ("version_accuracy", "版本准确率"),
    ):
        print(
            f"  {label:<16}{_fmt(with_filter[key].get('value')):>12}"
            f"{_fmt(without[key].get('value')):>12}"
        )
    print(
        f"  {'过期版本泄漏':<14}{with_filter['stale_hit_chunks']:>12}"
        f"{without['stale_hit_chunks']:>12}"
    )
    print()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="企业知识库 Agent 评估（阶段 7）")
    parser.add_argument("--ids", default="", help="只跑指定题目，逗号分隔，如 q1,t1")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    parser.add_argument("--top-k", type=int, default=0, help="覆盖 TOP_K")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只检索不生成（with_answer=false），零 LLM 配额",
    )
    parser.add_argument("--no-judge", action="store_true", help="不送 LLM-as-judge")
    parser.add_argument(
        "--include-expired",
        action="store_true",
        help="本次主评测纳入已过期文档（对照实验用）",
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="额外跑一遍时效设置相反的纯检索对照，量化时效过滤的贡献",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="单题报错后的重试次数（默认 1）。只对报错题重试，不对空召回重试",
    )
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="报告输出路径")
    parser.add_argument("--compare", action="store_true", help="与阶段 1 baseline 对比")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    payload = load_golden_set()
    args.golden_version = payload.get("version", "")
    cases = select_cases(payload, ids=args.ids, limit=args.limit)

    facts = corpus_facts()
    stale_versions = set(facts["stale_versions"])
    top_k = args.top_k or None

    graph = build_graph(checkpointer=None)
    started = time.perf_counter()

    report_cases: list[dict[str, Any]] = []
    chunks_by_id: dict[str, list[Any]] = {}
    for index, case in enumerate(cases, start=1):
        report_case, chunks = run_case(
            graph,
            case,
            with_answer=not args.retrieval_only,
            top_k=top_k,
            include_expired=bool(args.include_expired),
            stale_versions=stale_versions,
            retries=args.retries,
        )
        report_cases.append(report_case)
        chunks_by_id[str(case.get("id"))] = chunks

    if not args.retrieval_only and not args.no_judge:
        engine = judge_mod.get_judge_llm()
        attach_judgments(report_cases, chunks_by_id, engine)

    contrast: list[dict[str, Any]] | None = None
    if args.contrast:
        contrast = []
        for case in cases:
            item, _ = run_case(
                graph,
                case,
                with_answer=False,
                top_k=top_k,
                include_expired=not args.include_expired,
                stale_versions=stale_versions,
                retries=args.retries,
            )
            contrast.append(item)

    elapsed_s = time.perf_counter() - started
    report = build_report(
        report_cases, facts=facts, args=args, contrast=contrast, elapsed_s=elapsed_s
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_report(report)
    if contrast is not None:
        print_contrast(report)
    if args.compare:
        compare_with_baseline(report)

    print(f"报告已写入：{out_path}")

    # 评估脚本没有 FastAPI lifespan 来收尾，必须自己关掉 MCP 子进程，
    # 否则进程退出前会留下悬空的 stdio 子进程。
    from app.mcp_client import reset_default_mcp  # noqa: PLC0415

    reset_default_mcp()

    errors = report["summary"]["lifecycle"]["error_cases"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
