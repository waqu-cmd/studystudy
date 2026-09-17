"""批量导入 data/docs/ 下的测试文档（阶段 1）。

用法：
    python scripts/seed_docs.py                    # 索引 data/docs/ 全部受支持文件
    python scripts/seed_docs.py --skip-unchanged   # 跳过内容 hash 未变的文档，省 Embedding 费用
    python scripts/seed_docs.py --reset            # 先清空 collection 再重建
    python scripts/seed_docs.py --dir data/docs --only sales_policy_2026q3

退出码：0 全部成功；1 存在失败项。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从任意工作目录直接运行本脚本
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.rag.indexer import SUPPORTED_SUFFIXES, IndexOutcome, Indexer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量导入知识库文档")
    parser.add_argument(
        "--dir",
        default=str(settings.docs_dir),
        help="文档目录，默认取 .env 的 DOCS_DIR",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="索引前清空整个 collection（危险操作，会删除全部已索引数据）",
    )
    parser.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="内容 hash 未变化的文档直接跳过，不重新调用 Embedding",
    )
    parser.add_argument(
        "--only",
        default="",
        help="只索引文件名包含该子串的文档，便于单篇调试",
    )
    return parser.parse_args()


def confirm_reset() -> bool:
    """破坏性操作的二次确认。非交互终端（CI）下直接返回 True。"""
    print("!" * 68)
    print("警告：--reset 会删除 collection 内的全部已索引数据，不可撤销。")
    print(f"      collection = {settings.chroma_collection}")
    print(f"      路径       = {settings.chroma_path}")
    print("!" * 68)
    if not sys.stdin.isatty():
        print("非交互终端，直接继续。")
        return True
    reply = input("确认继续请输入 yes：").strip().lower()
    return reply == "yes"


def _collect_files(target: Path) -> list[Path]:
    """目录下全部受支持文件，按文件名排序以保证执行顺序确定。"""
    return sorted(
        p
        for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def print_table(outcomes: list[IndexOutcome]) -> None:
    header = f"{'状态':<9}{'块数':>5}  {'版本':<9}{'文档ID':<28}{'文件'}"
    print()
    print(header)
    print("-" * 110)
    for outcome in outcomes:
        name = Path(outcome.path).name
        print(
            f"{outcome.status:<9}{outcome.chunks:>5}  "
            f"{outcome.version or '-':<9}{outcome.doc_id or '-':<28}{name}"
            + (f"   <- {outcome.message}" if outcome.message else "")
        )


def main() -> int:
    args = parse_args()

    indexer = Indexer()

    if args.reset:
        if not confirm_reset():
            print("已取消。")
            return 0
        indexer.reset()

    target = Path(args.dir)
    if not target.is_absolute():
        target = ROOT / target
    if not target.is_dir():
        print(f"目录不存在：{target}")
        return 1

    # --only 在索引前先筛选，避免为不需要的文档白付 Embedding 费用
    files = _collect_files(target)
    if args.only:
        files = [p for p in files if args.only in p.name]
    if not files:
        print(f"没有找到可索引的文件（目录：{target}，筛选条件：{args.only or '无'}）")
        return 1

    outcomes = indexer.index_paths(
        [str(p) for p in files], force=not args.skip_unchanged
    )

    print_table(outcomes)

    indexed = sum(1 for o in outcomes if o.status == "indexed")
    skipped = sum(1 for o in outcomes if o.status == "skipped")
    failed = sum(1 for o in outcomes if o.status == "failed")

    print()
    print(
        f"汇总：文件 {len(outcomes)} | 新增/更新 {indexed} | 跳过 {skipped} | "
        f"失败 {failed} | 本次切块 {sum(o.chunks for o in outcomes)}"
    )
    print(f"collection：{indexer.collection_name} 共 {indexer.count()} 块")

    if failed:
        print()
        print("存在失败项，请检查上面的 message 列。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
