#!/usr/bin/env bash
# ============================================================================
# 阶段 7 评估入口。
#
# 两档评测，按「花不花 LLM 配额」区分：
#
#   quick    纯检索（with_answer=false），零 LLM 配额。可随时跑、可进 CI。
#            同时带 --contrast：额外跑一遍 include_expired=true 的检索，对照结果
#            内联在同一份报告的 summary_include_expired 里，用来量化「版本时效
#            过滤」这一设计的独立贡献。对照只多花检索、不花 LLM 配额。
#   full     检索 + 生成 + LLM-as-judge。产出可写进简历的完整指标。
#
# 只有两档：对照数据内联在检索档报告里，再拆一档「只跑对照」会产出与前一份
# 几乎相同的文件。
#
# 用法：
#   scripts/run_eval.sh              # 跑全部两档
#   scripts/run_eval.sh quick        # 只跑纯检索档（含时效对照）
#   scripts/run_eval.sh full         # 只跑全量档
#
# 解释器：本项目所有脚本统一用 kbagent 环境。裸 `python` 会命中系统 3.12.5，
# 那里没装依赖，会把「缺依赖」误判成代码错误。
# ============================================================================
set -euo pipefail

PY="${PY:-D:/miniconda3/envs/kbagent/python.exe}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONIOENCODING=utf-8

MODE="${1:-all}"

run_quick() {
  echo "=== [1/2] 检索档（纯检索 + 时效对照，零 LLM 配额） ==="
  # --contrast 的对照结果内联在这份报告里（summary_include_expired），
  # 不另写文件：README 里那张「启用 / 关闭时效过滤」对照表就取自它。
  "$PY" -m eval.eval --retrieval-only --contrast --out eval/report_retrieval.json
}

run_full() {
  echo "=== [2/2] 全量档（检索 + 生成 + judge） ==="
  # 全量档写 report.json：它是仓库里的「最新权威结果」，
  # tests/test_regression.py 会对它校验指标下限。
  "$PY" -m eval.eval --out eval/report.json --compare
}

case "$MODE" in
  quick|contrast) run_quick ;;
  full)           run_full ;;
  all)
    run_quick
    run_full
    echo
    echo "完成。报告：eval/report.json（全量）、eval/report_retrieval.json（检索 + 对照）"
    ;;
  *)
    echo "未知模式：$MODE（可选 quick / full / all）" >&2
    exit 2
    ;;
esac
