"""MCP Server 与客户端的测试（阶段 4 起，阶段 5 扩充第 4 层）。

分四层
------
1. **纯函数层**：内容解析、异常展平、配置解析 —— 无副作用，占多数用例。
2. **工具函数层**：直接调用 Server 模块里的工具函数（不 spawn 进程）。
   选用 filesystem_server，因为它不触发 embedding 网络请求；chroma_server 只测
   空检索词的短路路径。
3. **端到端层**：真实 spawn stdio 子进程，验证「动态发现 → 调用 → 结果解析 →
   健康检查 → 降级」整条链路。同样用 filesystem_server 以避免网络依赖。
4. **并发启动层**（阶段 5 回归）：用替身把「握手窗口」拉长，验证并发 ``start()``
   不重入。真实场景是 ``Send`` fan-out 让 N 个 retriever 分支同时惰性启动，
   该缺陷曾导致 ``search_documents`` 未注册即被调用。

为什么端到端层允许 skip
--------------------
该层需要 spawn Python 子进程。若运行环境禁止创建子进程，跳过比失败更诚实 ——
它反映环境限制而非代码缺陷。跳过时会打印 status_map，便于区分「环境不允许」
与「Server 真的起不来」。
"""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import SecretStr

from app.api.routes.health import STATUS_DEGRADED, STATUS_OK, _evaluate_status
from app.mcp_client import (
    MCPToolClient,
    MCPToolResult,
    SERVER_FILESYSTEM,
    SERVER_MODULES,
    build_stdio_connection,
    enabled_server_names,
    flatten_exception,
    parse_tool_content,
    summarize_exception,
)
from app.mcp_servers import filesystem_server, search_server

# ============================== 1. 纯函数层 ==============================


def test_flatten_exception_unwraps_nested_groups() -> None:
    """anyio 的嵌套 ExceptionGroup 必须被完整展平 —— 否则只剩无信息量的外层文案。"""
    inner = ValueError("真因在这里")
    group = BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [inner])])

    lines = flatten_exception(group)

    assert len(lines) == 3
    assert "ExceptionGroup" in lines[0]
    assert "ValueError: 真因在这里" in lines[2]


def test_summarize_exception_picks_leaf() -> None:
    inner = RuntimeError("Connection closed")
    group = BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [inner])])

    assert summarize_exception(group) == "RuntimeError: Connection closed"


def test_summarize_exception_handles_plain_exception() -> None:
    assert summarize_exception(ValueError("单层")) == "ValueError: 单层"


def test_parse_tool_content_single_json_object() -> None:
    """工具返回 dict → 1 个 TextContent，文本是 JSON。"""

    class FakeText:
        text = '{"chunk_id": "a_p0", "score": 0.5}'

    data, text = parse_tool_content([FakeText()])

    assert data == {"chunk_id": "a_p0", "score": 0.5}
    assert "chunk_id" in text


def test_parse_tool_content_list_of_json() -> None:
    """工具返回 list → N 个 TextContent，每个元素一个 JSON（实测行为）。"""

    class FakeText:
        def __init__(self, value: str) -> None:
            self.text = value

    content = [FakeText('{"i": 1}'), FakeText('{"i": 2}')]
    data, _ = parse_tool_content(content)

    assert data == [{"i": 1}, {"i": 2}]


def test_parse_tool_content_plain_text_falls_back() -> None:
    """任一元素不是 JSON 就整体回退为文本，避免半结构化结果被误用。"""

    class FakeText:
        text = "noisy:ok"

    data, text = parse_tool_content([FakeText()])

    assert data == "noisy:ok"
    assert text == "noisy:ok"


def test_parse_tool_content_mixed_returns_text_form() -> None:
    class FakeText:
        def __init__(self, value: str) -> None:
            self.text = value

    content = [FakeText('{"ok": true}'), FakeText("纯文本")]
    data, _ = parse_tool_content(content)

    assert isinstance(data, list)
    assert data[1] == "纯文本"


def test_parse_tool_content_empty() -> None:
    assert parse_tool_content(None) == (None, "")
    assert parse_tool_content([]) == (None, "")


def test_tool_result_as_items_normalizes_shapes() -> None:
    base = {"ok": True, "server": "chroma", "tool": "search_documents"}

    assert MCPToolResult(**base, data={"chunk_id": "a"}).as_items() == [{"chunk_id": "a"}]
    assert MCPToolResult(**base, data=[{"a": 1}, {"b": 2}]).as_items() == [{"a": 1}, {"b": 2}]
    # 非 dict 元素（例如工具失败时混入的诊断条目）会被过滤
    assert MCPToolResult(**base, data=[{"a": 1}, "x", 3]).as_items() == [{"a": 1}]
    assert MCPToolResult(**base, data="文本").as_items() == []
    assert MCPToolResult(**base, data=None).as_items() == []


def test_enabled_server_names_parses_and_ignores_unknown(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_servers", "chroma, filesystem , nonsense")
    assert enabled_server_names() == ["chroma", "filesystem"]

    monkeypatch.setattr(settings, "mcp_servers", "chroma,chroma")
    assert enabled_server_names() == ["chroma"]

    monkeypatch.setattr(settings, "mcp_servers", "")
    assert enabled_server_names() == []


def test_build_stdio_connection_shape() -> None:
    """连接配置必须带完整 env —— mcp 的 stdio_client 在 env 非空时是整体替换。"""
    conn = build_stdio_connection(SERVER_MODULES[SERVER_FILESYSTEM])

    assert conn["transport"] == "stdio"
    assert conn["args"] == ["-m", "app.mcp_servers.filesystem_server"]
    assert conn["env"]["PYTHONIOENCODING"] == "utf-8"
    # PATH 是「整体替换而非合并」的直接证据
    assert "PATH" in conn["env"] or "Path" in conn["env"]


def test_evaluate_status_rules() -> None:
    assert _evaluate_status({}) == STATUS_OK
    assert _evaluate_status({"chroma": "ok", "filesystem": "ok"}) == STATUS_OK
    assert _evaluate_status({"chroma": "ok", "filesystem": "error: x"}) == STATUS_DEGRADED
    assert _evaluate_status({"chroma": "未启动"}) == STATUS_DEGRADED


# ============================== 2. 工具函数层 ==============================


def test_chroma_search_empty_query_short_circuits() -> None:
    """空检索词直接返回空列表 —— 不初始化检索器、不打 embedding 接口。"""
    from app.mcp_servers.chroma_server import search_documents

    assert search_documents("") == []
    assert search_documents("   ") == []


def test_filesystem_list_docs_reads_real_docs() -> None:
    items = filesystem_server.list_docs()

    assert items, "data/docs 下应有示例文档"
    by_id = {item["doc_id"]: item for item in items}
    assert "sales_policy_2026q3" in by_id

    q3 = by_id["sales_policy_2026q3"]
    assert q3["version"] == "2026Q3"
    assert q3["effective_date"] == "2026-07-01"
    assert q3["expire_date"] == "2026-09-30"
    assert q3["chunks"] > 0
    assert q3["error"] == ""

    # 未声明 expire_date 的长期有效文档应写哨兵值，而不是留空
    assert by_id["it_security_policy"]["expire_date"] == "9999-12-31"


def test_filesystem_read_file_returns_content_and_meta() -> None:
    result = filesystem_server.read_file("sales_policy_2026q3.md")

    assert result["doc_id"] == "sales_policy_2026q3"
    assert result["version"] == "2026Q3"
    assert "返点" in result["content"]
    assert result["lines"] > 1


@pytest.mark.parametrize(
    "bad",
    [
        "../.env",
        "..\\..\\.env",
        "../../requirements.txt",
        "../../app/main.py",
        "subdir/../../.env",
    ],
)
def test_filesystem_read_file_rejects_traversal(bad: str) -> None:
    """路径穿越必须被拒绝 —— MCP 工具的入参是不可信输入。"""
    with pytest.raises(ValueError):
        filesystem_server.read_file(bad)


def test_filesystem_read_file_rejects_disallowed_suffix() -> None:
    """白名单之外的类型一律拒绝（.env / .pem 之类既非文档也非二进制）。"""
    with pytest.raises(ValueError, match="文件类型"):
        filesystem_server.read_file("sales_policy_2026q3.py")


def test_filesystem_read_file_missing_file() -> None:
    with pytest.raises(ValueError, match="不存在"):
        filesystem_server.read_file("no_such_doc_xyz.md")


def test_filesystem_read_file_empty_name() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        filesystem_server.read_file("")


def test_search_server_unconfigured_is_explicit(monkeypatch) -> None:
    """未配置时必须返回显式错误，而不是空列表冒充「没搜到」。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "search_api_base", "")
    monkeypatch.setattr(settings, "search_api_key", SecretStr(""))

    assert search_server.is_configured() is False
    result = search_server.web_search("任意问题")

    assert result["results"] == []
    assert "SEARCH_API_BASE" in result["error"]


def test_search_server_empty_query() -> None:
    result = search_server.web_search("")

    assert result["results"] == []
    assert "不能为空" in result["error"]


# ============================== 3. 端到端层 ==============================


def _filesystem_connection() -> dict:
    return {SERVER_FILESYSTEM: build_stdio_connection(SERVER_MODULES[SERVER_FILESYSTEM])}


@pytest.fixture(scope="module")
def live_client():
    """真实 spawn filesystem_server 的客户端。整个模块只启动一次。"""
    client = MCPToolClient(_filesystem_connection(), call_timeout=60, start_timeout=90)
    started = client.start()
    if not started:
        status = client.status_map
        client.stop()
        pytest.skip(f"无法启动 MCP stdio 子进程（环境限制）| status={status}")
    yield client
    client.stop()


def test_end_to_end_discovers_tools(live_client: MCPToolClient) -> None:
    """启动时用 list_tools() 动态发现工具，并建立 工具名 → Server 的路由表。"""
    index = live_client.tool_index

    assert index.get("list_docs") == SERVER_FILESYSTEM
    assert index.get("read_file") == SERVER_FILESYSTEM
    assert live_client.status_map[SERVER_FILESYSTEM] == "ok"


def test_end_to_end_call_list_docs(live_client: MCPToolClient) -> None:
    """真实跨进程调用，验证返回 list 时被解析成 list[dict]。"""
    result = live_client.call_tool("list_docs", {})

    assert result.ok, result.error
    assert result.server == SERVER_FILESYSTEM
    items = result.as_items()
    assert items, "应返回至少一篇文档"
    assert all("doc_id" in item for item in items)
    assert result.elapsed_ms >= 0


def test_end_to_end_call_read_file(live_client: MCPToolClient) -> None:
    result = live_client.call_tool("read_file", {"filename": "sales_policy_2026q3.md"})

    assert result.ok, result.error
    items = result.as_items()
    assert items[0]["doc_id"] == "sales_policy_2026q3"


def test_end_to_end_tool_error_is_flagged_not_raised(
    live_client: MCPToolClient,
) -> None:
    """工具侧异常（路径越界）应以 isError 返回，客户端不抛异常而是 ok=False。"""
    result = live_client.call_tool("read_file", {"filename": "../.env"})

    assert result.ok is False
    assert result.error
    assert "路径越界" in result.error or "不存在" in result.error


def test_end_to_end_unknown_tool(live_client: MCPToolClient) -> None:
    result = live_client.call_tool("no_such_tool", {})

    assert result.ok is False
    assert "未注册" in result.error


def test_end_to_end_probe_reports_ok(live_client: MCPToolClient) -> None:
    checks = live_client.probe()

    assert checks == {SERVER_FILESYSTEM: "ok"}


def test_end_to_end_tools_snapshot_written(live_client: MCPToolClient) -> None:
    """工具清单可落盘核查 —— 「动态发现」不应只存在于内存里。"""
    path = live_client.write_tools_snapshot()

    assert path is not None
    assert path.exists()
    assert "list_docs" in path.read_text(encoding="utf-8")


def test_unavailable_server_degrades_instead_of_raising() -> None:
    """指向不存在的 Server：start() 返回 False 而非抛异常，调用返回 ok=False。

    这正是蓝图「关掉 chroma_server，/health 应报告 unhealthy、/query 应优雅降级」
    的单测形态 —— 不必真的去 kill 一个进程。
    """
    base = build_stdio_connection("x")
    conn = {
        "ghost": {
            "transport": "stdio",
            "command": base["command"],
            "args": ["-m", "app.mcp_servers.no_such_server_xyz"],
            "cwd": base["cwd"],
        }
    }
    client = MCPToolClient(conn, call_timeout=10, start_timeout=40, ping_timeout=5)

    assert client.start() is False
    assert client.status_map["ghost"].startswith("error")
    assert client.tool_index == {}

    result = client.call_tool("search_documents", {"query": "返点"})
    assert result.ok is False
    assert result.error

    client.stop()

    # 完全没有配置 Server 的客户端也要给出可读错误，而不是抛 AttributeError
    bare = MCPToolClient({})
    assert bare.start() is False
    bare_result = bare.call_tool("x", {})
    assert bare_result.ok is False
    assert "未启动" in bare_result.error


# ============================== 4. 并发启动（阶段 5 回归） ==============================


def _fake_connection() -> dict:
    """只为满足「至少配置了一个 Server」这一前置条件，不会真的 spawn 子进程。"""
    return {"fake": build_stdio_connection("x")}


def test_concurrent_start_is_not_reentrant(monkeypatch: pytest.MonkeyPatch) -> None:
    """阶段 5 回归：并发 ``start()`` 只能启动一个后台线程，且不得重清共享状态。

    **缺陷背景**：旧实现用 ``running`` 判「是否已启动」，而 ``running`` 依赖
    ``_loop``，``_loop`` 由后台线程内部创建 —— 在握手完成前它一直是 ``False``。
    阶段 5 的 ``Send`` fan-out 会让 N 个 retriever 分支同时落到惰性启动路径，
    于是每个分支都判定「尚未启动」、依次进入 ``_start_lock`` 把 ``_tool_index``
    清空并另起一个事件循环，``_pending`` 计数随之错乱，``_ready`` 在 chroma 注册
    ``search_documents`` **之前**就放行（实测日志：filesystem 12:50:54 连上，
    chroma 12:50:57 才连上），复合问题因此整体降级为「无法确认」。

    **测试手法**：把 ``_thread_main`` 换成一个被 Event 卡住的替身，人为拉长
    「握手窗口」，让 N 个调用者必然在窗口内竞争；再断言后台线程只被启动一次。
    这比真实 spawn 三个子进程快得多，且结论等价。
    """
    client = MCPToolClient(_fake_connection())
    entered = threading.Event()
    release = threading.Event()
    launches: list[int] = []
    launch_guard = threading.Lock()

    def fake_thread_main(self) -> None:  # noqa: ANN001 - 需绑定到实例
        with launch_guard:
            launches.append(threading.get_ident())
        entered.set()
        # 模拟「握手进行中」：在放行前 _ready 不得被置位，_starting 也必须保持为真
        release.wait(10)
        with self._lock:
            self._pending = 0
            self._tool_index["search_documents"] = "fake"
            self._status["fake"] = "ok"
        self._ready.set()

    monkeypatch.setattr(MCPToolClient, "_thread_main", fake_thread_main)

    barrier = threading.Barrier(3, timeout=10)
    results: list[bool] = []
    result_guard = threading.Lock()

    def worker() -> None:
        barrier.wait()  # 三个线程同时越过 → 制造真正的并发入口
        ok = client.start()
        with result_guard:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()

    assert entered.wait(10), "后台线程应已启动（必有且仅有一个 leader）"
    # 让另外两个调用者落位到 _start_lock / _ready 上再放行。若此处不放宽窗口，
    # 迟到的调用者会在 leader 清完 _starting 之后才进入，从而被误判为「新一轮启动」
    # —— 那测的就不是并发重入，而是线程调度延迟了。
    time.sleep(0.3)
    release.set()
    for thread in threads:
        thread.join(10)

    assert len(launches) == 1, f"并发 start() 只应启动一个后台线程，实际 {len(launches)} 个"
    assert results == [True, True, True], "所有并发调用者都应拿到同一个就绪结果"
    assert client.tool_index == {"search_documents": "fake"}, "路由表不应被重入清空"
    assert client.ready is False, "替身未建事件循环，故 running 为假、ready 亦为假"
