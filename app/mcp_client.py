"""MCP 客户端：进程内长驻的工具调用通道。

连接模型（本模块最重要的设计决策）
----------------------------------
同步的 retriever 节点要调用异步的 MCP 工具，有两条路：

===================================  ==============  ==========================
方案                                   每次调用耗时     说明
===================================  ==============  ==========================
每次调用新建 stdio 连接（spawn 子进程）   ~750 ms        实测；真实 server 还要
                                                      import chromadb，更慢
专用后台事件循环 + 每 Server 长驻会话      ~4 ms          启动时付一次 ~750 ms
===================================  ==============  ==========================

实测差距约 **150 倍**，且自纠正循环一轮问答最多触发 3 次检索，每次重建连接会把
延迟放大到秒级。因此本模块采用长驻模型：

::

    主线程（同步）                    后台线程（专用事件循环）
    ─────────────                    ──────────────────────
    call_tool(tool, args)  ──►       ClientSession.call_tool()  ──► Server 子进程
      run_coroutine_threadsafe             ▲
      .result(timeout)                     │ 长驻，跨调用复用
                                     _serve_one(server) 挂在 stop event 上

为什么不是「只把每次连接包进 to_thread」
--------------------------------------
那样每个调用都要付一次进程启动成本。而 LangGraph 目前是同步图（``invoke``），
把整图改成异步链路的收益远小于风险。

为什么健康状态用实时 ping 而不是缓存标记
--------------------------------------
子进程可能在运行中崩溃（例如 Chroma 目录被删）。缓存标记会一直显示 ok，而
``probe()`` 每次真实调用 ``list_tools()``（仅数毫秒），能反映真实状态。
这正是蓝图「关掉 chroma_server，/health 应报告 unhealthy」的落地方式。

启动等待为什么必须等「全部」Server 落定
--------------------------------------
``_ready`` 若在第一个 Server 就绪时就放行，``start()`` 会在其余 Server 仍在握手时
返回，于是「调用紧跟启动」的调用方会拿到 ``工具未注册`` —— 实测就是这样丢掉
了 filesystem 的 ``list_docs`` / ``read_file``。因此用 ``_pending`` 计数，
等所有 Server 完成握手（成功或失败）才放行。

启动为什么必须防并发重入
----------------------
阶段 5 引入 ``Send`` fan-out 后，N 个 retriever 分支会**同时**落到惰性启动路径。
而「是否已启动」不能用 ``running`` 判定：``running`` 依赖 ``_loop``，``_loop``
由后台线程内部创建，在握手完成前它一直是 False。于是每个并发调用者都判定
「尚未启动」、依次进入 ``_start_lock`` 重清 ``_tool_index`` 并另起一个事件循环，
``_pending`` 计数随之错乱 —— 实测 ``_ready`` 会在 chroma 注册
``search_documents`` **之前**就放行，复合问题整体降级为「无法确认」。

因此 ``start()`` 采用显式的 leader/follower：持锁期间的第一个调用者成为 leader
并真正启动，其余调用者只等**同一个** ``_ready``，绝不重入清理共享状态。
对外暴露 ``ready``（running **且** 握手已落定）作为「可以调用工具了」的唯一判据。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from app.core.config import PROJECT_ROOT, settings

SERVER_CHROMA = "chroma"
SERVER_FILESYSTEM = "filesystem"
SERVER_SEARCH = "search"

# 各 Server 的入口模块。用 `-m` 而非脚本路径：不依赖 editable 安装，
# 只要 cwd 在项目根即可（`-m` 会把 cwd 加入 sys.path）。
SERVER_MODULES: dict[str, str] = {
    SERVER_CHROMA: "app.mcp_servers.chroma_server",
    SERVER_FILESYSTEM: "app.mcp_servers.filesystem_server",
    SERVER_SEARCH: "app.mcp_servers.search_server",
}


# ============================== 异常处理 ==============================


def flatten_exception(exc: BaseException, depth: int = 0, max_depth: int = 8) -> list[str]:
    """把嵌套的 ExceptionGroup 展平成可读字符串列表。

    anyio 的 TaskGroup 会把错误包成 ``ExceptionGroup``；stdio 场景实测嵌套两层
    （``ExceptionGroup > ExceptionGroup > McpError: Connection closed``）。
    直接 ``str(exc)`` 只能得到最外层那句毫无信息量的 "unhandled errors in a
    TaskGroup"，必须递归展开才能看到真因。
    """
    lines: list[str] = []
    if depth > max_depth:
        return lines
    pad = "  " * depth
    if isinstance(exc, BaseExceptionGroup):
        subs = list(exc.exceptions)
        lines.append(f"{pad}{type(exc).__name__}({len(subs)}): {exc}")
        for sub in subs:
            lines.extend(flatten_exception(sub, depth + 1, max_depth))
    else:
        text = str(exc).strip()
        head = text.splitlines()[0] if text else ""
        lines.append(f"{pad}{type(exc).__name__}: {head}")
    return lines


def summarize_exception(exc: BaseException) -> str:
    """取展平后的叶子异常（信息量最大）作为单行摘要。"""
    lines = flatten_exception(exc)
    if not lines:
        return type(exc).__name__
    return lines[-1].strip()


# ============================== 结果契约 ==============================


@dataclass(slots=True)
class MCPToolResult:
    """一次工具调用的结果。

    刻意**不抛异常**：调用方（retriever 节点）需要区分「检索到 0 条」与
    「检索失败」—— 前者是内容问题、后者是可用性问题，降级策略不同。
    用 ``ok`` 标志表达比 try/except 更不容易写漏。
    """

    ok: bool
    server: str
    tool: str
    data: Any = None
    text: str = ""
    error: str = ""
    elapsed_ms: int = 0

    def as_items(self) -> list[dict]:
        """把 data 规范化为 ``list[dict]``。

        工具返回形态实测有三种：单个 JSON 对象 → dict；多个 JSON 对象
        （函数返回 list 时 FastMCP 拆成 N 个 TextContent）→ list[dict]；
        纯文本 → str。检索类工具只关心前两种。
        """
        value = self.data
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []


def parse_tool_content(content: Any) -> tuple[Any, str]:
    """把 ``CallToolResult.content`` 解析成 (data, text)。

    实测行为（mcp 1.9.2 + FastMCP）：
    - 工具返回 ``dict`` → content 有 1 个 TextContent，text 是 JSON
    - 工具返回 ``list`` → content 有 **N 个** TextContent，每个元素一个 JSON
    - 工具返回 ``str``  → content 有 1 个 TextContent，text 是**纯文本**（非 JSON）

    因此不能无条件 ``json.loads``，必须逐个容错。
    """
    texts: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(str(text))
    if not texts:
        return None, ""

    joined = "\n".join(texts)
    parsed: list[Any] = []
    for text in texts:
        try:
            parsed.append(json.loads(text))
        except (ValueError, TypeError):
            # 任一元素不是 JSON 就整体回退为文本，不做部分解析 ——
            # 半结构化结果比纯文本更难用，也更容易被误当成有效数据。
            return (texts[0] if len(texts) == 1 else texts), joined

    return (parsed[0] if len(parsed) == 1 else parsed), joined


# ============================== 连接配置 ==============================


def build_stdio_connection(module: str) -> Connection:
    """构造一条 stdio 连接配置。

    ``env`` 必须显式给出完整环境：mcp 的 stdio_client 在 ``env`` 非空时是
    **整体替换**子进程环境（不是合并），只传 PYTHONIOENCODING 会让子进程丢掉
    PATH 等变量。这里以 ``os.environ`` 为基底再叠加。
    """
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", module],
        "cwd": str(PROJECT_ROOT),
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
    }


def enabled_server_names() -> list[str]:
    """从配置解析启用的 Server 列表（顺序即配置顺序，去重）。"""
    raw = (settings.mcp_servers or "").strip()
    if not raw:
        return []
    names: list[str] = []
    for piece in raw.split(","):
        name = piece.strip().lower()
        if not name:
            continue
        if name not in SERVER_MODULES:
            continue
        if name not in names:
            names.append(name)
    return names


def default_connections() -> dict[str, Connection]:
    """按配置构造全部 stdio 连接。"""
    return {
        name: build_stdio_connection(SERVER_MODULES[name])
        for name in enabled_server_names()
    }


# ============================== 客户端 ==============================


class MCPToolClient:
    """进程内 MCP 客户端。

    生命周期：``start()`` → 多次 ``call_tool()`` → ``stop()``。
    ``start()`` **永不抛异常**：任一 Server 启动失败只记录状态，让服务仍能起来
    并以降级模式对外服务 —— 这正是阶段 4 的验收场景。
    """

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
        *,
        call_timeout: float | None = None,
        start_timeout: float | None = None,
        ping_timeout: float | None = None,
    ) -> None:
        self._connections: dict[str, Connection] = (
            default_connections() if connections is None else dict(connections)
        )
        self._call_timeout = call_timeout or settings.mcp_timeout
        self._start_timeout = start_timeout or settings.mcp_start_timeout
        self._ping_timeout = ping_timeout or settings.mcp_ping_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: MultiServerMCPClient | None = None

        self._sessions: dict[str, Any] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._tool_names: dict[str, list[str]] = {}
        self._tool_index: dict[str, str] = {}
        self._status: dict[str, str] = {name: "未启动" for name in self._connections}

        self._ready = threading.Event()
        self._pending = 0
        # 「有一次启动正在进行」。不能靠 running 代替：running 依赖 _loop，
        # 而 _loop 是后台线程内部创建的，启动尚未落定时它为 False（详见 start()）。
        self._starting = False
        # 保护 _sessions / _tool_index / _status / _pending 等跨线程共享状态。
        # 单次 dict 读写本身是原子的，但「先判断再赋值」这类复合操作不是。
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()

    # ---------------- 只读属性 ----------------

    @property
    def server_names(self) -> list[str]:
        return list(self._connections)

    @property
    def running(self) -> bool:
        return bool(
            self._loop is not None
            and self._loop.is_running()
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def ready(self) -> bool:
        """是否已「启动完成」：事件循环在跑**且**全部 Server 握手已落定。

        与 ``running`` 的区别只在时序。``running`` 依赖 ``_loop``，而 ``_loop``
        是后台线程内部创建的，线程一跑起来它就为真 —— 此刻各 Server 的工具可能
        还没进路由表（实测 chroma 比 filesystem 晚约 3 秒）。所以凡是「启动后马上
        调用工具」的代码路径都必须判 ``ready`` 而不是 ``running``，否则会拿到
        ``工具未注册`` 并把整轮检索降级掉。
        """
        return self.running and self._ready.is_set()

    @property
    def status_map(self) -> dict[str, str]:
        with self._lock:
            return dict(self._status)

    @property
    def tool_index(self) -> dict[str, str]:
        """工具名 → Server 名，启动时由 ``list_tools()`` 动态发现填充。"""
        with self._lock:
            return dict(self._tool_index)

    def tool_names(self, server: str | None = None) -> list[str]:
        with self._lock:
            if server is None:
                return [n for names in self._tool_names.values() for n in names]
            return list(self._tool_names.get(server, []))

    # ---------------- 生命周期 ----------------

    def _mark_settled(self, name: str) -> None:
        """标记一个 Server 已完成握手（成功或失败）。

        只有全部 Server 落定才放行 ``start()`` —— 否则「启动后立即调用」的调用方
        会拿到「工具未注册」，因为后完成的 Server 的工具还没进路由表。
        """
        with self._lock:
            self._pending -= 1
            remaining = self._pending
        if remaining <= 0:
            self._ready.set()

    def start(self) -> bool:
        """启动后台事件循环并与全部 Server 握手。

        并发语义（阶段 5 修复的缺陷）
        --------------------------
        本方法必须能在多线程下被**并发调用**：阶段 5 的 ``Send`` fan-out 会让
        N 个 retriever 分支同时落到惰性启动路径上。旧实现只判 ``running``，
        而 ``running`` 在后台线程建好 loop 之前恒为 False，于是每个并发调用者都
        判定「还没启动」、依次进入 ``_start_lock`` 把 ``_tool_index`` 清空并另起
        一个事件循环 —— 实测后果是路由表被反复清零、``_pending`` 计数错乱，
        ``_ready`` 在 chroma 注册 ``search_documents`` **之前**就放行，复合问题
        因此整体降级为「无法确认」。

        修法是显式的 leader/follower：持锁期间的第一个调用者成为 leader，置
        ``_starting`` 并真正启动；其后到达的调用者一律作为 follower，与 leader
        等待**同一个** ``_ready``。绝不允许重入清理共享状态。

        Returns:
            True 表示至少一个 Server 就绪。全部失败返回 False，但**不抛异常**
            —— 调用方据此继续启动服务。
        """
        if self.ready:
            return True
        if not self._connections:
            return False

        with self._start_lock:
            # running 已为真但尚未 ready，说明另一次启动仍在握手 —— 此时也必须
            # 当 follower，否则会另起一个事件循环（旧实现正是这样丢掉 chroma 的）。
            if self.running or self._starting:
                leader = False
            else:
                leader = True
                self._starting = True
                with self._lock:
                    self._sessions.clear()
                    self._tool_names.clear()
                    self._tool_index.clear()
                    self._status = {name: "启动中" for name in self._connections}
                    self._pending = len(self._connections)
                # 必须在持锁期内清 _ready：否则 follower 可能在 leader 清空之前
                # 就读到上一次启动遗留的「已置位」事件而立刻返回。
                self._ready.clear()

                self._thread = threading.Thread(
                    target=self._thread_main, name="mcp-client-loop", daemon=True
                )
                self._thread.start()

        if not self._ready.wait(self._start_timeout):
            if leader:
                # 交还 leadership，让后续调用可以重试，而不是永久卡在 follower 态
                with self._start_lock:
                    self._starting = False
            return False

        ok = [name for name, state in self.status_map.items() if state == "ok"]
        if leader:
            with self._start_lock:
                self._starting = False
        return bool(ok)

    def ensure_started(self) -> bool:
        """幂等启动，保证返回 True 时**全部 Server 已完成握手**。

        供没有 FastAPI lifespan 的入口使用（评估脚本、独立调用 ``build_graph()``）。
        没有它，这些入口拿到的是「未启动」的单例，检索会永远静默降级。

        判据用 ``ready`` 而不是 ``running``：后者在握手完成前就为真，会让
        「启动后立即检索」的调用方带着空路由表进去（见 ``ready`` 的说明）。
        并发安全完全交给 ``start()`` 的 leader/follower 机制，本方法不做额外加锁。
        """
        if self.ready:
            return True
        return self.start()

    def _thread_main(self) -> None:
        """后台线程主体：建 loop、跑长驻协程、退出时关闭 loop。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve_all())
        except BaseException:  # noqa: BLE001 - 后台线程不能把异常抛给解释器
            # 各 Server 的失败已在 _serve_one 里收敛进 status_map。此处只需保证
            # 线程不把异常抛给解释器，且不打断 finally 的事件循环收尾。
            pass
        finally:
            try:
                # 收尾未完成的异步生成器，避免 "Task was destroyed but it is pending"
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            loop.close()
            self._loop = None
            # 让仍在等待的 start() 不至于空等到超时
            self._ready.set()

    async def _serve_all(self) -> None:
        """并发维持全部 Server 的长驻会话，直到收到停止信号。"""
        self._client = MultiServerMCPClient(self._connections)
        await asyncio.gather(
            *(self._serve_one(name, self._client) for name in self._connections),
            return_exceptions=True,
        )

    async def _serve_one(self, name: str, client: MultiServerMCPClient) -> None:
        """单个 Server 的长驻协程：握手 → 发现工具 → 挂起等待停止。"""
        stop = asyncio.Event()
        with self._lock:
            self._stop_events[name] = stop
        settled = False
        try:
            async with client.session(name) as session:
                listed = await session.list_tools()
                names = [tool.name for tool in listed.tools]

                with self._lock:
                    self._sessions[name] = session
                    self._tool_names[name] = names
                    for tool_name in names:
                        # setdefault：先连上的 Server 取得路由权，冲突不覆盖
                        self._tool_index.setdefault(tool_name, name)
                    self._status[name] = "ok"

                # 工具已进路由表 → 本 Server 视为落定；之后挂起等停止信号
                self._mark_settled(name)
                settled = True
                await stop.wait()

        except BaseException as exc:  # noqa: BLE001 - 启动失败必须转成状态而非抛出
            reason = summarize_exception(exc)
            with self._lock:
                self._sessions.pop(name, None)
                self._status[name] = f"error: {reason}"
        finally:
            if not settled:
                self._mark_settled(name)

    def stop(self, timeout: float = 20.0) -> None:
        """停止长驻会话并回收后台线程。幂等。"""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return

        with self._lock:
            events = list(self._stop_events.values())
        for event in events:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # loop 已关闭：说明后台线程先一步退出，无需再通知
                break

        thread.join(timeout)
        self._thread = None
        with self._lock:
            self._stop_events.clear()
            self._sessions.clear()

    # ---------------- 工具调用 ----------------

    def _describe_unavailable(self) -> str:
        """给出「为什么不能调用」的精确原因。

        区分「从未启动」与「启动过但已退出」很重要：前者是配置/入口问题，
        后者是 Server 进程崩溃，运维动作完全不同。
        """
        failed = {
            name: state
            for name, state in self.status_map.items()
            if state.startswith("error")
        }
        if failed:
            return f"MCP 通道已退出；失败原因：{failed}"
        return "MCP 通道未启动"

    def call_tool(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        server: str | None = None,
        timeout: float | None = None,
    ) -> MCPToolResult:
        """同步调用工具。**永不抛异常**，失败信息放在 ``MCPToolResult.error``。"""
        started = time.perf_counter()
        target = server or self.tool_index.get(tool)

        def fail(message: str) -> MCPToolResult:
            return MCPToolResult(
                ok=False,
                server=target or "",
                tool=tool,
                error=message,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        if not self.running:
            return fail(self._describe_unavailable())
        if not target:
            with self._lock:
                no_session = not self._sessions
            if no_session:
                # 尚未建立任何会话（例如刚启动就失败、loop 正在收尾）：
                # 「通道为什么不可用」比「工具未注册」更能指向真因。
                return fail(self._describe_unavailable())
            return fail(f"工具 {tool!r} 未注册；当前可用工具：{sorted(self.tool_index)}")

        status = self.status_map.get(target, "")
        if status.startswith("error"):
            return fail(f"Server {target!r} 不可用（{status}）")

        with self._lock:
            session = self._sessions.get(target)
        if session is None:
            return fail(f"Server {target!r} 会话不存在")

        limit = timeout or self._call_timeout
        loop = self._loop
        if loop is None:
            return fail("MCP 事件循环已关闭")

        try:
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(tool, args or {}), loop
            )
        except RuntimeError as exc:
            return fail(f"提交调用失败：{exc}")

        try:
            raw = future.result(limit)
        except FuturesTimeout:
            future.cancel()
            return fail(f"调用 {tool!r} 超时（>{limit}s）")
        except BaseException as exc:  # noqa: BLE001 - 工具调用失败必须降级而非中断
            reason = summarize_exception(exc)
            with self._lock:
                self._status[target] = f"error: {reason}"
            return fail(f"调用 {tool!r} 失败：{reason}")

        elapsed = int((time.perf_counter() - started) * 1000)
        if getattr(raw, "isError", False):
            _, text = parse_tool_content(getattr(raw, "content", None))
            message = text.strip() or "工具返回 isError 但无内容"
            return MCPToolResult(
                ok=False,
                server=target,
                tool=tool,
                text=text,
                error=message,
                elapsed_ms=elapsed,
            )

        data, text = parse_tool_content(getattr(raw, "content", None))
        return MCPToolResult(
            ok=True,
            server=target,
            tool=tool,
            data=data,
            text=text,
            elapsed_ms=elapsed,
        )

    async def acall_tool(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        server: str | None = None,
        timeout: float | None = None,
    ) -> MCPToolResult:
        """异步调用工具。

        ``call_tool`` 会在 ``future.result()`` 上阻塞线程，直接在 async 上下文
        （如 FastAPI 路由）调用会卡住事件循环，故这里把阻塞转移到工作线程。
        """
        return await asyncio.to_thread(
            self.call_tool, tool, args, server=server, timeout=timeout
        )

    # ---------------- 健康检查 ----------------

    def probe(self, timeout: float | None = None) -> dict[str, str]:
        """逐个 Server 实时 ping，返回 ``{server: "ok" | "error: ..."}``。

        用 ``list_tools()`` 作为 ping：只往返一次 JSON-RPC，实测仅数毫秒，却能
        真实反映子进程是否还活着（缓存状态无法发现运行中崩溃）。
        """
        if not self.running:
            return {
                name: state if state.startswith("error") else "未启动"
                for name, state in self.status_map.items()
            }

        limit = timeout or self._ping_timeout
        loop = self._loop
        result: dict[str, str] = {}

        for name in self._connections:
            with self._lock:
                session = self._sessions.get(name)
            if session is None:
                result[name] = self.status_map.get(name, "未就绪")
                continue
            try:
                future = asyncio.run_coroutine_threadsafe(session.list_tools(), loop)  # type: ignore[arg-type]
                future.result(limit)
                result[name] = "ok"
            except FuturesTimeout:
                future.cancel()
                result[name] = f"error: ping 超时（>{limit}s）"
            except BaseException as exc:  # noqa: BLE001
                result[name] = f"error: {summarize_exception(exc)}"

            with self._lock:
                self._status[name] = result[name]

        return result


# ============================== 单例 ==============================


_default: MCPToolClient | None = None
_default_lock = threading.Lock()


def get_default_mcp() -> MCPToolClient:
    """进程内 MCP 客户端单例（**不自动启动**）。

    启动由调用方显式负责：FastAPI 走 lifespan，脚本走 ``ensure_started()``，
    retriever 节点在首次检索时惰性兜底。不在构造函数里自动启动，是为了让
    「构造」与「建立跨进程连接」这两件事在调用栈上可见 —— 后者有约 1 秒开销，
    藏在 getter 里会让启动耗时难以归因。
    """
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = MCPToolClient(default_connections())
    return _default


def reset_default_mcp() -> None:
    """关闭并丢弃单例。供测试与热重载使用。"""
    global _default
    with _default_lock:
        if _default is not None:
            _default.stop()
        _default = None


__all__ = [
    "MCPToolClient",
    "MCPToolResult",
    "SERVER_CHROMA",
    "SERVER_FILESYSTEM",
    "SERVER_MODULES",
    "SERVER_SEARCH",
    "build_stdio_connection",
    "default_connections",
    "enabled_server_names",
    "flatten_exception",
    "get_default_mcp",
    "parse_tool_content",
    "reset_default_mcp",
    "summarize_exception",
]
