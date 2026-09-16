"""MCP Server 集合：把工具层从 Agent 进程中剥离出来。

三个 Server 与客户端的关系
--------------------------
::

    app/mcp_servers/chroma_server.py       search_documents       向量 + BM25 混合检索
    app/mcp_servers/filesystem_server.py   list_docs / read_file  文档清单与全文
    app/mcp_servers/search_server.py       web_search             （可选）外部检索
                │
                │ 各自独立进程，stdio 传输
                ▼
    app/mcp_client.py   进程内 MCPToolClient（后台事件循环 + 长驻会话）
                │
                ▼
    app/graph/nodes/retriever.py   retriever 节点改为 call_tool

为什么本包不在 ``__init__`` 里 import 子模块
------------------------------------------
每个 server 模块在导入时都会实例化一个 ``FastMCP`` 对象并注册工具。若在此处
做导入转发，任何 ``import app.mcp_servers`` 都会连带实例化全部 server —— 包括
只想取单个工具函数做单测的场景。保持空 ``__init__``，让导入路径显式到模块级。

Server 侧的两条硬约束（详见各模块头部注释）
------------------------------------------
1. 禁止 ``from __future__ import annotations`` —— 会让工具函数注解变成字符串，
   mcp 1.9.2 的 ``Tool.from_function`` 对注解直接调 ``issubclass()``，会抛
   ``TypeError: issubclass() arg 1 must be a class``。
2. 禁止向 stdout 输出 —— stdio 传输下 stdout 是 JSON-RPC 独占通道。
"""
