"""pytest 全局配置。

当前不含自定义 fixture —— 用例各自通过参数注入替身
（图节点支持 ``llm`` / ``supervisor_llm`` / ``verifier_llm`` / ``retriever``），
不依赖全局 fixture。
"""
