"""
Agent 核心 —— LangChain Agent

三个模式，各自独立：

  run_off()    → ChatOpenAI.astream()     → 纯 LLM 流式对话（逐 token）
  run_auto()   → create_agent() + messages → Agent 自主决定调工具
  run_forced() → search_vector_db + Agent  → 先检索，再 Agent

关键概念：
  - ChatOpenAI:  统一的 LLM 调用接口，兼容 OpenAI API 格式
  - create_agent: LangChain Agent API，内置 ReAct 循环，返回 CompiledStateGraph
  - stream_mode="messages": 按消息粒度流式，产出 (msg, metadata) 元组
     AIMessage(tool_calls) → ToolMessage → AIMessage(content) → ...
  - SystemMessage/HumanMessage/AIMessage/ToolMessage: 消息类型
"""
import os
import asyncio
from typing import AsyncGenerator, List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent

from tools import search_vector_db, query_knowledge_graph
from checkpoint_store import get_checkpointer, get_thread_config
from context_manager import should_compress, compress_history, estimate_messages_tokens

LOCAL_TOOLS = [search_vector_db, query_knowledge_graph]

# ========== 配置 ==========
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_TEMPERATURE = float(os.environ["LLM_TEMPERATURE"])

# ========== 系统提示词 ==========
SYSTEM_OFF = """你是威胁情报分析助手。请基于你的知识回答用户问题，如果不知道就明确说不知道。禁止编造信息。"""

SYSTEM_AUTO = """你是威胁情报分析助手。你必须使用以下工具检索威胁情报信息：

工具：
- search_vector_db: 本地威胁情报库。查询已知APT组织、历史攻击事件、恶意软件家族、漏洞分析
- query_knowledge_graph: 实体关联查询。查询APT组织的别名、所属国家、使用的工具等关系

严格规则：
1. 收到任何问题后，必须先调用 search_vector_db 检索本地库，严禁跳过检索直接回答
2. 涉及APT组织、恶意软件名称时，同时调用 query_knowledge_graph 查关联
3. 检索完后基于结果回答，引用时用文档标题（如"据《2024年勒索软件报告》显示"），不要引用内部编号
4. 检索不到就明确告知，禁止编造
5. 中文回答，300字以内"""

# SYSTEM_FORCED = """..."""  # 已废弃：前端不再使用强制检索模式
# 保留供参考


def create_llm(streaming: bool = True) -> ChatOpenAI:
    """ChatOpenAI 封装 OpenAI 兼容 API（本地 vLLM 就是 OpenAI 兼容接口）"""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
        streaming=streaming,
        max_tokens=3072,  # 8192 上下文内留足空间给 tool 结果
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def to_langchain_messages(history: list) -> list:
    """dict 格式的对话历史 → LangChain 消息列表"""
    out = []
    for m in history:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
    return out


async def _compress_checkpoint_state(checkpointer, config: dict, llm) -> bool:
    """检查并压缩 checkpoint 中的消息。返回 True 表示已压缩。"""
    try:
        tuple_ = await checkpointer.aget_tuple(config)
        if not tuple_:
            return False
        checkpoint = tuple_.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        if not messages:
            return False
        if not should_compress(messages):
            return False
        compressed = await compress_history(messages, llm)
        # 更新 checkpoint
        new_checkpoint = dict(checkpoint)
        new_checkpoint["channel_values"] = {**channel_values, "messages": compressed}
        await checkpointer.aput(config, new_checkpoint, {
            "source": "context_compress",
        })
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        return False


# =====================================================================
#  三个运行模式 —— 每个都是 async generator
#  产出事件：
#    {"type": "thinking", "lines": [...]}    工具调用过程
#    {"type": "token",   "text": "..."}      回答的每个字
#    {"type": "done",    "answer": "...", "thinking": [...]}  完成
# =====================================================================


async def run_off(history: list, user_input: str) -> AsyncGenerator[dict, None]:
    """
    模式一：关闭 RAG，纯 LLM 对话。
    不绑定工具，不创建 Agent，直接 stream 回答。
    超过 80% 上下文时自动压缩旧消息。
    """
    llm = create_llm(streaming=True)
    llm_no_stream = create_llm(streaming=False)

    msgs = to_langchain_messages(history)

    if should_compress(msgs):
        msgs = await compress_history(msgs, llm_no_stream)

    msgs = [SystemMessage(content=SYSTEM_OFF)] + msgs + [HumanMessage(content=user_input)]

    full = ""
    finish_reason = ""
    async for chunk in llm.astream(msgs):
        if chunk.content:
            full += chunk.content
            yield {"type": "token", "text": chunk.content}
        if hasattr(chunk, "response_metadata"):
            finish_reason = chunk.response_metadata.get("finish_reason", "")

    truncated = (finish_reason == "length")
    yield {"type": "done", "answer": full, "thinking": [], "truncated": truncated}


async def run_auto(history: list, user_input: str,
                   thread_id: str = None) -> AsyncGenerator[dict, None]:
    """
    模式二：Agent 自主决定是否、何时调用工具。
    thread_id 不为空时启用 LangGraph checkpoint 持久化。
    超过 80% 上下文时自动压缩 checkpoint 中的旧消息。
    """
    llm = create_llm(streaming=True)
    llm_no_stream = create_llm(streaming=False)
    checkpointer = await get_checkpointer() if thread_id else None
    config = get_thread_config(thread_id) if thread_id else None

    if checkpointer and config:
        await _compress_checkpoint_state(checkpointer, config, llm_no_stream)

    agent = create_agent(
        model=llm,
        tools=LOCAL_TOOLS,
        system_prompt=SYSTEM_AUTO,
        checkpointer=checkpointer,
    )

    thinking_lines = []
    answer = ""
    finish_reason = ""

    async for msg, _metadata in agent.astream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="messages",
        config=config,
    ):
        if isinstance(msg, AIMessage):
            # 工具调用决策（chunk 的 tool_calls 非空列表时进入）
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    if not name:
                        continue
                    # 新一轮工具调用 → 重置回答
                    answer = ""
                    args = tc.get("args", {})
                    thinking_lines.append(f"🔧 调用工具: **{name}**")
                    if args:
                        thinking_lines.append(f"   输入: `{str(args)[:200]}`")
                    yield {"type": "thinking", "lines": list(thinking_lines)}
            # 文本内容：逐 token 累加（AIMessageChunk 每个 chunk 一个 token）
            if msg.content and not msg.tool_calls:
                answer += msg.content
                yield {"type": "token", "text": msg.content}
            # 捕获 finish_reason
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                finish_reason = msg.response_metadata.get("finish_reason", "")

        elif isinstance(msg, ToolMessage):
            content = msg.content or ""
            thinking_lines.append(f"   返回: `{str(content)[:500]}{'......' if len(str(content)) > 500 else ''}`")
            yield {"type": "thinking", "lines": list(thinking_lines)}

    if not answer:
        answer = "*(未生成回答)*"

    truncated = (finish_reason == "length")
    yield {"type": "done", "answer": answer, "thinking": thinking_lines, "truncated": truncated}


# run_forced() 已废弃 — 前端不再使用，保留代码供参考
# 见 git history 或下方注释块
"""
async def run_forced(history: list, user_input: str,
                     thread_id: str = None) -> AsyncGenerator[dict, None]:
    ...
"""
