"""
多 Agent 协同 — 手写 StateGraph（不用 create_agent）

  researcher_node → tool_node → researcher_node (最多2轮工具调用)
                             → web_search_node → synthesizer_node → END

流式事件：
  {"type": "thinking", "lines": [...]}   工具调用 / 搜索过程
  {"type": "token",   "text": "..."}     逐 token 回答
  {"type": "phase",   "phase": "..."}    阶段切换
  {"type": "done",    "answer": "..."}   完成

使用 event_queue（asyncio.Queue in state）实现真正的实时 token 流式输出。
graph.astream_events() 无法捕获自定义 async 函数节点内部的 on_chat_model_stream 事件，
因此改为节点内部通过 event_queue 推送事件，run_multi_agent 在后台运行 graph，
前台读取 event_queue。
"""
import os
import re
import asyncio
import contextvars
from typing import Annotated, TypedDict, Any

from dotenv import load_dotenv
load_dotenv()

# ContextVar 替代 state 中的 event_queue，避免 checkpointer 序列化 asyncio.Queue 时报错
_event_ctx: contextvars.ContextVar = contextvars.ContextVar("event_queue", default=None)

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from tools import search_vector_db, query_knowledge_graph, search_web
from checkpoint_store import get_checkpointer, get_thread_config

# ========== 配置 ==========
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_TEMPERATURE = float(os.environ["LLM_TEMPERATURE"])

LOCAL_TOOLS = [search_vector_db, query_knowledge_graph]
TOOLS_BY_NAME = {t.name: t for t in LOCAL_TOOLS}
MAX_TOOL_ROUNDS = 2
MAX_WEB_SEARCHES = 3


def _make_llm(streaming: bool = True, temperature: float = LLM_TEMPERATURE,
              max_tokens: int = 2048) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
        temperature=temperature, streaming=streaming, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ========== State ==========

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    local_answer: str
    local_tool_count: int
    thinking_log: list
    web_entities: list
    web_queries: list
    web_results_raw: list
    comparison: str
    error: str


# ========== Prompts ==========

RESEARCHER_SYSTEM = """你是威胁情报分析研究员。你只能用以下工具检索，禁止凭自己知识回答：

- search_vector_db: 本地威胁情报库
- query_knowledge_graph: 实体关联查询（APT组织别名、所属国家、工具等）

规则：
1. 先调工具检索，再回答。不调工具直接回答是违规
2. 引用时用文档标题（如"据《2024年勒索软件报告》显示"），不要引用内部编号
3. 不确定写"待验证"，不编造
4. 中文回答，300字以内"""

SYNTHESIZER_PROMPT = """你是威胁情报分析师。对比"本地库检索结果"和"联网搜索结果"，找出差异和补充。

【用户问题】
{question}

【本地库回答】
{local_answer}

【联网搜索结果】
{web_results}

输出格式（Markdown）：

## 🔍 联网核查结果

**双重确认**: （本地和联网都确认的，一行一条 "- xxx"。没有写"无"）

**🆕 Web 补充**: （联网有但本地没有的，每条带 [来源: xxx]。没有写"无"）

**⚠️ 差异**: 写清楚本地和联网不一致的部分（明确不一致的地方才写。没有写"无"。不要强行找差异！）

**总结**: （一句话，20字内）"""


# ========== 实体提取 ==========

_RE_APT = re.compile(r'\bAPT\d+', re.IGNORECASE)
_RE_CVE = re.compile(r'\bCVE-\d{4}-\d{4,}', re.IGNORECASE)
_RE_TOOL = re.compile(
    r'\b(Sunburst|SolarWinds|EnvyScout|ROOTSAW|WellMess|WellMail|'
    r'TEARDROP|RAINDROP|GoldMax|Sibot|'
    r'CosmicDuke|MiniDuke|OnionDuke|CozyDuke|SeaDuke|HammerDuke|'
    r'AppleJeus|Dtrack|WannaCry|DarkSeoul|Fallchill|BlindingCan|'
    r'Cobalt\s*Strike|Mimikatz|PlugX|Poison\s*Ivy|CrackMapExec|'
    r'Emotet|TrickBot|Conti|LockBit|BlackCat|REvil|Sodinokibi|DarkSide|'
    r'Stuxnet|Flame|Duqu|Equation|Regin|'
    r'Dridex|Gozi|Ursnif|QakBot|IcedID|BazarLoader|BumbleBee|'
    r'BloodHound|Empire|PowerSploit|Rubeus|Sliver|Metasploit)',
    re.IGNORECASE,
)


def _extract_key_terms(text: str) -> list[str]:
    found: list[str] = []
    seen = set()
    for pat in (_RE_APT, _RE_CVE, _RE_TOOL):
        for m in pat.finditer(text):
            raw = m.group(0).strip()
            key = raw.lower()
            if key not in seen:
                seen.add(key)
                found.append(raw)
    return found[:6]


def _build_search_queries(question: str, entities: list[str]) -> list[str]:
    queries = []
    q_lower = question.lower()
    if any(w in q_lower for w in ("漏洞", "cve", "exploit")):
        context = "vulnerability details impact"
    elif any(w in q_lower for w in ("攻击手法", "ttp", "technique", "攻击链")):
        context = "attack techniques TTPs"
    elif any(w in q_lower for w in ("区别", "对比", "异同", "comparison", "vs")):
        context = "comparison analysis"
    elif any(w in q_lower for w in ("恶意软件", "malware", "工具", "tool")):
        context = "malware tools"
    elif any(w in q_lower for w in ("电信", "telecom", "能源", "金融", "医疗")):
        context = "cyber attack campaign"
    else:
        context = "threat intelligence report"

    for ent in entities:
        if len(queries) >= MAX_WEB_SEARCHES:
            break
        q = f"{ent} {context}"
        if q not in queries:
            queries.append(q)

    if len(queries) < 2:
        short_q = question[:120]
        if short_q not in queries:
            queries.append(short_q)
    if len(queries) < 2 and len(entities) >= 2:
        q = f"{entities[0]} {entities[1]} {context}"
        if q not in queries:
            queries.append(q)

    return queries[:MAX_WEB_SEARCHES]


# ========== Graph Nodes ==========

async def researcher_node(state: MultiAgentState) -> dict:
    """调用 LLM + LOCAL_TOOLS，通过 event_queue 推送 token 和工具调用事件"""
    llm = _make_llm(streaming=True, max_tokens=1024)
    llm_with_tools = llm.bind_tools(LOCAL_TOOLS)
    msgs = [SystemMessage(content=RESEARCHER_SYSTEM)] + state["messages"]

    q = _event_ctx.get()

    merged = None
    async for chunk in llm_with_tools.astream(msgs):
        if merged is None:
            merged = chunk
        else:
            merged = merged + chunk
        text = chunk.content
        if q is not None and text:
            q.put_nowait({"type": "token", "text": text})
            await asyncio.sleep(0)  # yield control so main loop can read queue

    response = merged
    new_count = state.get("local_tool_count", 0) + 1

    if q is not None and isinstance(response, AIMessage) and response.tool_calls:
        for tc in response.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            q.put_nowait({"type": "event", "event": {
                "type": "thinking",
                "lines": [f"🔧 [检索] 调用工具: **{name}**"],
            }})
            if args:
                q.put_nowait({"type": "event", "event": {
                    "type": "thinking",
                    "lines": [f"   输入: `{str(args)[:200]}`"],
                }})

    result = {
        "messages": [response],
        "local_tool_count": new_count,
    }
    if isinstance(response, AIMessage) and response.content and not response.tool_calls:
        result["local_answer"] = response.content

    return result


async def tool_node(state: MultiAgentState) -> dict:
    """执行工具调用，通过 event_queue 推送结果"""
    q = _event_ctx.get()
    last_msg = state["messages"][-1]

    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    results = []
    for tc in last_msg.tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool = TOOLS_BY_NAME.get(tool_name)

        if tool:
            try:
                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(None, tool.invoke, tool_args)
            except Exception as e:
                raw = f"工具执行失败: {e}"
        else:
            raw = f"未知工具: {tool_name}"

        result_str = str(raw)
        results.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

        if q is not None:
            q.put_nowait({"type": "event", "event": {
                "type": "thinking",
                "lines": [f"   返回: `{result_str[:500]}{'......' if len(result_str) > 500 else ''}`"],
            }})

    return {"messages": results}


async def _run_one_search(sq: str) -> tuple[str, str]:
    """执行单次搜索，返回 (查询词, 结果文本)"""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, search_web.invoke, sq)
        return (sq, str(result))
    except Exception as e:
        return (sq, f"搜索失败: {e}")


async def web_search_node(state: MultiAgentState) -> dict:
    """提取实体 → 生成搜索词 → 并行搜索 → 通过 event_queue 推送进度"""
    q = _event_ctx.get()
    query = state["query"]
    entities = _extract_key_terms(query)
    search_queries = _build_search_queries(query, entities)

    if q is not None:
        q.put_nowait({"type": "event", "event": {
            "type": "thinking",
            "lines": ["---", f"🌐 正在联网搜索补充信息... ({len(search_queries)} 个搜索并行)"],
        }})
        q.put_nowait({"type": "event", "event": {"type": "phase", "phase": "verifying"}})

    if not search_queries:
        return {
            "web_entities": entities,
            "web_queries": [],
            "web_results_raw": [],
        }

    # 并行搜索（比串行快 2-3 倍）
    tasks = [_run_one_search(sq) for sq in search_queries]
    raw = await asyncio.gather(*tasks)

    web_results = []
    for sq, result in raw:
        web_results.append(result)
        if q is not None:
            q.put_nowait({"type": "event", "event": {
                "type": "thinking",
                "lines": [f"🔧 [搜索] `{sq[:80]}` → `{result[:500]}{'......' if len(result) > 500 else ''}`"],
            }})

    return {
        "web_entities": entities,
        "web_queries": search_queries,
        "web_results_raw": web_results,
    }


async def forced_answer_node(state: MultiAgentState) -> dict:
    """工具轮次用完后，强制 LLM 无工具生成文本回答"""
    q = _event_ctx.get()
    llm = _make_llm(streaming=True, max_tokens=1024)
    # 不绑定工具 — 强制 LLM 生成文本
    msgs = [SystemMessage(content=RESEARCHER_SYSTEM + "\n\n你已经调用了最大次数的工具，现在必须基于已有检索结果给出回答。禁止再调用工具。")] + state["messages"]

    merged = None
    async for chunk in llm.astream(msgs):
        if merged is None:
            merged = chunk
        else:
            merged = merged + chunk
        text = chunk.content
        if q is not None and text:
            q.put_nowait({"type": "token", "text": text})
            await asyncio.sleep(0)

    response = merged
    result = {"messages": [response]}
    if isinstance(response, AIMessage) and response.content:
        result["local_answer"] = response.content
    return result


async def synthesizer_node(state: MultiAgentState) -> dict:
    """流式 LLM 调用：对比本地 vs 联网，实时推送 token"""
    q = _event_ctx.get()

    if q is not None:
        q.put_nowait({"type": "event", "event": {
            "type": "thinking",
            "lines": ["🔎 正在对比分析..."],
        }})

    combined = "\n---\n".join(
        f"[搜索 {i+1}] {q_text}\n{r}"
        for i, (q_text, r) in enumerate(zip(state["web_queries"], state["web_results_raw"]))
    )

    prompt = SYNTHESIZER_PROMPT.format(
        question=state["query"],
        local_answer=state.get("local_answer", "")[:3000],
        web_results=combined[:4000],
    )

    llm = _make_llm(streaming=True, temperature=0.1, max_tokens=512)
    merged = None
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if merged is None:
            merged = chunk
        else:
            merged = merged + chunk
        if q is not None and chunk.content:
            q.put_nowait({"type": "token", "text": chunk.content})
            await asyncio.sleep(0)

    comparison = merged.content if hasattr(merged, "content") else str(merged)

    if q is not None:
        q.put_nowait({"type": "event", "event": {
            "type": "thinking",
            "lines": ["   ✅ 对比完成"],
        }})
        q.put_nowait({"type": "event", "event": {"type": "phase", "phase": "verify_pass"}})

    return {"comparison": comparison}


# ========== Router ==========

def route_after_researcher(state: MultiAgentState) -> str:
    """决定 researcher 之后走哪里"""
    if state.get("error"):
        return "fallback"

    last_msg = state["messages"][-1]
    tool_count = state.get("local_tool_count", 0)

    # 有 tool_calls 且未达上限 → 去执行工具
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        if tool_count < MAX_TOOL_ROUNDS:
            return "tools"
        # 达到上限但还没生成文本 → 强制回答（不绑工具）
        return "forced_answer"

    # 无 tool_calls（文本回答）→ 进入 web_search
    return "web_search"


def route_after_tools(state: MultiAgentState) -> str:
    """工具执行后回到 researcher"""
    if state.get("error"):
        return "fallback"
    return "researcher"


# ========== Build Graph ==========

def build_graph(checkpointer=None):
    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("tools", tool_node)
    graph.add_node("forced_answer", forced_answer_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("researcher")

    graph.add_conditional_edges("researcher", route_after_researcher, {
        "tools": "tools",
        "web_search": "web_search",
        "forced_answer": "forced_answer",
        "fallback": END,
    })
    graph.add_conditional_edges("tools", route_after_tools, {
        "researcher": "researcher",
        "fallback": END,
    })
    graph.add_edge("forced_answer", "web_search")
    graph.add_edge("web_search", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


# ========== 对外接口：流式事件生成器 ==========

async def run_multi_agent(query: str, thread_id: str = None) -> "AsyncGenerator[dict, None]":
    """graph 后台运行 → event_queue 前台读取，实现真正的实时流式输出"""
    checkpointer = await get_checkpointer() if thread_id else None
    config = get_thread_config(thread_id) if thread_id else None

    graph = build_graph(checkpointer=checkpointer)
    event_queue: asyncio.Queue = asyncio.Queue()
    _event_ctx.set(event_queue)

    multi_think = ["🤖 **检索 Agent** 开始工作（本地工具：向量库 + 知识图谱）"]
    yield {"type": "thinking", "lines": list(multi_think)}

    local_answer = ""
    comparison = ""
    has_error = False

    async def run_graph():
        try:
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=query)],
                "query": query,
            }, config=config)
            return result
        except Exception:
            import traceback, sys
            tb = traceback.format_exc()
            print(f"\n[MULTI_AGENT ERROR] {tb}\n", file=sys.stderr, flush=True)
            event_queue.put_nowait({"type": "event", "event": {
                "type": "thinking",
                "lines": [f"⚠️ 异常: {tb[:2000]}"],
            }})
            return {"error": tb}
        finally:
            event_queue.put_nowait(None)  # sentinel — 通知主循环结束

    runner_task = asyncio.create_task(run_graph())

    # 主循环：阻塞读取 event_queue，直到 sentinel
    while True:
        item = await event_queue.get()
        if item is None:
            break

        item_type = item["type"]

        if item_type == "token":
            local_answer += item["text"]
            yield item

        elif item_type == "event":
            ev = item["event"]
            # 累积所有 thinking 行（各节点只发增量），避免前端被覆盖
            if ev.get("type") == "thinking":
                multi_think.extend(ev["lines"])
                ev = {**ev, "lines": list(multi_think)}
            yield ev

    # 等待 graph 完成并获取结果
    final_state = await runner_task

    if isinstance(final_state, dict):
        if final_state.get("error"):
            has_error = True
            local_answer = final_state.get("local_answer", "") or local_answer
        else:
            comparison = final_state.get("comparison", "")
            if not local_answer:
                local_answer = final_state.get("local_answer", "")

    # ===== 组装最终回答 =====
    if not local_answer:
        answer = "*(未生成回答)*"
    else:
        answer = local_answer

    if comparison:
        answer += f"\n\n{comparison}"
    elif has_error:
        answer += "\n\n---\n⚠️ 联网核查未能完成"

    if not has_error:
        answer += "\n\n✅ 输出完成"

    yield {"type": "done", "answer": answer, "thinking": multi_think}
